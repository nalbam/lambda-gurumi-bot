"""Tests for src.tools.slack."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tests.tools._helpers import _ctx, _settings, _streamed_read
from src.llms import ToolCall
from src.tools.registry import ToolContext, ToolExecutor
from src.tools.slack import (
    fetch_thread_history,
    read_attached_document,
    read_attached_images,
)


# --------------------------------------------------------------------------- #
# read_attached_images SSRF guard
# --------------------------------------------------------------------------- #


def test_read_attached_images_rejects_non_slack_host():
    event = {"files": [{"mimetype": "image/png", "url_private_download": "https://evil.example.com/x.png"}]}
    with pytest.raises(ValueError):
        read_attached_images(_ctx(event=event), limit=1)


def test_read_attached_images_rejects_http_scheme():
    event = {"files": [{"mimetype": "image/png", "url_private_download": "http://files.slack.com/x.png"}]}
    with pytest.raises(ValueError):
        read_attached_images(_ctx(event=event), limit=1)


def test_read_attached_images_accepts_slack_host_variants():
    event = {
        "files": [
            {"mimetype": "image/png", "url_private_download": "https://files-pri.slack.com/x.png", "name": "a"},
        ]
    }
    llm = MagicMock()
    llm.describe_image.return_value = "a cat"
    ctx = _ctx(event=event, llm=llm)
    with patch("src.tools.slack.urllib.request.urlopen") as opener:
        opener.return_value.__enter__.return_value.read.return_value = b"fake"
        result = read_attached_images(ctx, limit=1)
    assert result == [{"name": "a", "summary": "a cat"}]


def test_read_attached_images_skips_non_image_mimetypes():
    event = {"files": [{"mimetype": "application/pdf", "url_private_download": "https://files.slack.com/x.pdf"}]}
    assert read_attached_images(_ctx(event=event), limit=1) == []


# --------------------------------------------------------------------------- #
# fetch_thread_history
# --------------------------------------------------------------------------- #


def test_fetch_thread_history_resolves_user_files_and_reactions():
    """History should carry display names, file metadata, and reactions so the
    LLM can answer things like "누가 좋아요 눌렀어?" or "아까 그 이미지 분석해줘"."""
    from src.slack_helpers import user_name_cache

    # Reset the module-level cache so prior tests don't leak.
    user_name_cache._cache.clear()

    client = MagicMock()
    client.conversations_replies.return_value = {
        "messages": [
            {
                "user": "U1",
                "text": "look at this",
                "ts": "1713.1",
                "files": [
                    {
                        "name": "cat.png",
                        "mimetype": "image/png",
                        "url_private_download": "https://files.slack.com/x/cat.png",
                        "permalink": "https://slack/p1",
                        "title": "cute",
                    }
                ],
            },
            {
                "user": "U2",
                "text": "nice!",
                "ts": "1713.2",
                "reactions": [
                    {"name": "thumbsup", "count": 2, "users": ["U1", "U3"]},
                ],
            },
        ]
    }

    def _users_info(user):
        return {"user": {"profile": {"display_name": f"name-{user}"}}}

    client.users_info.side_effect = _users_info

    out = fetch_thread_history(_ctx(slack_client=client), limit=5)
    assert len(out) == 2
    first, second = out
    assert first["user"] == "name-U1"
    assert first["text"] == "look at this"
    assert first["ts"] == "1713.1"
    assert first["files"] == [
        {
            "name": "cat.png",
            "mimetype": "image/png",
            "url_private_download": "https://files.slack.com/x/cat.png",
            "permalink": "https://slack/p1",
            "title": "cute",
        }
    ]
    assert first["reactions"] == []

    assert second["user"] == "name-U2"
    assert second["files"] == []
    assert second["reactions"] == [
        {"emoji": "thumbsup", "count": 2, "users": ["name-U1", "name-U3"]}
    ]


def test_read_attached_images_accepts_extra_urls():
    """Images referenced from fetch_thread_history (url_private_download) must
    be loadable via read_attached_images(urls=[...])."""
    ctx = _ctx()
    ctx.llm.describe_image.return_value = "a cat history"
    with patch("src.tools.slack.urllib.request.urlopen") as opener:
        opener.return_value.__enter__.return_value.read.return_value = b"fake-bytes"
        out = read_attached_images(
            ctx,
            limit=5,
            urls=["https://files.slack.com/x/cat.png"],
        )
    assert out == [{"name": "cat.png", "summary": "a cat history"}]


def test_read_attached_images_urls_reject_non_slack_host():
    ctx = _ctx()
    with pytest.raises(ValueError):
        read_attached_images(ctx, urls=["https://evil.example.com/cat.png"])


def test_read_attached_images_respects_total_limit_across_event_and_urls():
    event = {
        "files": [
            {
                "mimetype": "image/png",
                "url_private_download": "https://files.slack.com/e1.png",
                "name": "e1.png",
            }
        ]
    }
    ctx = _ctx(event=event)
    ctx.llm.describe_image.return_value = "desc"
    with patch("src.tools.slack.urllib.request.urlopen") as opener:
        opener.return_value.__enter__.return_value.read.return_value = b"x"
        out = read_attached_images(
            ctx,
            limit=2,
            urls=[
                "https://files.slack.com/u1.png",
                "https://files.slack.com/u2.png",  # should be skipped (limit=2)
            ],
        )
    assert len(out) == 2
    assert {item["name"] for item in out} == {"e1.png", "u1.png"}


# --------------------------------------------------------------------------- #
# read_attached_document
# --------------------------------------------------------------------------- #


def test_read_attached_document_text_file():
    event = {
        "files": [
            {
                "mimetype": "text/plain",
                "url_private_download": "https://files.slack.com/notes.txt",
                "name": "notes.txt",
            }
        ]
    }
    ctx = _ctx(event=event)
    body = b"Hello\n  world.\nLine 3."
    with patch("src.tools.slack.urllib.request.urlopen") as opener:
        resp = opener.return_value.__enter__.return_value
        resp.read.return_value = body
        resp.headers = {"Content-Length": str(len(body))}
        out = read_attached_document(ctx, limit=1)
    assert len(out) == 1
    entry = out[0]
    assert entry["name"] == "notes.txt"
    assert entry["mimetype"] == "text/plain"
    assert entry["truncated"] is False
    assert "Hello" in entry["text"]
    assert entry["chars"] == len(entry["text"])
    assert entry["pages"] == 0  # text files report 0 pages


def _build_pdf_bytes(pages_text: list[str]) -> bytes:
    """Build a minimal PDF (one page per string) using reportlab. Test-only."""
    from io import BytesIO
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.pagesizes import letter

    buf = BytesIO()
    canvas = Canvas(buf, pagesize=letter)
    for text in pages_text:
        canvas.drawString(72, 720, text)
        canvas.showPage()
    canvas.save()
    return buf.getvalue()


def _mock_pdf_response(opener, body: bytes, headers=None):
    """Wire the urlopen mock to stream `body` in chunks through `_fetch_slack_file`."""
    resp = opener.return_value.__enter__.return_value
    buf = {"pos": 0}

    def _chunked(n=-1):
        if n == -1:
            remaining = body[buf["pos"]:]
            buf["pos"] = len(body)
            return remaining
        chunk = body[buf["pos"]:buf["pos"] + n]
        buf["pos"] += len(chunk)
        return chunk

    resp.read.side_effect = _chunked
    resp.headers = dict(headers or {"Content-Length": str(len(body)), "Content-Type": "application/pdf"})


def test_read_attached_document_pdf_happy_path():
    pdf = _build_pdf_bytes(["Hello PDF page one.", "Page two here."])
    event = {
        "files": [
            {
                "mimetype": "application/pdf",
                "url_private_download": "https://files.slack.com/report.pdf",
                "name": "report.pdf",
            }
        ]
    }
    ctx = _ctx(event=event)
    with patch("src.tools.slack.urllib.request.urlopen") as opener:
        _mock_pdf_response(opener, pdf)
        out = read_attached_document(ctx, limit=1)
    assert len(out) == 1
    entry = out[0]
    assert entry["name"] == "report.pdf"
    assert entry["pages"] == 2
    assert entry["truncated"] is False
    assert entry["chars"] > 0


def test_read_attached_document_pdf_truncation():
    pdf = _build_pdf_bytes(["A" * 500])
    event = {
        "files": [
            {
                "mimetype": "application/pdf",
                "url_private_download": "https://files.slack.com/big.pdf",
                "name": "big.pdf",
            }
        ]
    }
    ctx = _ctx(
        event=event,
    )
    ctx = ToolContext(
        slack_client=ctx.slack_client,
        channel=ctx.channel,
        thread_ts=ctx.thread_ts,
        event=ctx.event,
        settings=_settings(max_doc_chars=50),
        llm=ctx.llm,
    )
    with patch("src.tools.slack.urllib.request.urlopen") as opener:
        _mock_pdf_response(opener, pdf)
        out = read_attached_document(ctx, limit=1)
    assert out[0]["truncated"] is True
    assert out[0]["chars"] == 50


def test_read_attached_document_page_cap():
    pdf = _build_pdf_bytes(["p1", "p2", "p3"])
    event = {
        "files": [
            {
                "mimetype": "application/pdf",
                "url_private_download": "https://files.slack.com/pages.pdf",
                "name": "pages.pdf",
            }
        ]
    }
    ctx = ToolContext(
        slack_client=MagicMock(),
        channel="C1",
        thread_ts="ts1",
        event=event,
        settings=_settings(max_doc_pages=2),
        llm=MagicMock(),
    )
    with patch("src.tools.slack.urllib.request.urlopen") as opener:
        _mock_pdf_response(opener, pdf)
        out = read_attached_document(ctx, limit=1)
    assert "error" in out[0]
    assert "MAX_DOC_PAGES" in out[0]["error"]


def test_read_attached_document_size_cap_via_content_length():
    event = {
        "files": [
            {
                "mimetype": "text/plain",
                "url_private_download": "https://files.slack.com/huge.txt",
                "name": "huge.txt",
            }
        ]
    }
    ctx = ToolContext(
        slack_client=MagicMock(),
        channel="C1",
        thread_ts="ts1",
        event=event,
        settings=_settings(max_doc_bytes=100),  # tiny cap
        llm=MagicMock(),
    )
    with patch("src.tools.slack.urllib.request.urlopen") as opener:
        resp = opener.return_value.__enter__.return_value
        resp.headers = {"Content-Length": "200"}  # > cap
        resp.read.return_value = b"x" * 10  # should never be read past cap
        out = read_attached_document(ctx, limit=1)
    assert "error" in out[0]
    assert "MAX_DOC_BYTES" in out[0]["error"]


def test_read_attached_document_size_cap_via_streamed_read():
    event = {
        "files": [
            {
                "mimetype": "text/plain",
                "url_private_download": "https://files.slack.com/nohead.txt",
                "name": "nohead.txt",
            }
        ]
    }
    ctx = ToolContext(
        slack_client=MagicMock(),
        channel="C1",
        thread_ts="ts1",
        event=event,
        settings=_settings(max_doc_bytes=100),
        llm=MagicMock(),
    )
    body = b"y" * 200
    with patch("src.tools.slack.urllib.request.urlopen") as opener:
        resp = opener.return_value.__enter__.return_value
        resp.headers = {}  # no Content-Length
        buf = {"pos": 0}

        def _chunked(n=-1):
            if n == -1:
                remaining = body[buf["pos"]:]
                buf["pos"] = len(body)
                return remaining
            chunk = body[buf["pos"]:buf["pos"] + n]
            buf["pos"] += len(chunk)
            return chunk

        resp.read.side_effect = _chunked
        out = read_attached_document(ctx, limit=1)
    assert "error" in out[0]
    assert "MAX_DOC_BYTES" in out[0]["error"]


def test_read_attached_document_rejects_non_slack_host():
    ctx = _ctx()
    out = read_attached_document(
        ctx, urls=["https://evil.example.com/foo.pdf"], limit=1
    )
    assert len(out) == 1
    assert "error" in out[0]
    assert "invalid" in out[0]["error"].lower()


def test_read_attached_document_skips_encrypted_pdf():
    from io import BytesIO
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    # NOTE: pypdf>=4.0 uses keyword-only user_password. If requirements.txt's
    # upper pin is ever relaxed past 6.0, verify this signature still holds.
    writer.encrypt(user_password="secret")
    buf = BytesIO()
    writer.write(buf)
    encrypted_pdf = buf.getvalue()

    event = {
        "files": [
            {
                "mimetype": "application/pdf",
                "url_private_download": "https://files.slack.com/enc.pdf",
                "name": "enc.pdf",
            }
        ]
    }
    ctx = _ctx(event=event)
    with patch("src.tools.slack.urllib.request.urlopen") as opener:
        _mock_pdf_response(opener, encrypted_pdf)
        out = read_attached_document(ctx, limit=1)
    assert "error" in out[0]
    assert "encrypted" in out[0]["error"]


def test_read_attached_document_skips_image_mime():
    event = {
        "files": [
            {
                "mimetype": "image/png",
                "url_private_download": "https://files.slack.com/a.png",
                "name": "a.png",
            }
        ]
    }
    ctx = _ctx(event=event)
    # urlopen should NOT be called — image MIMEs are filtered before fetch
    with patch("src.tools.slack.urllib.request.urlopen") as opener:
        out = read_attached_document(ctx, limit=1)
    opener.assert_not_called()
    assert out == []


def test_read_attached_document_http_error_returns_per_item():
    import urllib.error

    event = {
        "files": [
            {
                "mimetype": "application/pdf",
                "url_private_download": "https://files.slack.com/missing.pdf",
                "name": "missing.pdf",
            }
        ]
    }
    ctx = _ctx(event=event)
    with patch("src.tools.slack.urllib.request.urlopen") as opener:
        opener.side_effect = urllib.error.HTTPError(
            url="https://files.slack.com/missing.pdf",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )
        out = read_attached_document(ctx, limit=1)
    assert len(out) == 1
    assert "error" in out[0]
    assert "404" in out[0]["error"]
