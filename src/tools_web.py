"""Web-fetch helpers used by the fetch_webpage tool.

Split out from src/tools.py to keep each module focused. Symbols defined
here are re-exported from src.tools for backward compatibility with test
imports and callers.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser


_PUBLIC_WEB_UA = "lambda-gurumi-bot/1.0 (+https://github.com/nalbam/lambda-gurumi-bot)"
_WEB_FETCH_TIMEOUT = 12  # Jina Reader + direct raw GET; capped shorter than Slack internal fetch (15s) because web targets have no retry budget inside the tool.


def _validate_public_https_url(url: str) -> tuple[str, str]:
    """Return (scheme, hostname) after asserting the URL is safe to fetch.

    Rules:
      - scheme == 'https'
      - hostname is present and is NOT an IP literal
      - every address returned by getaddrinfo for the hostname is a public,
        routable unicast address (not private / loopback / link-local /
        reserved / multicast / unspecified).
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("fetch_webpage requires https")
    host = parsed.hostname
    if not host:
        raise ValueError("URL missing hostname")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("IP literals not allowed")
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"DNS resolution failed: {exc}") from exc
    if not infos:
        raise ValueError("DNS resolution returned no addresses")
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
            or addr.is_unspecified
            or not addr.is_global
        ):
            raise ValueError("hostname resolves to non-public address")
    return parsed.scheme, host


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """urllib handler that refuses to follow redirects.

    Raw fetches must hit exactly the host whose DNS we pre-validated. A 3xx
    pointing at a private host would silently defeat the SSRF guard.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, "redirects not allowed", headers, fp
        )


_JINA_LINK_RE = re.compile(
    r"(?<!!)\[([^\]]+)\]\((https?://[^\s()]*(?:\([^\s)]*\))?[^\s)]*)\)"
)
_JINA_HEADER_MAX_LINES = 10


class _HtmlTextExtractor(HTMLParser):
    """Streams visible text + <a> links out of raw HTML.

    Skips content inside script/style/noscript/template. Breaks paragraphs
    on block-level end tags. Link hrefs are resolved against ``base_url``;
    filtering (https-only, dedup, self-ref drop) is left to ``_filter_links``.
    """

    _SKIP_TAGS = {"script", "style", "noscript", "template"}
    _BREAK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr"}

    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self._base_url = base_url
        self._skip_depth = 0
        self._text_chunks: list[str] = []
        self._title_chunks: list[str] = []
        self._in_title = False
        self._current_link_url: str | None = None
        self._current_link_text: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag == "a":
            # Nested <a>: the inner link overwrites the outer one. Matches
            # how HTML5 parsers implicitly close an unclosed <a>; the outer
            # link is lost but the inner (usually more specific) is kept.
            href = dict(attrs).get("href")
            if href:
                self._current_link_url = urllib.parse.urljoin(self._base_url, href)
                self._current_link_text = []

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
            return
        if tag == "a" and self._current_link_url is not None:
            text = " ".join("".join(self._current_link_text).split())
            self.links.append((text, self._current_link_url))
            self._current_link_url = None
            self._current_link_text = []
            return
        if tag in self._BREAK_TAGS:
            self._text_chunks.append("\n")

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        if self._in_title:
            self._title_chunks.append(data)
            return
        if self._current_link_url is not None:
            self._current_link_text.append(data)
        self._text_chunks.append(data)

    def title(self) -> str:
        return " ".join("".join(self._title_chunks).split())

    def text(self) -> str:
        joined = "".join(self._text_chunks)
        lines = [" ".join(line.split()) for line in joined.split("\n")]
        return "\n".join(line for line in lines if line)


def _filter_links(
    raw: list[tuple[str, str]], base_url: str, limit: int
) -> list[dict[str, str]]:
    if limit <= 0:
        return []
    base_parsed = urllib.parse.urlparse(base_url)
    base_key = base_parsed._replace(
        scheme=base_parsed.scheme.lower(),
        netloc=base_parsed.netloc.lower(),
        fragment="",
    ).geturl()
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for text, url in raw:
        if not url.startswith("https://"):
            continue
        parsed = urllib.parse.urlparse(url)
        if not parsed.netloc:
            continue
        # Normalize host case for comparison (scheme+host are case-insensitive)
        normalized = parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=parsed.netloc.lower(),
            fragment="",
        ).geturl()
        if normalized == base_key:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append({"title": (text or url).strip(), "url": url})
        if len(out) >= limit:
            break
    return out


def _extract_markdown_links(
    md: str, base_url: str, limit: int
) -> list[dict[str, str]]:
    raw = [(m.group(1), m.group(2)) for m in _JINA_LINK_RE.finditer(md)]
    return _filter_links(raw, base_url, limit)


def _read_body_capped(response, max_bytes: int) -> bytes:
    content_length = response.headers.get("Content-Length") if response.headers else None
    if content_length and content_length.isdigit() and int(content_length) > max_bytes:
        raise ValueError(f"webpage exceeds MAX_WEB_BYTES={max_bytes}")
    body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise ValueError(f"webpage exceeds MAX_WEB_BYTES={max_bytes}")
    return body


def _jina_fetch(base: str, target_url: str, max_bytes: int) -> str:
    """Call Jina Reader and return the markdown body as text.

    Size gate is identical to the raw path; oversize responses raise so the
    caller can fall through to a direct fetch (which may be smaller).
    """
    quoted = urllib.parse.quote(target_url, safe=":/?#[]@!$&'()*+,;=")
    endpoint = f"{base.rstrip('/')}/{quoted}"
    req = urllib.request.Request(
        endpoint,
        headers={
            "Accept": "text/markdown",
            "User-Agent": _PUBLIC_WEB_UA,
            "X-Return-Format": "markdown",
        },
    )
    with urllib.request.urlopen(req, timeout=_WEB_FETCH_TIMEOUT) as response:  # noqa: S310 (URL built from validated base + percent-encoded target)
        body = _read_body_capped(response, max_bytes)
    return body.decode("utf-8", errors="replace")


def _raw_fetch(url: str, max_bytes: int) -> str:
    """Direct GET on the target URL with redirects disabled.

    Caller is expected to have passed ``url`` through
    ``_validate_public_https_url`` so the connection target is public.
    """
    opener = urllib.request.build_opener(_NoRedirectHandler())
    req = urllib.request.Request(url, headers={"User-Agent": _PUBLIC_WEB_UA})
    with opener.open(req, timeout=_WEB_FETCH_TIMEOUT) as response:  # noqa: S310 (URL pre-validated by _validate_public_https_url; redirects disabled by _NoRedirectHandler)
        body = _read_body_capped(response, max_bytes)
    return body.decode("utf-8", errors="replace")


def _parse_jina_response(text: str) -> tuple[str, str]:
    """Split the Jina Reader preamble ("Title:", "URL Source:", "Markdown
    Content:") from the body. Returns (title, body)."""
    if not text:
        return "", ""
    all_lines = text.split("\n")
    title = ""
    body_start = 0
    for i, line in enumerate(all_lines[:_JINA_HEADER_MAX_LINES]):
        if line.startswith("Title: "):
            title = line[len("Title: "):].strip()
            body_start = max(body_start, i + 1)
        elif line.startswith("URL Source: "):
            body_start = max(body_start, i + 1)
        elif line.startswith("Markdown Content:"):
            inline = line[len("Markdown Content:"):].strip()
            if inline:
                # inline content on same line as the marker: keep it
                body_lines = [inline] + all_lines[i + 1:]
                return title, "\n".join(body_lines).lstrip("\n")
            body_start = i + 1
            break
    if body_start == 0:
        return title, text
    body = "\n".join(all_lines[body_start:]).lstrip("\n")
    return title, body
