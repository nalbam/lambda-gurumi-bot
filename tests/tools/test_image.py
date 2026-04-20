"""Tests for src.tools.image."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tests.tools._helpers import _ctx, _settings, _streamed_read
from src.tools.image import generate_image


# --------------------------------------------------------------------------- #
# generate_image
# --------------------------------------------------------------------------- #


def test_generate_image_returns_permalink():
    llm = MagicMock()
    llm.generate_image.return_value = b"imgbytes"
    client = MagicMock()
    client.files_upload_v2.return_value = {"file": {"permalink": "https://slack/abc", "title": "t"}}
    ctx = _ctx(slack_client=client, llm=llm)
    out = generate_image(ctx, prompt="cat")
    assert out == {"permalink": "https://slack/abc", "title": "t"}
    llm.generate_image.assert_called_once_with("cat")
