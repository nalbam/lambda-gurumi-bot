"""XAIProvider — Grok chat + grok-imagine at https://api.x.ai/v1."""
from __future__ import annotations

from typing import Any

from src.llms.openai_wire import _OpenAICompatProvider


class XAIProvider(_OpenAICompatProvider):
    """xAI (Grok) — OpenAI-wire compatible, different base URL and image params.

    Models:
      text:  grok-4-1-fast-reasoning, grok-4.20-0309-reasoning, ...
      image: grok-imagine-image, grok-imagine-image-pro

    Differences from OpenAI that matter here:
      - `images.generate` rejects `size` (uses `aspect_ratio`/`resolution`).
        We omit `size` and request `response_format=b64_json` so we can
        decode bytes locally, matching the rest of the pipeline.
      - All current grok chat models accept `max_tokens` + `temperature`
        the classic way — no `max_completion_tokens` split.
    """

    BASE_URL = "https://api.x.ai/v1"
    API_KEY_ENV_VAR = "XAI_API_KEY"

    def _image_generate_kwargs(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.image_model,
            "prompt": prompt,
            "n": 1,
            "response_format": "b64_json",
        }
