"""Thin wrapper around the official google-genai SDK. Returns a plain callable so
the rest of the code (and the tests) can inject a fake."""
from __future__ import annotations

from typing import Callable, Optional

from config import GEMINI_MODEL, gemini_api_key

GenerateFn = Callable[[str, str, bool], str]   # (prompt, system_instruction, json_mode) -> text


class GeminiUnavailable(Exception):
    pass


def make_generate_fn(api_key: Optional[str] = None, model: Optional[str] = None) -> GenerateFn:
    key = (api_key if api_key is not None else gemini_api_key()).strip()
    if not key:
        raise GeminiUnavailable("GEMINI_API_KEY is not set (see .env.example).")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)
    model_name = model or GEMINI_MODEL

    def generate(prompt: str, system_instruction: str = "", json_mode: bool = False) -> str:
        cfg = {"system_instruction": system_instruction or None, "temperature": 0.2}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = {
                "type": "OBJECT",
                "properties": {"ticker": {"type": "STRING"}, "sentiment": {"type": "NUMBER"},
                               "reason": {"type": "STRING"}},
                "required": ["ticker", "sentiment", "reason"]}
        resp = client.models.generate_content(model=model_name, contents=prompt,
                                              config=types.GenerateContentConfig(**cfg))
        return resp.text or ""

    return generate
