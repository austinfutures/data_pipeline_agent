"""
Thin wrapper around the Gemini client so agent code depends on a small
interface (`generate(system, user) -> str`) rather than the SDK directly.
Makes it trivial to swap models/providers or inject a fake for tests.
"""
from __future__ import annotations

import os
from typing import Optional, Protocol


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        from google import genai  # imported lazily so tests can run without the package

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY environment variable or api_key parameter is required.")
        self.client = genai.Client(api_key=key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self.client.models.generate_content(model=self.model, contents=full_prompt)
        return response.text


class FakeLLMClient:
    """Deterministic stand-in for tests/eval — returns scripted responses
    in order, or a default if the script runs out."""

    def __init__(self, scripted_responses: list[str], default: str = ""):
        self._responses = list(scripted_responses)
        self._default = default
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self._responses:
            return self._responses.pop(0)
        return self._default
