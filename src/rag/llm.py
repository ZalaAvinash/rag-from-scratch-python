"""LLM call abstraction.

Default: Ollama (free, local, no API key). The same `llama3.2` model used in
the .NET AI-Document-Chatbot-RAG- project.

You can swap to OpenAI or any other provider by implementing the same
`complete(system, user)` interface.
"""
from __future__ import annotations

from typing import Protocol

import httpx


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OllamaLLM:
    """Ollama-backed chat completion. Runs locally on http://localhost:11434."""

    DEFAULT_MODEL = "llama3.2"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass

    def complete(self, system: str, user: str) -> str:
        response = self._client.post(
            "/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
