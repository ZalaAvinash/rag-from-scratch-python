"""Generate embeddings for text.

Default backend: Ollama with `nomic-embed-text` (free, local, 274 MB).
The same model used in the .NET AI-Document-Chatbot-RAG- project, so the two
implementations stay interchangeable.

Swap to any other provider by implementing the same `embed` / `embed_batch`
interface — e.g. sentence-transformers, OpenAI, Cohere.
"""
from __future__ import annotations

from typing import Protocol

import httpx


class Embedder(Protocol):
    """Anything that turns text into a fixed-size vector."""
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimension(self) -> int: ...


class OllamaEmbedder:
    """Ollama-backed embedder. Runs locally on http://localhost:11434 by default."""

    DEFAULT_MODEL = "nomic-embed-text"
    DEFAULT_BASE_URL = "http://localhost:11434"
    KNOWN_DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass

    @property
    def dimension(self) -> int:
        return self.KNOWN_DIMENSIONS.get(self.model, 768)

    def embed(self, text: str) -> list[float]:
        response = self._client.post(
            "/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
