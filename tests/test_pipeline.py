"""Tests for the RAG pipeline using a fake embedder + fake LLM + ephemeral store.

The fakes let us test the orchestration logic — chunking, retrieval ordering,
prompt construction, source attribution — without spinning up Ollama or
persisting Chroma data.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from rag.chunker import Chunk
from rag.embedder import Embedder
from rag.llm import LLM
from rag.pipeline import Answer, RAGPipeline
from rag.store import VectorStore


class FakeEmbedder:
    """A deterministic embedder: same text → same vector; related text → closer vectors.

    Uses simple bag-of-words hashing so we don't depend on numpy.
    """
    DIM = 8

    def __init__(self):
        self._calls: list[str] = []

    @property
    def dimension(self) -> int:
        return self.DIM

    def embed(self, text: str) -> list[float]:
        self._calls.append(text)
        v = [0.0] * self.DIM
        for word in text.lower().split():
            v[hash(word) % self.DIM] += 1.0
        return v

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class FakeLLM:
    """Echoes the user prompt back, prefixed with a marker, so we can assert
    on what the pipeline sent to the LLM.
    """
    def __init__(self):
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return f"[FAKE-LLM-REPLY] user_prompt_chars={len(user)}"


@pytest.fixture
def tmp_store_dir():
    d = Path(tempfile.mkdtemp(prefix="rag-test-"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def pipeline(tmp_store_dir):
    return RAGPipeline(
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
        store=VectorStore(persist_dir=tmp_store_dir, collection_name=f"test-{id(tmp_store_dir)}"),
        chunk_size=200,
        chunk_overlap=20,
        top_k=3,
    )


def test_ingest_adds_chunks_to_store(pipeline):
    pdf = Path(__file__).parent / "fixtures" / "sample.pdf"
    if not pdf.exists():
        pytest.skip("sample.pdf fixture not present")

    doc_id = pipeline.ingest(pdf)

    assert doc_id
    assert pipeline.store.count() > 0


def test_ask_returns_answer_with_sources(pipeline):
    pdf = Path(__file__).parent / "fixtures" / "sample.pdf"
    if not pdf.exists():
        pytest.skip("sample.pdf fixture not present")

    pipeline.ingest(pdf)
    result = pipeline.ask("What is this document about?")

    assert isinstance(result, Answer)
    assert result.answer  # non-empty
    assert len(result.sources) > 0
    assert len(result.sources) <= pipeline.top_k


def test_ask_with_empty_store_returns_graceful_answer(tmp_store_dir):
    pipeline = RAGPipeline(
        embedder=FakeEmbedder(),
        llm=FakeLLM(),
        store=VectorStore(persist_dir=tmp_store_dir, collection_name="empty-test"),
    )
    result = pipeline.ask("anything")

    assert "cannot find" in result.answer.lower()
    assert result.sources == []


def test_ask_passes_top_k_chunks_to_llm(pipeline):
    pdf = Path(__file__).parent / "fixtures" / "sample.pdf"
    if not pdf.exists():
        pytest.skip("sample.pdf fixture not present")

    pipeline.ingest(pdf)
    pipeline.ask("A question")

    # The LLM's user prompt should contain a [Passage N] marker for each retrieved chunk
    user = pipeline.llm.last_user
    assert user is not None
    assert "[Passage 1]" in user
    assert "Question:" in user


def test_document_id_is_stable(pipeline):
    pdf = Path(__file__).parent / "fixtures" / "sample.pdf"
    if not pdf.exists():
        pytest.skip("sample.pdf fixture not present")

    id1 = pipeline.ingest(pdf)
    id2 = pipeline.ingest(pdf)

    # Same path should produce the same document id (idempotency)
    assert id1 == id2
