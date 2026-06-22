"""The RAG pipeline — wire loader + chunker + embedder + store + llm together.

Public surface:
    RAGPipeline.ingest(path)        -> document_id
    RAGPipeline.ask(question)       -> Answer
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .chunker import Chunk, chunk_text
from .embedder import Embedder, OllamaEmbedder
from .loaders import load_pdf
from .llm import LLM, OllamaLLM
from .store import SearchHit, VectorStore


SYSTEM_PROMPT = """You are a careful assistant that answers questions based ONLY on the
provided document context. Follow these rules strictly:

1. Use ONLY the information in the context below. Do not use outside knowledge.
2. If the context does not contain the answer, say: "I cannot find this in the
   provided document." Do NOT guess.
3. Quote or paraphrase the relevant passages. Keep answers concise.
4. When you use information from a passage, mention which passage number it came from.
"""


@dataclass(frozen=True)
class Answer:
    """A pipeline result: the generated answer plus the sources that grounded it."""
    question: str
    answer: str
    sources: list[SearchHit]


class RAGPipeline:
    """End-to-end RAG: ingest PDFs, then ask questions about them."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        store: VectorStore | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        top_k: int = 4,
    ):
        self.embedder = embedder or OllamaEmbedder()
        self.llm = llm or OllamaLLM()
        self.store = store or VectorStore()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

    # ---- ingestion -------------------------------------------------------

    def ingest(self, path: str | Path) -> str:
        """Load a PDF, chunk it, embed it, store it. Returns a document id.

        The document id is a short sha256 of the file path so the same PDF
        produces the same id across runs (cheap idempotency).
        """
        pdf_path = Path(path)
        text = load_pdf(pdf_path)
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            raise ValueError(f"No text extracted from {pdf_path}")

        document_id = self._document_id(pdf_path)
        embeddings = self.embedder.embed_batch([c.text for c in chunks])
        self.store.add(
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
            source_path=str(pdf_path.resolve()),
        )
        return document_id

    # ---- querying --------------------------------------------------------

    def ask(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int | None = None,
    ) -> Answer:
        """Embed the question, retrieve top_k chunks, ask the LLM to answer."""
        k = top_k or self.top_k
        qvec = self.embedder.embed(question)
        hits = self.store.search(qvec, top_k=k, document_id=document_id)

        if not hits:
            return Answer(
                question=question,
                answer="I cannot find this in the provided document.",
                sources=[],
            )

        user_prompt = self._build_prompt(question, hits)
        text = self.llm.complete(SYSTEM_PROMPT, user_prompt)
        return Answer(question=question, answer=text.strip(), sources=hits)

    # ---- internals -------------------------------------------------------

    @staticmethod
    def _build_prompt(question: str, hits: list[SearchHit]) -> str:
        context_blocks: list[str] = []
        for i, h in enumerate(hits, start=1):
            context_blocks.append(f"[Passage {i}]\n{h.text}")
        context = "\n\n".join(context_blocks)
        return (
            f"Context from the document:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

    @staticmethod
    def _document_id(path: Path) -> str:
        h = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
        return h[:16]
