"""Vector store backed by ChromaDB.

ChromaDB is a real, production-shaped vector DB that runs in-process or as a
standalone server. It persists to disk, supports metadata filtering, and uses
cosine similarity by default — exactly what we want for semantic search.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.config import Settings

from .chunker import Chunk
from .embedder import Embedder


@dataclass(frozen=True)
class SearchHit:
    """One result from a vector search."""
    text: str
    score: float          # higher = more similar (we convert distance → similarity)
    metadata: dict
    chunk_index: int


class VectorStore:
    """Thin wrapper around a ChromaDB collection."""

    def __init__(
        self,
        persist_dir: str | Path = "./chroma_db",
        collection_name: str = "rag",
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=False),
        )
        # cosine space — works regardless of embedding norm and is standard for semantic search
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def add(
        self,
        chunks: Iterable[Chunk],
        embeddings: list[list[float]],
        document_id: str,
        source_path: str = "",
    ) -> None:
        """Add chunks + their pre-computed embeddings to the store."""
        chunks = list(chunks)
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Got {len(chunks)} chunks but {len(embeddings)} embeddings"
            )
        if not chunks:
            return

        ids = [f"{document_id}::{c.index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "source_path": source_path,
                "chunk_index": c.index,
                "char_start": c.char_start,
                "char_end": c.char_end,
            }
            for c in chunks
        ]

        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        """Find the top_k most similar chunks to the query embedding.

        Args:
            query_embedding: Vector for the question.
            top_k: How many results to return.
            document_id: If given, restrict the search to one document.

        Returns:
            List of SearchHit ordered by descending similarity.
        """
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if document_id is not None:
            kwargs["where"] = {"document_id": document_id}

        result = self._collection.query(**kwargs)

        hits: list[SearchHit] = []
        # Chroma returns parallel arrays — one entry per query (we sent one query)
        for doc, dist, meta in zip(
            result["documents"][0],
            result["distances"][0],
            result["metadatas"][0],
        ):
            # cosine distance in [0, 2]; convert to similarity in [-1, 1] (clamped to [0, 1] for display)
            similarity = max(0.0, 1.0 - float(dist))
            hits.append(
                SearchHit(
                    text=doc,
                    score=similarity,
                    metadata=meta,
                    chunk_index=int(meta.get("chunk_index", 0)),
                )
            )
        return hits
