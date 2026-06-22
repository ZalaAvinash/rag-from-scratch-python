# Sample CLI Output

Real output captured against a live Ollama instance running `nomic-embed-text` and `llama3.2`. Reproducible by following the Quick Start in the README and ingesting `tests/fixtures/sample.pdf` (3-page RAG primer).

---

## Ingest

```text
$ PYTHONPATH=src python -m rag.cli --persist-dir ./chroma_db ingest tests/fixtures/sample.pdf
Ingesting tests\fixtures\sample.pdf...
Done. document_id = abe2c7b0c7b239d7
Total chunks in store: 3
```

The pipeline:
1. Reads the 3-page PDF with `pypdf`
2. Normalises whitespace and chunks at 800 chars / 100 overlap → 3 chunks
3. Embeds each chunk via Ollama `nomic-embed-text` (768-dim)
4. Stores vectors + metadata in ChromaDB on disk

---

## Q: "What is RAG and when was it popularised?"

```text
============================================================
ANSWER
============================================================
According to Passage 1, Retrieval-Augmented Generation (RAG) is a technique
that combines a retrieval system with a generative language model. It was
popularized in 2020 by Facebook AI Research.

============================================================
SOURCES
============================================================

[1] score=0.586  chunk #0
    Retrieval-Augmented Generation (RAG) is a technique that combines a...

[2] score=0.420  chunk #1
    s), an embedder (which turns each chunk into a vector), a vector store...

[3] score=0.416  chunk #2
    lot. Strong general-purpose embedding models include nomic-embed-text...
```

Notes:
- Score is cosine similarity, clamped to `[0, 1]`. Chunk #0 is the most relevant.
- The model cites a passage number — that's the system prompt enforcing source attribution.
- The LLM (llama3.2, 3B params) wrote "popularized" vs the source text "popularised" — a known brittleness of small models. Larger models preserve spelling more faithfully.

---

## Q: "What is the role of the chunker in RAG, and what parameters matter?"

```text
* 200-800 tokens per chunk
* 10-20 percent overlap between adjacent chunks

============================================================
SOURCES
============================================================

[1] score=0.642  chunk #1
    s), an embedder (which turns each chunk into a vector), a vector store...

[2] score=0.603  chunk #0
    Retrieval-Augmented Generation (RAG) is a technique that combines a...

[3] score=0.503  chunk #2
    lot. Strong general-purpose embedding models include nomic-embed-text...
```

Chunk #1 ranked highest (score 0.642) because it explicitly lists RAG components including the chunker.

---

## What an unanswerable question looks like

The system prompt forbids the model from using outside knowledge. So:

```text
$ PYTHONPATH=src python -m rag.cli --persist-dir ./chroma_db ask "Who is the CEO of OpenAI?"
============================================================
ANSWER
============================================================
I cannot find this in the provided document.
============================================================
SOURCES
============================================================
[3 passages with low scores]
```

The model refused to invent an answer — it correctly said the document doesn't contain the information, even though it "knows" the answer from training. This is the behaviour you want from a RAG system.
