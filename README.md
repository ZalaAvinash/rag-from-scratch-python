```
______  ___  _____
| ___ \/ _ \|  __ \
| |_/ / /_\ \ |  \/
|    /|  _  | | __
| |\ \| | | | |_\ \
\_| \_\_| |_/\____/
```

# RAG From Scratch (Python)

A small, readable, **fully local** Retrieval-Augmented Generation pipeline in ~500 lines of Python. No cloud APIs, no API keys, no surprise bills. Reads PDFs, chunks them, embeds with a local model, stores them in a real vector DB, and answers questions grounded in the document.

Built as a companion / alternative to the .NET version in [`AI-Document-Chatbot-RAG-`](https://github.com/ZalaAvinash/AI-Document-Chatbot-RAG-) — same problem, same Ollama models, different language.

---

## ✨ What you get

```
PDF → extract text → chunk → embed → store → retrieve → prompt LLM → answer
```

| Piece | Library | Why |
|-------|---------|-----|
| PDF text extraction | `pypdf` | Pure Python, no system deps, MIT licensed |
| Chunking | (hand-rolled, ~60 lines) | No magic, no opaque tokenizers, full control over overlap |
| Embeddings | `Ollama` + `nomic-embed-text` | Free, local, 274 MB, 768-dim |
| Vector store | `ChromaDB` (persistent) | Real production DB, not in-memory dict |
| LLM | `Ollama` + `llama3.2` | Free, local, 2 GB, runs on CPU |
| CLI | `click` | Standard, well-known |

**Zero API keys. Zero cloud calls. Runs fully offline once models are pulled.**

---

## 📦 Install

```bash
# 1. Get the code
git clone https://github.com/ZalaAvinash/rag-from-scratch-python.git
cd rag-from-scratch-python

# 2. Create venv + install
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Install Ollama + pull the models
#    https://ollama.ai/download
ollama pull nomic-embed-text
ollama pull llama3.2
```

> Don't want to use Ollama? Implement the `Embedder` and `LLM` protocols in `embedder.py` and `llm.py` and pass them into `RAGPipeline(...)` — the rest of the code doesn't care.

---

## 🚀 Quick start

```bash
# Ingest a PDF
PYTHONPATH=src python -m rag.cli --persist-dir ./chroma_db ingest path/to/document.pdf

# Ask a question
PYTHONPATH=src python -m rag.cli --persist-dir ./chroma_db ask "What is the main conclusion?"
```

Sample output:

```
============================================================
ANSWER
============================================================
RAG (Retrieval-Augmented Generation) is a technique that combines a retrieval
system with a generative language model. According to Passage 1, RAG was
"popularised in 2020 by Facebook AI Research".

============================================================
SOURCES
============================================================

[1] score=0.586  chunk #0
    Retrieval-Augmented Generation (RAG) is a technique that combines a...
[2] score=0.420  chunk #1
    ...an embedder (which turns each chunk into a vector), a vector store...
[3] score=0.416  chunk #2
    ...Strong general-purpose embedding models include nomic-embed-text...
```

### Use it from Python

```python
from rag import RAGPipeline, OllamaEmbedder, OllamaLLM, VectorStore

pipeline = RAGPipeline(
    embedder=OllamaEmbedder(),
    llm=OllamaLLM(),
    store=VectorStore(persist_dir="./chroma_db"),
    chunk_size=800,
    chunk_overlap=100,
    top_k=4,
)

pipeline.ingest("path/to/document.pdf")

result = pipeline.ask("Summarize the key points")
print(result.answer)
for hit in result.sources:
    print(f"  [{hit.chunk_index}] score={hit.score:.2f}: {hit.text[:80]}...")
```

---

## 🗂 Project structure

```
rag-from-scratch-python/
├── src/rag/
│   ├── __init__.py
│   ├── __main__.py        ← entrypoint for `python -m rag`
│   ├── loaders.py         ← pypdf wrapper
│   ├── chunker.py         ← sliding-window chunker with overlap
│   ├── embedder.py        ← OllamaEmbedder (Protocol-based for swap)
│   ├── store.py           ← ChromaDB wrapper
│   ├── llm.py             ← OllamaLLM (Protocol-based for swap)
│   ├── pipeline.py        ← RAGPipeline orchestrator
│   └── cli.py             ← click-based CLI
├── tests/
│   ├── test_chunker.py    ← 9 unit tests, no external deps
│   ├── test_pipeline.py   ← 5 tests with fake embedder/LLM
│   └── fixtures/
│       └── sample.pdf     ← generated test PDF
├── .github/workflows/ci.yml  ← CI on Python 3.11, 3.12, 3.13
├── pyproject.toml         ← pytest config + src layout
├── requirements.txt
├── requirements-dev.txt   ← pytest, coverage
├── LICENSE                ← MIT
└── README.md
```

---

## 🧪 Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

14 tests:
* 9 for the chunker (no I/O, no network)
* 5 for the pipeline using deterministic fakes (no Ollama, no Chroma persistence)

CI runs on Python 3.11, 3.12, 3.13 via GitHub Actions.

---

## ⚙️ Configuration

All settings are constructor parameters — no hidden globals:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `chunk_size` | 800 chars | Window size for the chunker |
| `chunk_overlap` | 100 chars | Overlap between adjacent chunks |
| `top_k` | 4 | How many chunks to retrieve per question |
| `embedder.model` | `nomic-embed-text` | Any Ollama embedding model |
| `llm.model` | `llama3.2` | Any Ollama chat model |
| `store.persist_dir` | `./chroma_db` | Where ChromaDB writes to disk |

---

## 🧠 How it works

1. **Load** — `pypdf` extracts text from each page; empty pages are dropped.
2. **Chunk** — Text is normalized (whitespace collapsed) then split into overlapping windows of ~800 characters. The overlap keeps continuity at boundaries.
3. **Embed** — Each chunk is sent to Ollama's `nomic-embed-text` endpoint, returning a 768-dim vector.
4. **Store** — Vectors are upserted into a persistent ChromaDB collection. The document id is a short hash of the file path, so re-ingesting the same file is idempotent.
5. **Query** — The user's question is embedded, and the top-4 most similar chunks are retrieved (cosine similarity).
6. **Generate** — The retrieved chunks are formatted as numbered passages and sent to `llama3.2` with a strict system prompt that forbids the model from using outside knowledge.

The strict system prompt is the difference between RAG and "search-augmented hallucination." Without it, llama3.2 will happily invent facts that aren't in the document.

---

## 📐 Architecture

See [docs/architecture.md](docs/architecture.md) for the full data-flow diagram, design rationale, and known failure modes.

---

## 📸 Sample output

See [docs/sample-output.md](docs/sample-output.md) for real CLI captures — including a question the model correctly refuses to answer.

---

## 🤔 Why I built this

Most RAG tutorials skip the parts that actually matter: chunker parameters, prompt construction, and source attribution. This project is small enough to read in one sitting but covers the real mechanics end-to-end. Use it as:

* A reference when you need to add RAG to an existing app
* A starting point for production work (add FastAPI, async, document refresh, eval)
* A teaching tool — every module is < 200 lines

---

## 🔗 Related projects

* [AI-Document-Chatbot-RAG-](https://github.com/ZalaAvinash/AI-Document-Chatbot-RAG-) — same idea, built in .NET 8
* [dotnet-clean-architecture-starter](https://github.com/ZalaAvinash/dotnet-clean-architecture-starter) — Clean Architecture reference in .NET 8

---

## 📄 License

MIT — see [LICENSE](LICENSE).
