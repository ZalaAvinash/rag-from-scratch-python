# Architecture

```
                          ┌─────────────────┐
                          │   User (CLI)    │
                          └────────┬────────┘
                                   │ python -m rag.cli
                                   ▼
                          ┌─────────────────┐
                          │  RAGPipeline    │ ◄─── strict system prompt
                          │  (orchestrator) │      (forbids outside knowledge)
                          └────────┬────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
            ▼                      ▼                      ▼
     ┌────────────┐         ┌────────────┐         ┌────────────┐
     │  Loaders   │         │  Chunker   │         │  Vector    │
     │  (pypdf)   │         │  (sliding  │         │  Store     │
     │            │         │   window)  │         │ (ChromaDB) │
     └────────────┘         └────────────┘         └────────────┘
                                   │                      ▲
                                   ▼                      │
                          ┌────────────┐                  │
                          │ Embedder   │                  │
                          │ (Ollama:   │                  │
                          │  nomic-    │                  │
                          │  embed-    │                  │
                          │  text)     │                  │
                          └────────────┘                  │
                                   │                      │
                                   └──────► add() ◄───────┘
                                                          │
                          ┌────────────┐                  │
                          │   LLM      │ ◄─── retrieve() ┘
                          │ (Ollama:   │
                          │  llama3.2) │
                          └────────────┘
                                   │
                                   ▼
                            ┌─────────────┐
                            │  Answer +   │
                            │  Sources    │
                            └─────────────┘
```

## The data flow

### Ingest
1. `pypdf` extracts text from each page → full document string
2. `chunker.chunk_text()` normalises whitespace and slides an 800-char window with 100-char overlap → list of `Chunk` records
3. `embedder.OllamaEmbedder.embed_batch()` sends each chunk to `nomic-embed-text` → 768-dim vectors
4. `store.VectorStore.add()` upserts chunks + vectors + metadata to ChromaDB (persisted to disk)

### Query
1. `embedder.embed(question)` embeds the question
2. `store.search()` returns top-4 chunks by cosine similarity
3. `pipeline._build_prompt()` formats them as numbered `[Passage N]` blocks
4. `llm.OllamaLLM.complete()` sends system + user prompts to `llama3.2` → answer text
5. `Answer` returned with `text` + the `SearchHit` list (so callers can show citations)

## Why this design

| Choice | Why |
|--------|-----|
| Protocol-typed Embedder/LLM (`embedder.py`, `llm.py`) | Swap to sentence-transformers, OpenAI, Cohere, Anthropic without touching the pipeline. |
| Dataclass `Chunk` with original-text offsets | Lets you map a retrieved chunk back to its position in the source PDF for source highlighting in a UI. |
| In-process ChromaDB (not a server) | Zero infra. Single-user, single-machine RAG. For production scale, swap to `chromadb.HttpClient` or a hosted service. |
| Strict system prompt | The single biggest reason RAG systems hallucinate is a permissive system prompt. This one explicitly forbids outside knowledge. |
| Sliding window with overlap (not sentence-based chunking) | Simpler, predictable, and works well for the kinds of documents this template targets (prose, manuals, reports). For code or tables, swap in a structural chunker. |
| `top_k=4` default | Empirically a sweet spot for llama-class models — enough context to answer, not enough to overflow the prompt or dilute relevance. |

## Failure modes worth knowing

- Embedding model mismatch — embeddings and queries must use the same model. Switching `nomic-embed-text` to `mxbai-embed-large` requires re-ingesting the whole corpus.
- PDFs with scanned images — `pypdf` only extracts text. For OCR, plug a `TesseractLoader` into the same `load_pdf()` interface.
- Tables and code blocks — the chunker treats them as prose. For structured data, consider a layout-aware chunker (e.g. `unstructured.io`).
- Small models (3B) paraphrase — llama3.2 will often reword retrieved text rather than quote it verbatim. Larger models preserve wording more faithfully.
