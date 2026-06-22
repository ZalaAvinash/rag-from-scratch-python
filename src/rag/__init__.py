"""RAG from scratch — a minimal, free, local Retrieval-Augmented Generation pipeline.

Modules:
    loaders  — extract text from PDFs
    chunker  — split long text into overlapping windows
    embedder — turn text into vectors (Ollama / sentence-transformers)
    store    — persist + search vectors (ChromaDB)
    llm      — generate answers (Ollama)
    pipeline — orchestrate the above
    cli      — click-based command-line entrypoint
"""

__version__ = "0.1.0"
