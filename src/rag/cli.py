"""Command-line interface: `python -m rag.cli ingest <pdf>` and `ask <question>`."""
from __future__ import annotations

import sys
from pathlib import Path

import click

from .pipeline import RAGPipeline


@click.group()
@click.option(
    "--persist-dir",
    default="./chroma_db",
    show_default=True,
    help="Directory where the vector store is persisted.",
)
@click.pass_context
def main(ctx: click.Context, persist_dir: str) -> None:
    """RAG from scratch — local, free, no API keys."""
    ctx.ensure_object(dict)
    ctx.obj["persist_dir"] = persist_dir


@main.command()
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--chunk-size", default=800, show_default=True)
@click.option("--chunk-overlap", default=100, show_default=True)
@click.pass_context
def ingest(ctx: click.Context, pdf_path: Path, chunk_size: int, chunk_overlap: int) -> None:
    """Ingest a PDF into the vector store."""
    pipeline = RAGPipeline(
        store=None,  # we re-create below with the configured persist dir
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    # Re-bind the store to use the CLI's persist dir
    from .store import VectorStore
    pipeline.store = VectorStore(persist_dir=ctx.obj["persist_dir"])

    click.echo(f"Ingesting {pdf_path}...")
    doc_id = pipeline.ingest(pdf_path)
    click.echo(f"Done. document_id = {doc_id}")
    click.echo(f"Total chunks in store: {pipeline.store.count()}")


@main.command()
@click.argument("question")
@click.option("--top-k", default=4, show_default=True)
@click.option("--document-id", default=None, help="Restrict to one document.")
@click.pass_context
def ask(ctx: click.Context, question: str, top_k: int, document_id: str | None) -> None:
    """Ask a question and print the answer with sources."""
    from .store import VectorStore
    pipeline = RAGPipeline(
        store=VectorStore(persist_dir=ctx.obj["persist_dir"]),
        top_k=top_k,
    )

    result = pipeline.ask(question, document_id=document_id)

    click.echo("\n" + "=" * 60)
    click.echo("ANSWER")
    click.echo("=" * 60)
    click.echo(result.answer)

    if result.sources:
        click.echo("\n" + "=" * 60)
        click.echo("SOURCES")
        click.echo("=" * 60)
        for i, hit in enumerate(result.sources, start=1):
            preview = hit.text[:160].replace("\n", " ")
            click.echo(f"\n[{i}] score={hit.score:.3f}  chunk #{hit.chunk_index}")
            click.echo(f"    {preview}{'...' if len(hit.text) > 160 else ''}")


if __name__ == "__main__":
    main(obj={})
