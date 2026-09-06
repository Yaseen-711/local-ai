"""Terminal console interface for the RAG Developer Test Harness.

Provides an interactive CLI menu, formatted visual outputs, timing breakdowns,
provenance displays, and robust error handling for manual end-to-end RAG verification.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from rag.cli.harness import (
    ChunkDetails,
    DocumentSummary,
    IngestionStats,
    QueryResult,
    RAGTestHarness,
)
from rag.reranking.models import RankedChunk
from rag.retrieval.models import RetrievedChunk


def _print_banner() -> None:
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "LOCAL AI RAG DEVELOPER CONSOLE".center(58) + "║")
    print("║" + "Docling → Chunking → Nomic → pgvector → Reranking".center(58) + "║")
    print("╚" + "═" * 58 + "╝")


def _print_menu() -> None:
    print("\nAvailable Actions:")
    print("  1. Ingest Document (PDF, Markdown, DOCX, TXT)")
    print("  2. Ask Question (Vector Retrieval + Reranking)")
    print("  3. Inspect Documents & Chunks")
    print("  4. Clear Test Data")
    print("  5. Exit")


def _truncate_text(text: str, max_chars: int = 180) -> str:
    """Format text preview cleanly for terminal display."""
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "..."


def display_ingestion_progress(step_name: str, step_idx: int, total_steps: int = 6) -> None:
    """Progress callback displayed during document ingestion."""
    dots = "." * max(2, 34 - len(step_name))
    print(f"  [{step_idx}/{total_steps}] {step_name} {dots} ✓")


def display_ingestion_stats(stats: IngestionStats) -> None:
    """Display clean statistics and timings after ingestion."""
    print("\n" + "─" * 60)
    print(f"DOCUMENT INGESTION COMPLETE ({stats.action.upper()})")
    print("─" * 60)
    print(f"  Document File:    {stats.file_name}")
    print(f"  Document ID:      {stats.document_id}")
    print(f"  Format:           {stats.format.upper() or 'N/A'}")
    print(f"  Pages:            {stats.page_count}")
    print(f"  Elements:         {stats.element_count}")
    print(f"  Chunks Created:   {stats.chunk_count}")
    print(f"  Embeddings:       {stats.embedding_count} (768-dim, unit-normalized)")
    print(f"  Indexed in DB:    {stats.indexed_count}")
    print("\n  Stage Timings:")
    t = stats.timings
    print(f"    - Docling Parsing:    {t.ingestion_sec:.3f}s")
    print(f"    - Normalization:      {t.normalization_sec:.3f}s")
    print(f"    - Chunking:           {t.chunking_sec:.3f}s")
    print(f"    - Metadata Pipeline:  {t.metadata_sec:.3f}s")
    print(f"    - Nomic Embeddings:   {t.embedding_sec:.3f}s")
    print(f"    - pgvector Indexing:  {t.indexing_sec:.3f}s")
    print(f"    - Total Time:         {t.total_sec:.3f}s")
    print("─" * 60)


def display_query_results(result: QueryResult, verbose_content: bool = False) -> None:
    """Display vector retrieval, cross-encoder reranking, and provenance side-by-side."""
    print("\n" + "═" * 60)
    print(f"QUERY: \"{result.query}\"")
    print(f"Vector Dimension: {result.query_vector_dim} | top_k: {result.top_k} | top_n: {result.top_n}")
    if result.document_id_scope:
        print(f"Scope: restricted to document_id='{result.document_id_scope}'")
    if result.similarity_threshold is not None:
        print(f"Threshold: minimum cosine similarity >= {result.similarity_threshold}")
    print("═" * 60)

    # 1. VECTOR RETRIEVAL SECTION
    print("\n" + "─" * 60)
    print(f"STAGE 1: VECTOR RETRIEVAL ({len(result.retrieved_candidates)} candidates)")
    print("─" * 60)

    if not result.retrieved_candidates:
        print("  No candidate chunks met the search criteria.")
    else:
        for c in result.retrieved_candidates:
            meta = c.metadata or {}
            source_name = meta.get("file_name") or meta.get("source_path") or c.document_id
            page = meta.get("primary_page") or meta.get("page_numbers") or "N/A"
            heading = meta.get("heading") or meta.get("heading_path") or "N/A"

            print(f"\n  [Rank {c.rank}] Similarity Score: {c.similarity_score:.4f}")
            print(f"    Chunk ID:     {c.chunk_id}")
            print(f"    Source Doc:   {source_name}")
            print(f"    Page:         {page}")
            print(f"    Heading:      {heading}")
            content_str = c.content if verbose_content else _truncate_text(c.content)
            print(f"    Content:      {content_str}")

    # 2. CROSS-ENCODER RERANKING SECTION
    print("\n" + "─" * 60)
    print(f"STAGE 2: CROSS-ENCODER RERANKING ({len(result.ranked_candidates)} final)")
    print("─" * 60)

    if not result.ranked_candidates:
        print("  No candidate chunks were reranked.")
    else:
        for r in result.ranked_candidates:
            meta = r.metadata or {}
            source_name = meta.get("file_name") or meta.get("source_path") or r.document_id
            page = meta.get("primary_page") or meta.get("page_numbers") or "N/A"
            heading = meta.get("heading") or meta.get("heading_path") or "N/A"
            rank_shift = r.original_retrieval_rank - r.rerank_rank
            shift_str = f"+{rank_shift}" if rank_shift > 0 else f"{rank_shift}" if rank_shift < 0 else "="

            print(f"\n  [Rank {r.rerank_rank}] Cross-Encoder Score: {r.reranking_score:.4f} (Shift: {shift_str})")
            print(f"    Original Rank: {r.original_retrieval_rank} (Vector Cosine: {r.original_similarity_score:.4f})")
            print(f"    Chunk ID:      {r.chunk_id}")
            print(f"    Source Doc:    {source_name}")
            print(f"    Page:          {page}")
            print(f"    Heading:       {heading}")
            content_str = r.content if verbose_content else _truncate_text(r.content)
            print(f"    Content:       {content_str}")

    # 3. PIPELINE SUMMARY
    t = result.timings
    print("\n" + "─" * 60)
    print("DEBUG PIPELINE SUMMARY")
    print("─" * 60)
    print(f"  [1] Query Embedding:     ✓ ({t.query_embedding_sec:.3f}s)")
    print(f"  [2] Vector Retrieval:    ✓ ({len(result.retrieved_candidates)} candidates in {t.retrieval_sec:.3f}s)")
    print(f"  [3] Cross-Encoder:       ✓ ({len(result.ranked_candidates)} selected in {t.reranking_sec:.3f}s)")
    print(f"  [4] Total Latency:       {t.total_sec:.3f}s")
    print("  [5] LLM Generation:      NOT IMPLEMENTED (Pipeline stops after reranking)")
    print("─" * 60)


def display_documents(docs: List[DocumentSummary]) -> None:
    """Display table of all currently indexed documents."""
    print("\n" + "─" * 60)
    print(f"INDEXED DOCUMENTS ({len(docs)} found)")
    print("─" * 60)
    if not docs:
        print("  No documents currently indexed in PostgreSQL.")
        return

    # Dynamically scale column width so the full document ID is always displayed and copyable
    id_width = max(len("Document ID"), max((len(d.document_id) for d in docs), default=26))

    header = f"  {'#':<3} {'Document ID':<{id_width}} {'Format':<8} {'Pages':<6} {'Chunks':<8} {'Created At'}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, d in enumerate(docs, start=1):
        print(f"  {i:<3} {d.document_id:<{id_width}} {d.format:<8} {d.page_count:<6} {d.chunk_count:<8} {d.created_at}")


def display_chunk_details(details: ChunkDetails) -> None:
    """Display in-depth information for a single chunk."""
    print("\n" + "─" * 60)
    print(f"CHUNK DETAILS: {details.chunk_id}")
    print("─" * 60)
    print(f"  Document ID:      {details.document_id}")
    print(f"  Chunk Index:      {details.chunk_index}")
    print(f"  Vector Dimension: {details.vector_dimension}")
    print(f"  Vector L2 Norm:   {details.vector_norm}")
    print(f"  Vector Preview:   {details.vector_preview} ...")
    print(f"  Created At:       {details.created_at}")
    print("\n  Metadata:")
    for k, v in details.metadata.items():
        print(f"    - {k}: {v}")
    print("\n  Full Content:")
    print("  " + "-" * 40)
    for line in details.content.split("\n"):
        print(f"    {line}")
    print("  " + "-" * 40)


class RAGConsoleApp:
    """Interactive console application for the RAG Developer Test Harness."""

    def __init__(self, harness: Optional[RAGTestHarness] = None, debug: bool = False) -> None:
        self.harness = harness or RAGTestHarness()
        self.debug = debug

    def handle_ingest(self) -> None:
        """Handle Option 1: Ingest Document."""
        print("\n--- Ingest Document ---")
        path_input = input("Enter path to file (PDF, DOCX, Markdown, TXT): ").strip()
        if not path_input:
            print("No file path entered. Returning to menu.")
            return

        try:
            print(f"\nIngesting: {path_input}")
            stats = self.harness.ingest_document(
                file_path=path_input,
                progress_callback=display_ingestion_progress,
            )
            display_ingestion_stats(stats)
        except Exception as exc:
            print(f"\nERROR: Ingestion failed: {exc}")
            if self.debug:
                import traceback

                traceback.print_exc()

    def handle_query(self) -> None:
        """Handle Option 2: Ask Question."""
        print("\n--- Ask Question ---")
        question = input("Enter question: ").strip()
        if not question:
            print("Empty question. Returning to menu.")
            return

        # Optional parameter prompts with sensible defaults
        top_k_str = input("top_k candidate retrieval count [10]: ").strip()
        top_k = int(top_k_str) if top_k_str.isdigit() and int(top_k_str) > 0 else 10

        top_n_str = input("top_n final rerank count [5]: ").strip()
        top_n = int(top_n_str) if top_n_str.isdigit() and int(top_n_str) > 0 else 5

        doc_scope = input("Restrict to document_id (or press Enter for all): ").strip() or None

        thresh_str = input("Minimum similarity threshold in [-1.0, 1.0] (or press Enter for none): ").strip()
        threshold: Optional[float] = None
        if thresh_str:
            try:
                threshold = float(thresh_str)
            except ValueError:
                print(f"Invalid threshold '{thresh_str}', proceeding with none.")

        try:
            print("\nExecuting vector similarity search and reranking...")
            result = self.harness.query(
                question=question,
                top_k=top_k,
                top_n=top_n,
                document_id=doc_scope,
                similarity_threshold=threshold,
            )
            display_query_results(result)
        except Exception as exc:
            print(f"\nERROR: Query failed: {exc}")
            if self.debug:
                import traceback

                traceback.print_exc()

    def handle_inspect(self) -> None:
        """Handle Option 3: Inspect Documents & Chunks."""
        print("\n--- Inspect Documents ---")
        try:
            docs = self.harness.list_documents()
            display_documents(docs)
            if not docs:
                return

            sub_choice = input("\n[1] View chunks for a document, [2] Inspect specific chunk ID, [Enter] Return: ").strip()
            if sub_choice == "1":
                doc_input = input("Enter document ID (or row number): ").strip()
                matched_doc = next((d for d in docs if d.document_id == doc_input), None)
                if matched_doc:
                    doc_id = matched_doc.document_id
                elif doc_input.isdigit() and 1 <= int(doc_input) <= len(docs):
                    doc_id = docs[int(doc_input) - 1].document_id
                else:
                    doc_id = doc_input

                chunks = self.harness.list_chunks_for_document(doc_id)
                print(f"\nChunks in document '{doc_id}' ({len(chunks)} total):")
                for c in chunks:
                    print(f"  #{c['chunk_index']} ID: {c['chunk_id']} (p.{c['page']}, {c['heading']}): \"{c['preview']}\"")
            elif sub_choice == "2":
                chunk_id = input("Enter chunk ID: ").strip()
                details = self.harness.get_chunk_details(chunk_id)
                if details:
                    display_chunk_details(details)
                else:
                    print(f"Chunk '{chunk_id}' not found.")
        except Exception as exc:
            print(f"\nERROR: Inspection failed: {exc}")
            if self.debug:
                import traceback

                traceback.print_exc()

    def handle_clear(self) -> None:
        """Handle Option 4: Clear Test Data."""
        print("\n--- Clear Test Data ---")
        docs = self.harness.list_documents()
        display_documents(docs)
        if not docs:
            return

        confirm = input("\nWARNING: This will delete test documents and chunks from PostgreSQL.\nProceed? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Operation aborted.")
            return

        target_input = input("Enter specific document_id to delete (or press Enter to clear ALL): ").strip()
        target_doc = None
        if target_input:
            matched_doc = next((d for d in docs if d.document_id == target_input), None)
            if matched_doc:
                target_doc = matched_doc.document_id
            elif target_input.isdigit() and 1 <= int(target_input) <= len(docs):
                target_doc = docs[int(target_input) - 1].document_id
            else:
                target_doc = target_input
        try:
            count = self.harness.clear_data(target_doc)
            print(f"\nSuccessfully deleted {count} document(s) and cascading chunks.")
        except Exception as exc:
            print(f"\nERROR: Failed to clear data: {exc}")
            if self.debug:
                import traceback

                traceback.print_exc()

    def run(self) -> None:
        """Main interactive application loop."""
        _print_banner()

        while True:
            _print_menu()
            choice = input("\nSelect choice [1-5]: ").strip()

            if choice == "1":
                self.handle_ingest()
            elif choice == "2":
                self.handle_query()
            elif choice == "3":
                self.handle_inspect()
            elif choice == "4":
                self.handle_clear()
            elif choice in ("5", "q", "exit", "quit"):
                print("\nExiting Local AI RAG Console. Goodbye!")
                break
            else:
                print("Invalid option. Please enter a number between 1 and 5.")


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct argument parser for CLI options."""
    parser = argparse.ArgumentParser(description="Local AI Foundation RAG Developer Test Harness")
    parser.add_argument("--debug", action="store_true", help="Print full Python tracebacks on error")
    parser.add_argument("--ingest", type=str, help="Ingest a document file directly without interactive prompt")
    parser.add_argument("--query", type=str, help="Ask a question directly without interactive prompt")
    parser.add_argument("--top-k", type=int, default=10, help="Candidate retrieval count (default: 10)")
    parser.add_argument("--top-n", type=int, default=5, help="Reranked final count (default: 5)")
    return parser


def main() -> None:
    """CLI entry point supporting arguments and interactive loop."""
    from rag.offline import ensure_offline_environment

    ensure_offline_environment()

    parser = build_arg_parser()
    args = parser.parse_args()

    app = RAGConsoleApp(debug=args.debug)

    # Non-interactive CLI mode
    if args.ingest:
        print(f"Direct Ingest Mode: {args.ingest}")
        try:
            stats = app.harness.ingest_document(args.ingest, progress_callback=display_ingestion_progress)
            display_ingestion_stats(stats)
        except Exception as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        return

    if args.query:
        print(f"Direct Query Mode: \"{args.query}\"")
        try:
            res = app.harness.query(args.query, top_k=args.top_k, top_n=args.top_n)
            display_query_results(res)
        except Exception as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        return

    # Interactive mode
    app.run()


if __name__ == "__main__":
    main()
