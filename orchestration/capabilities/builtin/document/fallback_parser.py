"""Lightweight fallback document parser.

Uses pypdf for digital PDFs and built-in readers for plain text, markdown,
and CSV files. Provides fast, zero-weight parsing when Docling is not needed
or as an immediate fallback.
"""

from __future__ import annotations

import hashlib
import mimetypes
import uuid
from pathlib import Path
from typing import List, Optional

from orchestration.capabilities.builtin.document.base import (
    DocumentParseOptions,
    DocumentParser,
)
from orchestration.capabilities.builtin.document.types import (
    DocumentPage,
    DocumentTable,
    NormalizedDocument,
    Provenance,
)


class FallbackDocumentParser(DocumentParser):
    """Lightweight pure-Python fallback document parser using pypdf and standard libraries."""

    def __init__(
        self,
        default_options: Optional[DocumentParseOptions] = None,
        options: Optional[DocumentParseOptions] = None,
    ) -> None:
        self._default_options = default_options or options


    def parse(
        self,
        file_path: Path,
        options: Optional[DocumentParseOptions] = None,
    ) -> NormalizedDocument:
        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        opts = options or self._default_options or DocumentParseOptions()

        mime_type, _ = mimetypes.guess_type(str(file_path))
        mime_type = mime_type or "application/octet-stream"

        file_bytes = file_path.read_bytes()
        sha256 = hashlib.sha256(file_bytes).hexdigest()
        file_size = len(file_bytes)
        doc_id = f"doc-{uuid.uuid4().hex[:12]}"

        ext = file_path.suffix.lower()

        if ext == ".pdf":
            return self._parse_pdf(file_path, file_size, sha256, doc_id, opts)
        elif ext in (".txt", ".md", ".markdown", ".csv", ".json", ".log"):
            return self._parse_text(file_path, file_size, sha256, doc_id, mime_type)
        else:
            # Best-effort text decode
            try:
                text = file_bytes.decode("utf-8")
                return NormalizedDocument(
                    document_id=doc_id,
                    filename=file_path.name,
                    mime_type=mime_type,
                    text=text,
                    page_count=1,
                    pages=[DocumentPage(page_number=1, text=text)],
                    tables=[],
                    figures=[],
                    metadata={
                        "parser_backend": "fallback_text",
                        "has_selectable_text": True,
                        "ocr_applied": False,
                        "file_size_bytes": file_size,
                        "sha256": sha256,
                    },
                )
            except UnicodeDecodeError:
                raise ValueError(
                    f"Unsupported binary file format '{ext}' without Docling OCR pipeline: {file_path.name}"
                )

    def _parse_pdf(
        self,
        file_path: Path,
        file_size: int,
        sha256: str,
        doc_id: str,
        opts: DocumentParseOptions,
    ) -> NormalizedDocument:
        import pypdf

        reader = pypdf.PdfReader(str(file_path))
        total_pages = len(reader.pages)
        limit = min(total_pages, opts.max_pages) if opts.max_pages else total_pages

        pages: List[DocumentPage] = []
        full_text_parts: List[str] = []
        total_extracted_chars = 0

        for idx in range(limit):
            page_num = idx + 1
            pdf_page = reader.pages[idx]
            page_text = pdf_page.extract_text() or ""
            total_extracted_chars += len(page_text.strip())

            dim = None
            if pdf_page.mediabox:
                dim = (float(pdf_page.mediabox.width), float(pdf_page.mediabox.height))

            pages.append(
                DocumentPage(
                    page_number=page_num,
                    text=page_text,
                    dimension=dim,
                    tables=[],
                    figures=[],
                )
            )
            full_text_parts.append(page_text)

        # A document has selectable text if there is a reasonable character density
        has_selectable_text = total_extracted_chars >= 50 or (total_pages > 0 and total_extracted_chars > 0)
        ocr_needed = not has_selectable_text and total_pages > 0

        unified_text = "\n\n".join(part for part in full_text_parts if part.strip())

        return NormalizedDocument(
            document_id=doc_id,
            filename=file_path.name,
            mime_type="application/pdf",
            text=unified_text,
            page_count=total_pages,
            pages=pages,
            tables=[],
            figures=[],
            metadata={
                "parser_backend": "fallback_pypdf",
                "has_selectable_text": has_selectable_text,
                "ocr_applied": False,
                "ocr_needed": ocr_needed,
                "file_size_bytes": file_size,
                "sha256": sha256,
            },
        )

    def _parse_text(
        self,
        file_path: Path,
        file_size: int,
        sha256: str,
        doc_id: str,
        mime_type: str,
    ) -> NormalizedDocument:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        tables = []

        # If CSV, parse into DocumentTable
        if file_path.suffix.lower() == ".csv":
            import csv
            import io

            rows = list(csv.reader(io.StringIO(text)))
            if rows:
                num_cols = max(len(r) for r in rows)
                # Build markdown table
                md_lines = []
                headers = rows[0]
                md_lines.append("| " + " | ".join(headers) + " |")
                md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for r in rows[1:]:
                    padded = r + [""] * (len(headers) - len(r))
                    md_lines.append("| " + " | ".join(padded) + " |")
                md_table = "\n".join(md_lines)

                tables.append(
                    DocumentTable(
                        table_id=f"tbl-1",
                        page_number=1,
                        num_rows=len(rows),
                        num_cols=num_cols,
                        grid=rows,
                        markdown=md_table,
                        provenance=Provenance(page_number=1),
                    )
                )

        return NormalizedDocument(
            document_id=doc_id,
            filename=file_path.name,
            mime_type=mime_type,
            text=text,
            page_count=1,
            pages=[DocumentPage(page_number=1, text=text, tables=tables)],
            tables=tables,
            figures=[],
            metadata={
                "parser_backend": "fallback_text",
                "has_selectable_text": True,
                "ocr_applied": False,
                "file_size_bytes": file_size,
                "sha256": sha256,
            },
        )
