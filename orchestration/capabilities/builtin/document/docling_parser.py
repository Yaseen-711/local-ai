"""Docling 2.x DocumentParser adapter.

Wraps IBM Docling DocumentConverter behind the DocumentParser seam.
Handles digital PDFs, scanned PDFs (with RapidOCR), layout analysis,
and TableFormer table extraction.

Strict Invariant:
  All Docling objects (DoclingDocument, DocItem, TableItem, etc.) are consumed
  strictly inside this module and mapped into NormalizedDocument.
  No Docling internal types are ever returned or leaked across this boundary.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import List, Optional

from orchestration.capabilities.builtin.document.base import (
    DocumentParseOptions,
    DocumentParser,
)
from orchestration.capabilities.builtin.document.types import (
    BoundingBox,
    DocumentFigure,
    DocumentPage,
    DocumentTable,
    NormalizedDocument,
    Provenance,
)

logger = logging.getLogger(__name__)


class DoclingDocumentParser(DocumentParser):
    """DocumentParser implementation backed by IBM Docling 2.x."""

    def __init__(
        self,
        default_options: Optional[DocumentParseOptions] = None,
        options: Optional[DocumentParseOptions] = None,
    ) -> None:
        self._default_options = default_options or options
        self._converter = None


    def _get_converter(self, options: DocumentParseOptions):
        """Lazy initialization of Docling DocumentConverter with requested options."""
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = options.do_ocr
        pipeline_options.do_table_structure = options.extract_tables

        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }

        return DocumentConverter(format_options=format_options)

    def parse(
        self,
        file_path: Path,
        options: Optional[DocumentParseOptions] = None,
    ) -> NormalizedDocument:
        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        opts = options or self._default_options or DocumentParseOptions()

        file_bytes = file_path.read_bytes()
        sha256 = hashlib.sha256(file_bytes).hexdigest()
        file_size = len(file_bytes)
        doc_id = f"doc-{uuid.uuid4().hex[:12]}"

        mime_type, _ = mimetypes.guess_type(str(file_path))
        mime_type = mime_type or "application/pdf"

        try:
            converter = self._get_converter(opts)
            conversion_result = converter.convert(str(file_path))
            doc = conversion_result.document
        except Exception as exc:
            logger.warning(
                "Docling conversion failed for %s (%s: %s).",
                file_path.name,
                type(exc).__name__,
                exc,
            )
            raise RuntimeError(f"Docling parsing failed for '{file_path.name}': {exc}") from exc

        # 1. Full unified markdown text
        unified_markdown = doc.export_to_markdown()

        # 2. Structured tables
        extracted_tables: List[DocumentTable] = []
        for idx, tbl in enumerate(getattr(doc, "tables", [])):
            tbl_id = f"tbl-{idx + 1}"
            page_no = 1
            prov = None

            # Extract provenance from Docling prov list
            if getattr(tbl, "prov", None) and len(tbl.prov) > 0:
                p_item = tbl.prov[0]
                page_no = getattr(p_item, "page_no", 1)
                bbox_obj = getattr(p_item, "bbox", None)
                if bbox_obj:
                    prov = Provenance(
                        page_number=page_no,
                        bbox=BoundingBox(
                            l=float(bbox_obj.l),
                            t=float(bbox_obj.t),
                            r=float(bbox_obj.r),
                            b=float(bbox_obj.b),
                            coord_origin=str(getattr(bbox_obj, "coord_origin", "BOTTOMLEFT")),
                        ),
                    )
                else:
                    prov = Provenance(page_number=page_no)

            grid: List[List[str]] = []
            try:
                df = tbl.export_to_dataframe()
                headers = [str(c) for c in df.columns]
                rows = [[str(val) for val in row] for row in df.values]
                grid = [headers] + rows
            except Exception:
                # Fallback to cell iteration if dataframe export fails
                data_grid = getattr(tbl, "data", None)
                if data_grid and hasattr(data_grid, "grid"):
                    grid = [[str(cell.text) for cell in row] for row in data_grid.grid]

            md_text = ""
            try:
                md_text = tbl.export_to_markdown()
            except Exception:
                pass

            extracted_tables.append(
                DocumentTable(
                    table_id=tbl_id,
                    page_number=page_no,
                    num_rows=len(grid),
                    num_cols=len(grid[0]) if grid else 0,
                    grid=grid,
                    markdown=md_text,
                    provenance=prov,
                )
            )

        # 3. Figures
        extracted_figures: List[DocumentFigure] = []
        for idx, fig in enumerate(getattr(doc, "pictures", [])):
            fig_id = f"fig-{idx + 1}"
            page_no = 1
            prov = None
            if getattr(fig, "prov", None) and len(fig.prov) > 0:
                p_item = fig.prov[0]
                page_no = getattr(p_item, "page_no", 1)
                bbox_obj = getattr(p_item, "bbox", None)
                if bbox_obj:
                    prov = Provenance(
                        page_number=page_no,
                        bbox=BoundingBox(
                            l=float(bbox_obj.l),
                            t=float(bbox_obj.t),
                            r=float(bbox_obj.r),
                            b=float(bbox_obj.b),
                        ),
                    )
                else:
                    prov = Provenance(page_number=page_no)

            caption = None
            if getattr(fig, "caption_text", None):
                caption = str(fig.caption_text(doc))

            extracted_figures.append(
                DocumentFigure(
                    figure_id=fig_id,
                    page_number=page_no,
                    caption=caption,
                    provenance=prov,
                )
            )

        # 4. Pages
        doc_pages: List[DocumentPage] = []
        page_dict = getattr(doc, "pages", {})
        total_pages = len(page_dict) if page_dict else 1

        for page_num in range(1, total_pages + 1):
            dim = None
            if page_num in page_dict:
                p_meta = page_dict[page_num]
                if getattr(p_meta, "size", None):
                    dim = (float(p_meta.size.width), float(p_meta.size.height))

            # Associate tables and figures for this page
            p_tables = [t for t in extracted_tables if t.page_number == page_num]
            p_figures = [f for f in extracted_figures if f.page_number == page_num]

            doc_pages.append(
                DocumentPage(
                    page_number=page_num,
                    text="",  # Page-level segmented text if required
                    dimension=dim,
                    tables=p_tables,
                    figures=p_figures,
                )
            )

        has_selectable_text = len(unified_markdown.strip()) > 50

        return NormalizedDocument(
            document_id=doc_id,
            filename=file_path.name,
            mime_type=mime_type,
            text=unified_markdown,
            page_count=total_pages,
            pages=doc_pages,
            tables=extracted_tables,
            figures=extracted_figures,
            metadata={
                "parser_backend": "docling_2.x",
                "has_selectable_text": has_selectable_text,
                "ocr_applied": opts.do_ocr,
                "file_size_bytes": file_size,
                "sha256": sha256,
            },
        )
