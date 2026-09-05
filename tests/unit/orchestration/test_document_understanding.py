"""Unit tests for Document Understanding capability and parsers."""

from pathlib import Path
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.builtin.document import (
    BoundingBox,
    DoclingDocumentParser,
    DocumentFigure,
    DocumentPage,
    DocumentParseOptions,
    DocumentParser,
    DocumentTable,
    DocumentUnderstandingCapability,
    FallbackDocumentParser,
    NormalizedDocument,
    Provenance,
)
from orchestration.domain.references import DataReference


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "financial_report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Annual Financial Summary Report", styles["Heading1"]),
        Spacer(1, 12),
        Paragraph("This document contains quarterly revenue breakdown for FY2025.", styles["Normal"]),
        Spacer(1, 12),
        Table([
            ["Quarter", "Revenue", "Expenses", "Profit"],
            ["Q1", "$100,000", "$60,000", "$40,000"],
            ["Q2", "$120,000", "$70,000", "$50,000"],
        ]),
    ]
    doc.build(elements)
    return pdf_path


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("Meeting notes:\nDiscussed project timeline and quarterly deliverables.")
    return txt_path


@pytest.fixture
def sample_csv_file(tmp_path: Path) -> Path:
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("Metric,Value,Target\nAccuracy,0.95,0.90\nLatency,120,150\n")
    return csv_path


def test_normalized_document_serialization():
    """Verify NormalizedDocument serialization to and from dictionary."""
    doc = NormalizedDocument(
        document_id="doc-test-123",
        filename="test.pdf",
        mime_type="application/pdf",
        text="Hello world",
        page_count=1,
        pages=[
            DocumentPage(
                page_number=1,
                text="Hello world",
                dimension=(612.0, 792.0),
                tables=[
                    DocumentTable(
                        table_id="tbl-1",
                        page_number=1,
                        num_rows=2,
                        num_cols=2,
                        grid=[["A", "B"], ["C", "D"]],
                        markdown="| A | B |\n| C | D |",
                        provenance=Provenance(page_number=1),
                    )
                ],
                figures=[
                    DocumentFigure(
                        figure_id="fig-1",
                        page_number=1,
                        caption="Sample Figure",
                        provenance=Provenance(
                            page_number=1,
                            bbox=BoundingBox(l=0, t=0, r=100, b=100, coord_origin="TOPLEFT"),
                        ),
                    )
                ],
            )
        ],
        tables=[
            DocumentTable(
                table_id="tbl-1",
                page_number=1,
                num_rows=2,
                num_cols=2,
                grid=[["A", "B"], ["C", "D"]],
                provenance=Provenance(page_number=1),
            )
        ],
        metadata={"file_size_bytes": 1024, "sha256": "abcdef123456"},
    )

    data = doc.to_dict()
    assert data["document_id"] == "doc-test-123"
    assert data["text"] == "Hello world"
    assert len(data["pages"]) == 1
    assert len(data["tables"]) == 1
    assert data["tables"][0]["grid"] == [["A", "B"], ["C", "D"]]
    assert doc.sha256_checksum == "abcdef123456"
    assert doc.file_size_bytes == 1024

    restored = NormalizedDocument.from_dict(data)
    assert restored.document_id == doc.document_id
    assert restored.filename == doc.filename
    assert restored.pages[0].page_number == 1
    assert restored.tables[0].grid == [["A", "B"], ["C", "D"]]
    assert restored.sha256_checksum == "abcdef123456"
    assert restored.file_size_bytes == 1024


def test_fallback_parser_text_file(sample_text_file: Path):
    """Verify FallbackDocumentParser on plain text file."""
    parser = FallbackDocumentParser()
    assert isinstance(parser, DocumentParser)

    doc = parser.parse(sample_text_file)
    assert doc.filename == "notes.txt"
    assert "Meeting notes:" in doc.text
    assert doc.page_count == 1
    assert len(doc.pages) == 1
    assert doc.sha256_checksum != ""
    assert doc.file_size_bytes > 0


def test_fallback_parser_csv_file(sample_csv_file: Path):
    """Verify FallbackDocumentParser on CSV file extracts structured table."""
    parser = FallbackDocumentParser()
    doc = parser.parse(sample_csv_file)

    assert doc.filename == "metrics.csv"
    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert table.num_rows == 3
    assert table.num_cols == 3
    assert table.grid[0] == ["Metric", "Value", "Target"]
    assert table.grid[1] == ["Accuracy", "0.95", "0.90"]
    assert "| Metric | Value | Target |" in table.markdown


def test_fallback_parser_pdf(sample_pdf: Path):
    """Verify FallbackDocumentParser on generated digital PDF."""
    parser = FallbackDocumentParser()
    doc = parser.parse(sample_pdf)

    assert doc.filename == "financial_report.pdf"
    assert doc.page_count >= 1
    assert "Annual Financial Summary Report" in doc.text
    assert doc.metadata["has_selectable_text"] is True
    assert doc.sha256_checksum != ""


def test_docling_parser_pdf(sample_pdf: Path):
    """Verify DoclingDocumentParser parsing digital PDF with layout and structure."""
    parser = DoclingDocumentParser()
    assert isinstance(parser, DocumentParser)

    options = DocumentParseOptions(do_ocr=False, extract_tables=True)
    doc = parser.parse(sample_pdf, options=options)

    assert doc.filename == "financial_report.pdf"
    assert doc.page_count >= 1
    assert "Annual Financial Summary Report" in doc.text
    assert doc.metadata["parser_backend"] == "docling_2.x"
    assert doc.sha256_checksum != ""

    # Verify no Docling internal classes leaked into NormalizedDocument
    for p in doc.pages:
        assert isinstance(p, DocumentPage)
    for t in doc.tables:
        assert isinstance(t, DocumentTable)
        assert isinstance(t.grid, list)


def test_document_capability_with_file_path(sample_pdf: Path):
    """Verify DocumentUnderstandingCapability execution using parameters.file_path."""
    cap = DocumentUnderstandingCapability()
    assert cap.capability_id == "document.understand"

    ctx = CapabilityContext(execution_id="exec-doc-1")
    result = cap.execute(
        parameters={"file_path": str(sample_pdf), "force_fallback": True},
        inputs={},
        context=ctx,
    )

    assert result.output is not None
    assert "text" in result.output
    assert "pages" in result.output
    assert "tables" in result.output
    assert "Annual Financial Summary Report" in result.output["text"]
    assert len(result.references) == 2
    assert result.references[0].mime_type == "text/markdown"
    assert result.references[1].mime_type == "application/json"


def test_document_capability_with_data_reference(sample_pdf: Path):
    """Verify DocumentUnderstandingCapability resolving a DataReference input."""
    cap = DocumentUnderstandingCapability()
    ctx = CapabilityContext(execution_id="exec-doc-2")

    ref = DataReference(
        key="report_file",
        uri=f"file://{sample_pdf.resolve()}",
        mime_type="application/pdf",
    )

    result = cap.execute(
        parameters={"force_fallback": True},
        inputs={"document": ref},
        context=ctx,
    )

    assert result.output is not None
    assert "Annual Financial Summary Report" in result.output["text"]
    assert result.metadata["parser_backend"] == "fallback_pypdf"


def test_document_capability_file_not_found():
    """Verify proper exception when target document file does not exist."""
    cap = DocumentUnderstandingCapability()
    ctx = CapabilityContext(execution_id="exec-doc-3")

    with pytest.raises(FileNotFoundError):
        cap.execute(
            parameters={"file_path": "/non/existent/path/doc.pdf"},
            inputs={},
            context=ctx,
        )


def test_document_capability_missing_input():
    """Verify error when no file_path, document_uri, or document reference provided."""
    cap = DocumentUnderstandingCapability()
    ctx = CapabilityContext(execution_id="exec-doc-4")

    with pytest.raises(ValueError, match="requires a document file reference"):
        cap.execute(parameters={}, inputs={}, context=ctx)
