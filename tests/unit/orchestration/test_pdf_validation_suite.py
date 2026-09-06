"""Focused PDF validation suite with three explicit test cases:

Case 1: Digital PDF fixture -> document.understand -> exact extracted text/table values and provenance.
Case 2: Scanned PDF fixture -> rasterized/image-only PDF -> Docling with do_ocr=True -> OCR recovery of key tags.
Case 3: User-facing flow -> upload PDF via FastAPI (/api/v1/files/upload) -> ask document question (/api/v1/direct/document) -> verify grounded answer and provenance.
"""

import hashlib
import io
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from PIL import Image, ImageDraw
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table
from fastapi.testclient import TestClient

from apps.api.app import create_app
from apps.api.dependencies import set_app_context
from apps.context import AppContext
from core.common.types import FinishReason
from core.inference.types import InferenceResponse, Message, TokenUsage
from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.builtin.document import (
    DoclingDocumentParser,
    DocumentParseOptions,
    DocumentUnderstandingCapability,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture
def digital_pdf_fixture(tmp_path: Path) -> Path:
    """Create a minimal digital PDF containing known text and a structured table."""
    from reportlab.lib import colors
    from reportlab.platypus import TableStyle

    pdf_path = tmp_path / "equipment_spec_digital.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()

    table_data = [
        ["Component", "Material", "Design Rating", "Corrosion Allowance"],
        ["Shell", "ASTM A516 Gr 70", "300# RF", "3.0 mm"],
        ["Tubes", "Inconel 625", "600# RF", "1.5 mm"],
        ["Channel", "Carbon Steel", "300# RF", "3.0 mm"],
    ]
    t = Table(table_data, colWidths=[100, 160, 100, 120])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("BOX", (0, 0), (-1, -1), 2, colors.black),
    ]))

    elements = [
        Paragraph("EQUIPMENT SPECIFICATION DATA SHEET", styles["Heading1"]),
        Spacer(1, 10),
        Paragraph("Asset Tag: HX-104", styles["Normal"]),
        Paragraph("Service: Reboiler Feed Preheat", styles["Normal"]),
        Paragraph("Design Pressure: 42.5 barg", styles["Normal"]),
        Paragraph("Design Temperature: 360 °C", styles["Normal"]),
        Spacer(1, 12),
        t,
        Spacer(1, 12),
        Paragraph("Notes: All materials verified against ASME Sec VIII standards.", styles["Normal"]),
    ]
    doc.build(elements)
    return pdf_path


@pytest.fixture
def scanned_pdf_fixture(tmp_path: Path) -> Path:
    """Create a rasterized/image-only PDF with zero selectable text streams."""
    pdf_path = tmp_path / "inspection_card_scanned.pdf"

    # Create an RGB raster image and draw black text onto the pixels
    img = Image.new("RGB", (800, 300), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 30), "EQUIPMENT INSPECTION TAG: FV-201A", fill="black")
    draw.text((40, 80), "SERVICE: REBOILER CONDENSATE", fill="black")
    draw.text((40, 130), "DESIGN PRESSURE: 45.0 BARG", fill="black")
    draw.text((40, 180), "VALVE BODY TYPE: GLOBE", fill="black")
    draw.text((40, 230), "MEASURED CORROSION: 0.12 MM", fill="black")

    # Save as PDF (embeds raster JPEG/Flate image inside PDF wrapper)
    img.save(str(pdf_path), "PDF")
    return pdf_path


# --------------------------------------------------------------------------- #
# Case 1: Digital PDF Parser Testing & Provenance                             #
# --------------------------------------------------------------------------- #

def test_case_1_digital_pdf_parser_and_provenance(digital_pdf_fixture: Path):
    """Case 1: Parse digital PDF -> assert exact extracted text/tables and provenance."""
    cap = DocumentUnderstandingCapability()
    ctx = CapabilityContext(execution_id="exec-digital-pdf-1")

    result = cap.execute(
        parameters={
            "file_path": str(digital_pdf_fixture),
            "do_ocr": False,
            "extract_tables": True,
        },
        inputs={},
        context=ctx,
    )

    assert result.output is not None
    output = result.output

    # 1. Exact text values extracted
    raw_text = output["text"]
    assert "EQUIPMENT SPECIFICATION DATA SHEET" in raw_text
    assert "HX-104" in raw_text
    assert "42.5 barg" in raw_text
    assert "360 °C" in raw_text

    # 2. Structured table extraction
    tables = output["tables"]
    assert len(tables) >= 1, "Expected at least 1 table extracted from digital PDF"

    # Find the equipment table
    target_table = None
    for tbl in tables:
        grid = tbl.get("grid", [])
        if len(grid) > 0 and "Component" in grid[0]:
            target_table = grid
            break

    assert target_table is not None, f"Could not find component table in extracted tables: {tables}"
    assert target_table[0] == ["Component", "Material", "Design Rating", "Corrosion Allowance"]
    assert any(row[0] == "Shell" and "A516" in row[1] and "300#" in row[2] for row in target_table)
    assert any(row[0] == "Tubes" and "Inconel 625" in row[1] for row in target_table)

    # 3. Provenance assertions
    assert len(result.references) >= 2
    text_ref = next(r for r in result.references if r.key == "text")
    table_ref = next(r for r in result.references if r.key == "tables")

    assert text_ref.uri == digital_pdf_fixture.as_uri()
    assert text_ref.mime_type == "text/markdown"
    assert text_ref.metadata["page_count"] >= 1

    assert table_ref.uri == digital_pdf_fixture.as_uri()
    assert table_ref.mime_type == "application/json"

    # Check cryptographic digest matches file on disk
    expected_sha = hashlib.sha256(digital_pdf_fixture.read_bytes()).hexdigest()
    assert output["metadata"]["sha256"] == expected_sha


# --------------------------------------------------------------------------- #
# Case 2: Scanned (Raster) PDF OCR Recovery Testing                           #
# --------------------------------------------------------------------------- #

def test_case_2_scanned_pdf_ocr_recovery(scanned_pdf_fixture: Path):
    """Case 2: Verify scanned PDF has 0 selectable text without OCR, but Docling OCR recovers tags."""
    # 1. Negative Control: Assert that standard text extraction returns empty string
    reader = PdfReader(str(scanned_pdf_fixture))
    page_text = reader.pages[0].extract_text()
    assert page_text.strip() == "", (
        "Invariant violated: Synthetic scanned PDF must contain ZERO selectable digital text. "
        f"Got: '{page_text}'"
    )

    # 2. Positive Test: Run DoclingDocumentParser with do_ocr=True
    parser = DoclingDocumentParser()
    opts = DocumentParseOptions(do_ocr=True, extract_tables=False)
    doc = parser.parse(scanned_pdf_fixture, options=opts)

    ocr_text = doc.text.upper()

    # 3. Assert OCR recovers the exact equipment tag, service, and design parameters
    assert "FV-201A" in ocr_text, f"OCR failed to recover tag FV-201A. Extracted text: {doc.text}"
    assert "45.0" in ocr_text, f"OCR failed to recover pressure 45.0. Extracted text: {doc.text}"
    assert "CONDENSATE" in ocr_text or "REBOILER" in ocr_text, (
        f"OCR failed to recover service. Extracted text: {doc.text}"
    )

    # 4. Assert provenance is maintained
    expected_sha = hashlib.sha256(scanned_pdf_fixture.read_bytes()).hexdigest()
    assert doc.sha256_checksum == expected_sha
    assert doc.page_count == 1
    assert doc.metadata["parser_backend"] == "docling_2.x"


# --------------------------------------------------------------------------- #
# Case 3: User-Facing FastAPI Flow: Upload -> Document QA                     #
# --------------------------------------------------------------------------- #

def test_case_3_user_facing_fastapi_upload_and_qa(digital_pdf_fixture: Path, tmp_path: Path):
    """Case 3: Upload PDF through /api/v1/files/upload -> ask question via /api/v1/direct/document."""
    mock_core = MagicMock()
    mock_core.repo_root = tmp_path

    # Mock connector that verifies grounded document context was passed to the LLM
    mock_connector = MagicMock()

    def mock_infer_prompt(prompt: str, system_prompt: str = "", **kwargs):
        # Assert that prompt was grounded with the extracted document text
        assert "HX-104" in prompt
        assert "42.5 barg" in prompt
        return InferenceResponse(
            request_id="qa-123",
            model_id="mock-qwen",
            message=Message.assistant(
                "Based on the equipment data sheet, the asset tag is HX-104 with a design pressure of 42.5 barg "
                "and a design temperature of 360 °C."
            ),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
            latency_ms=10.0,
        )

    mock_connector.infer_prompt.side_effect = mock_infer_prompt

    app_ctx = AppContext(core=mock_core, inference=mock_connector)
    set_app_context(app_ctx)
    client = TestClient(create_app(app_context=app_ctx))

    # Step A: Upload PDF through FastAPI /api/v1/files/upload
    pdf_bytes = digital_pdf_fixture.read_bytes()
    upload_resp = client.post(
        "/api/v1/files/upload",
        files={"file": ("equipment_spec.pdf", pdf_bytes, "application/pdf")},
    )
    assert upload_resp.status_code == 201
    upload_data = upload_resp.json()
    file_id = upload_data["file_id"]
    assert file_id is not None
    assert upload_data["filename"] == "equipment_spec.pdf"
    assert upload_data["mime_type"] == "application/pdf"

    # Step B: Ask a document question using the uploaded file_id
    qa_payload = {
        "file_id": file_id,
        "query": "What is the design pressure and asset tag for this equipment?",
        "do_ocr": False,
        "extract_tables": True,
    }
    qa_resp = client.post("/api/v1/direct/document", json=qa_payload)
    assert qa_resp.status_code == 200
    qa_data = qa_resp.json()

    assert qa_data["capability_id"] == "document.understand"
    assert qa_data["status"] == "completed"

    # Step C: Assert grounded answer returned
    output = qa_data["output"]
    assert output["query"] == "What is the design pressure and asset tag for this equipment?"
    assert "HX-104" in output["answer"]
    assert "42.5 barg" in output["answer"]

    # Step D: Assert provenance references preserved
    refs = qa_data["references"]
    assert len(refs) >= 1
    assert any(r["key"] == "text" and r["uri"].startswith("file://") for r in refs)
    assert any(r["key"] == "tables" for r in refs)
