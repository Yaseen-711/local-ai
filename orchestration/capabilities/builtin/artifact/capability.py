"""Built-in Deterministic Artifact Generation capability.

Compiles structured tabular data, markdown, and reports into XLSX, DOCX, and PDF
binaries with SHA-256 provenance tracking.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.builtin.artifact.generators import (
    DocxGenerator,
    PdfGenerator,
    PptxGenerator,
    XlsxGenerator,
)
from orchestration.capabilities.builtin.artifact.types import (
    ArtifactFormat,
    ArtifactGenerationRequest,
)
from orchestration.domain.references import ArtifactReference, DataReference
from orchestration.domain.results import TaskResult

_MIME_MAP = {
    ArtifactFormat.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ArtifactFormat.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ArtifactFormat.PPTX: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ArtifactFormat.PDF: "application/pdf",
}


def _resolve_data_input(raw: Any) -> Any:
    """Resolve data input from raw data, file path, or serialized table representation."""
    if raw is None:
        return None

    # If it is a DataReference with a URI, load file
    if isinstance(raw, DataReference) and raw.uri:
        uri_path = Path(raw.uri.replace("file://", ""))
        if uri_path.exists() and uri_path.suffix.lower() == ".json":
            return json.loads(uri_path.read_text(encoding="utf-8"))

    # If it is a list of DocumentTable dictionaries (from document.understand)
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict) and "grid" in raw[0]:
        sheets: Dict[str, List[List[Any]]] = {}
        for idx, tbl in enumerate(raw):
            name = tbl.get("table_id") or f"Table_{idx + 1}"
            grid = tbl.get("grid", [])
            sheets[name] = grid
        return sheets

    return raw


class ArtifactGenerationCapability:
    """Capability generating deterministic XLSX, DOCX, and PDF artifacts.

    Semantic contract:
        Parameters / Inputs:
            - 'artifact_type' (str, required): 'xlsx', 'docx', or 'pdf'.
            - 'filename' (str, optional): Target filename. Auto-generated if omitted.
            - 'title' (str, optional): Human-readable document/report title.
            - 'data' (optional): Tabular data, list of dicts, or table references.
            - 'content' (str, optional): Markdown/text prose for documents.
            - 'output_dir' (str, optional): Directory to store artifacts.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self._output_dir = output_dir or Path("artifacts").resolve()
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def capability_id(self) -> str:
        return "artifact.generate"

    def execute(
        self,
        parameters: Dict[str, Any],
        inputs: Dict[str, Any],
        context: CapabilityContext,
    ) -> TaskResult:
        template_name = parameters.get("template") or inputs.get("template")
        template_data = inputs.get("template_data") if "template_data" in inputs else parameters.get("template_data")

        # 1. Format
        fmt_str = parameters.get("artifact_type") or inputs.get("artifact_type")
        if not fmt_str and template_name:
            from orchestration.capabilities.builtin.artifact.templates import SUPPORTED_TEMPLATES
            inferred = SUPPORTED_TEMPLATES.get(str(template_name).lower().strip())
            if inferred:
                fmt_str = inferred.value
        fmt_str = str(fmt_str or "xlsx").lower()

        try:
            art_format = ArtifactFormat(fmt_str)
        except ValueError:
            valid = [f.value for f in ArtifactFormat]
            raise ValueError(f"Unsupported artifact_type '{fmt_str}'. Expected one of: {valid}")

        # 2. Output directory & filename
        out_dir_param = parameters.get("output_dir") or inputs.get("output_dir")
        out_dir = Path(out_dir_param).resolve() if out_dir_param else self._output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        artifact_id = f"art-{uuid.uuid4().hex[:12]}"
        default_filename = f"artifact_{artifact_id}.{art_format.value}"
        filename = str(parameters.get("filename") or inputs.get("filename") or default_filename)
        if not filename.lower().endswith(f".{art_format.value}"):
            filename = f"{filename}.{art_format.value}"

        output_path = out_dir / filename

        # 3. Data & Content resolution (supports references and small inline structures)
        raw_data = inputs.get("data") if "data" in inputs else parameters.get("data")
        data = _resolve_data_input(raw_data)

        content = inputs.get("content") or parameters.get("content")
        title = str(parameters.get("title") or inputs.get("title") or "")

        request = ArtifactGenerationRequest(
            format=art_format,
            filename=filename,
            title=title,
            data=data,
            content=content,
        )

        # 4. Invoke deterministic generator or template renderer
        if template_name:
            from orchestration.capabilities.builtin.artifact.templates import render_template
            render_template(
                template_name=str(template_name),
                template_data=template_data or data,
                output_path=output_path,
                art_format=art_format,
            )
        elif art_format == ArtifactFormat.XLSX:
            XlsxGenerator.generate(request, output_path)
        elif art_format == ArtifactFormat.DOCX:
            DocxGenerator.generate(request, output_path)
        elif art_format == ArtifactFormat.PPTX:
            PptxGenerator.generate(request, output_path)
        elif art_format == ArtifactFormat.PDF:
            PdfGenerator.generate(request, output_path)

        # 5. Compute size and cryptographic checksum
        file_bytes = output_path.read_bytes()
        size_bytes = len(file_bytes)
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()
        mime_type = _MIME_MAP.get(art_format, "application/octet-stream")
        file_uri = output_path.as_uri()

        art_ref = ArtifactReference(
            artifact_id=artifact_id,
            name=filename,
            uri=file_uri,
            mime_type=mime_type,
            size_bytes=size_bytes,
            metadata={"sha256": sha256_hash, "format": art_format.value},
        )

        output_payload = {
            "artifact_id": artifact_id,
            "name": filename,
            "uri": file_uri,
            "size_bytes": size_bytes,
            "sha256": sha256_hash,
            "mime_type": mime_type,
        }

        return TaskResult(
            output=output_payload,
            artifacts=[art_ref],
            metadata={"format": art_format.value, "sha256": sha256_hash},
        )
