"""Router for synchronous direct capability execution."""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.dependencies import get_app_context, get_staging_dir
from apps.api.schemas.common import ArtifactReferenceSchema, DataReferenceSchema
from apps.api.schemas.direct import (
    DirectArtifactRequest,
    DirectCapabilityResponse,
    DirectDocumentRequest,
    DirectTextAnalysisRequest,
    DirectVisionRequest,
)
from apps.context import AppContext
from orchestration.capabilities.base import CapabilityContext
from orchestration.domain.results import TaskResult

router = APIRouter(prefix="/direct", tags=["Direct Execution"])


def _resolve_file(file_id: Optional[str], file_path: Optional[str], staging_dir: Path, repo_root: Path) -> Path:
    """Safely resolve an uploaded file ID or relative path within authorized roots."""
    if file_id:
        matches = list(staging_dir.glob(f"{file_id}_*"))
        if not matches or not matches[0].is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Uploaded file '{file_id}' not found.",
            )
        return matches[0].resolve()

    if file_path:
        p = Path(file_path).resolve()
        # Verify p is inside staging_dir or repo_root
        allowed_roots = [staging_dir.resolve(), repo_root.resolve()]
        is_safe = any(p.is_relative_to(root) for root in allowed_roots)
        if not is_safe:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Access denied: path '{file_path}' is outside authorized directory roots.",
            )
        if not p.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File '{file_path}' not found.",
            )
        return p

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Either 'file_id' or 'file_path' must be provided.",
    )


def _to_response(capability_id: str, result: TaskResult) -> DirectCapabilityResponse:
    """Map domain TaskResult to DirectCapabilityResponse schema."""
    data_refs = [
        DataReferenceSchema(
            key=ref.key,
            uri=ref.uri,
            mime_type=ref.mime_type,
            metadata=ref.metadata,
        )
        for ref in result.references
    ]

    from apps.api.routers.artifacts import register_artifact

    art_refs = []
    for art in result.artifacts:
        if art.uri and art.uri.startswith("file://"):
            register_artifact(art.artifact_id, Path(art.uri.replace("file://", "")))
        art_refs.append(
            ArtifactReferenceSchema(
                artifact_id=art.artifact_id,
                name=art.name,
                uri=art.uri,
                mime_type=art.mime_type,
                size_bytes=art.size_bytes,
                download_url=f"/api/v1/artifacts/{art.artifact_id}/download",
                metadata=art.metadata,
            )
        )

    return DirectCapabilityResponse(
        capability_id=capability_id,
        status="completed",
        output=result.output,
        metadata=result.metadata,
        references=data_refs,
        artifacts=art_refs,
    )


@router.post("/vision", response_model=DirectCapabilityResponse)
async def execute_direct_vision(
    req: DirectVisionRequest,
    context: AppContext = Depends(get_app_context),
    staging_dir: Path = Depends(get_staging_dir),
) -> DirectCapabilityResponse:
    """Execute synchronous visual inspection against a P&ID drawing or image."""
    repo_root = getattr(context.core, "repo_root", Path.cwd())
    resolved = _resolve_file(req.file_id, req.file_path, staging_dir, repo_root)

    cap = context.create_vision_inspection_capability()
    cap_ctx = CapabilityContext(execution_id=f"direct-vis-{uuid.uuid4().hex[:8]}")

    parameters: Dict[str, Any] = {
        "model_id": req.model_id or "qwen3.5-9b",
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }
    inputs: Dict[str, Any] = {
        "image_path": str(resolved),
    }
    if req.query:
        inputs["query"] = req.query

    # Execute in worker thread to prevent event loop starvation
    result: TaskResult = await asyncio.to_thread(
        cap.execute,
        parameters=parameters,
        inputs=inputs,
        context=cap_ctx,
    )

    return _to_response("vision.inspect", result)


@router.post("/document", response_model=DirectCapabilityResponse)
async def execute_direct_document(
    req: DirectDocumentRequest,
    context: AppContext = Depends(get_app_context),
    staging_dir: Path = Depends(get_staging_dir),
) -> DirectCapabilityResponse:
    """Execute synchronous document parsing, OCR, and table extraction."""
    repo_root = getattr(context.core, "repo_root", Path.cwd())
    resolved = _resolve_file(req.file_id, req.file_path, staging_dir, repo_root)

    cap = context.create_document_understanding_capability()
    cap_ctx = CapabilityContext(execution_id=f"direct-doc-{uuid.uuid4().hex[:8]}")

    parameters: Dict[str, Any] = {
        "do_ocr": req.do_ocr,
        "extract_tables": req.extract_tables,
        "extract_figures": req.extract_figures,
    }
    if req.max_pages is not None:
        parameters["max_pages"] = req.max_pages

    inputs: Dict[str, Any] = {
        "file_path": str(resolved),
    }

    result: TaskResult = await asyncio.to_thread(
        cap.execute,
        parameters=parameters,
        inputs=inputs,
        context=cap_ctx,
    )

    if req.query:
        extracted_text = result.output.get("text", "") if isinstance(result.output, dict) else ""
        system_prompt = (
            "You are an industrial engineering document assistant. Answer the user's question "
            "grounded strictly in the provided document content. If the information is not present, "
            "state so clearly. Preserve exact tags, numbers, and engineering units."
        )
        user_prompt = f"Document Context:\n{extracted_text}\n\nQuestion: {req.query}"

        qa_resp = await asyncio.to_thread(
            context.inference.infer_prompt,
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=512,
        )
        answer_text = qa_resp.message.content if hasattr(qa_resp, "message") else str(qa_resp)
        if isinstance(result.output, dict):
            result.output["query"] = req.query
            result.output["answer"] = answer_text

    return _to_response("document.understand", result)


@router.post("/artifact", response_model=DirectCapabilityResponse)
async def execute_direct_artifact(
    req: DirectArtifactRequest,
    context: AppContext = Depends(get_app_context),
) -> DirectCapabilityResponse:
    """Execute synchronous deterministic artifact compilation (XLSX, DOCX, PPTX, PDF)."""
    cap = context.create_artifact_generation_capability()
    cap_ctx = CapabilityContext(execution_id=f"direct-art-{uuid.uuid4().hex[:8]}")

    parameters: Dict[str, Any] = {
        "artifact_type": req.artifact_type,
    }
    if req.filename:
        parameters["filename"] = req.filename
    if req.title:
        parameters["title"] = req.title
    if req.template:
        parameters["template"] = req.template

    inputs: Dict[str, Any] = {}
    if req.data is not None:
        inputs["data"] = req.data
    if req.content is not None:
        inputs["content"] = req.content
    if req.template_data is not None:
        inputs["template_data"] = req.template_data

    result: TaskResult = await asyncio.to_thread(
        cap.execute,
        parameters=parameters,
        inputs=inputs,
        context=cap_ctx,
    )

    return _to_response("artifact.generate", result)


@router.post("/text-analysis", response_model=DirectCapabilityResponse)
async def execute_direct_text_analysis(
    req: DirectTextAnalysisRequest,
    context: AppContext = Depends(get_app_context),
) -> DirectCapabilityResponse:
    """Execute synchronous text analysis and summarization workflow."""
    from orchestration.capabilities.builtin.workflow import TextAnalysisCapability
    from workflows.text_analysis import AnalysisDepth

    workflow = context.create_text_analysis_workflow()
    cap = TextAnalysisCapability(workflow=workflow)
    cap_ctx = CapabilityContext(execution_id=f"direct-txt-{uuid.uuid4().hex[:8]}")

    depth = req.depth or "quick"
    parameters: Dict[str, Any] = {
        "depth": depth,
    }
    if req.focus:
        parameters["focus"] = req.focus

    inputs: Dict[str, Any] = {
        "text": req.text,
    }

    result: TaskResult = await asyncio.to_thread(
        cap.execute,
        parameters=parameters,
        inputs=inputs,
        context=cap_ctx,
    )

    return _to_response("workflow.text_analysis", result)
