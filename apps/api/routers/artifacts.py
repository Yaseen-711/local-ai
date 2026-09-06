"""Router for artifact retrieval and binary downloads."""

import hashlib
import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from apps.api.dependencies import get_artifacts_dir
from apps.api.schemas.common import ArtifactReferenceSchema

router = APIRouter(prefix="/artifacts", tags=["Artifacts"])


_ARTIFACT_REGISTRY: dict[str, Path] = {}


def register_artifact(artifact_id: str, path: Path) -> None:
    """Register an artifact's physical path by ID for reliable lookup."""
    _ARTIFACT_REGISTRY[artifact_id] = path.resolve()


def _find_artifact_file(artifact_id: str, artifacts_dir: Path) -> Path:
    """Safely locate an artifact file by ID within the authorized artifacts directory."""
    target: Path | None = None

    if artifact_id in _ARTIFACT_REGISTRY:
        candidate = _ARTIFACT_REGISTRY[artifact_id]
        if candidate.is_file():
            target = candidate

    if target is None:
        matches = list(artifacts_dir.glob(f"*{artifact_id}*"))
        if matches and matches[0].is_file():
            target = matches[0]

    if target is None:
        direct = (artifacts_dir / artifact_id).resolve()
        if direct.is_file():
            target = direct

    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact '{artifact_id}' not found.",
        )

    resolved = target.resolve()
    if not resolved.is_relative_to(artifacts_dir.resolve()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Access denied: artifact path traversal detected.",
        )
    return resolved


@router.get("/{artifact_id}", response_model=ArtifactReferenceSchema)
async def get_artifact_metadata(
    artifact_id: str,
    artifacts_dir: Path = Depends(get_artifacts_dir),
) -> ArtifactReferenceSchema:
    """Retrieve metadata for a generated artifact."""
    target = _find_artifact_file(artifact_id, artifacts_dir)
    mime, _ = mimetypes.guess_type(target.name)
    mime = mime or "application/octet-stream"
    size = target.stat().st_size

    h = hashlib.sha256()
    with open(target, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    digest = h.hexdigest()

    return ArtifactReferenceSchema(
        artifact_id=artifact_id,
        name=target.name,
        uri=target.as_uri(),
        mime_type=mime,
        size_bytes=size,
        download_url=f"/api/v1/artifacts/{artifact_id}/download",
        metadata={"sha256": digest},
    )


@router.get("/{artifact_id}/download")
async def download_artifact(
    artifact_id: str,
    artifacts_dir: Path = Depends(get_artifacts_dir),
) -> FileResponse:
    """Download the binary file of a generated artifact with Content-Disposition headers."""
    target = _find_artifact_file(artifact_id, artifacts_dir)
    mime, _ = mimetypes.guess_type(target.name)
    mime = mime or "application/octet-stream"

    h = hashlib.sha256()
    with open(target, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    digest = h.hexdigest()

    return FileResponse(
        path=target,
        media_type=mime,
        filename=target.name,
        headers={"ETag": f'"{digest}"'},
    )
