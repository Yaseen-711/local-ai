"""Router for file upload, staging, and reference resolution."""

from datetime import datetime, timezone
import hashlib
import mimetypes
from pathlib import Path
import re
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from apps.api.dependencies import get_staging_dir
from apps.api.schemas.files import FileMetadataResponse, FileUploadResponse
from core.inference.types import DEFAULT_MAX_ATTACHMENT_BYTES, SUPPORTED_IMAGE_MIME_TYPES

router = APIRouter(prefix="/files", tags=["Files"])

MAX_DOCUMENT_BYTES = 50 * 1024 * 1024  # 50 MiB for PDFs, spreadsheets, docs


def _sanitize_filename(name: str) -> str:
    """Sanitize filename to prevent directory traversal or invalid characters."""
    base = Path(name).name
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", base)
    return sanitized or "unnamed_file"


@router.post("/upload", response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    staging_dir: Path = Depends(get_staging_dir),
) -> FileUploadResponse:
    """Upload and stage a confidential industrial document or drawing.
    
    Streams chunks in 64 KiB blocks to calculate cryptographic SHA-256
    without loading the entire payload into unmanaged memory.
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename missing in upload.")

    safe_name = _sanitize_filename(file.filename)
    file_id = f"file-{uuid.uuid4().hex[:12]}"
    target_path = staging_dir / f"{file_id}_{safe_name}"

    detected_mime, _ = mimetypes.guess_type(safe_name)
    detected_mime = (file.content_type or detected_mime or "application/octet-stream").lower()

    # Determine limit based on type
    is_image = detected_mime in SUPPORTED_IMAGE_MIME_TYPES or target_path.suffix.lower() in {
        ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"
    }
    max_bytes = DEFAULT_MAX_ATTACHMENT_BYTES if is_image else MAX_DOCUMENT_BYTES

    hasher = hashlib.sha256()
    size_bytes = 0

    try:
        with open(target_path, "wb") as f_out:
            while True:
                chunk = await file.read(65536)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    target_path.unlink(missing_ok=True)
                    limit_mib = max_bytes // (1024 * 1024)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File size exceeds allowed limit of {limit_mib} MiB.",
                    )
                hasher.update(chunk)
                f_out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        target_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stream and store upload: {exc}",
        ) from exc

    if size_bytes <= 0:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    digest = hasher.hexdigest()
    created_at = datetime.now(timezone.utc)

    return FileUploadResponse(
        file_id=file_id,
        filename=safe_name,
        mime_type=detected_mime,
        size_bytes=size_bytes,
        sha256=digest,
        uri=target_path.as_uri(),
        created_at=created_at,
    )


@router.get("/{file_id}", response_model=FileMetadataResponse)
async def get_file_metadata(
    file_id: str,
    staging_dir: Path = Depends(get_staging_dir),
) -> FileMetadataResponse:
    """Retrieve metadata for a previously uploaded file."""
    matches = list(staging_dir.glob(f"{file_id}_*"))
    if not matches or not matches[0].is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File '{file_id}' not found.")

    target = matches[0]
    filename = target.name[len(file_id) + 1 :]
    mime, _ = mimetypes.guess_type(filename)
    mime = mime or "application/octet-stream"
    size = target.stat().st_size

    # Compute digest
    h = hashlib.sha256()
    with open(target, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)

    return FileMetadataResponse(
        file_id=file_id,
        filename=filename,
        mime_type=mime,
        size_bytes=size,
        sha256=h.hexdigest(),
        uri=target.as_uri(),
        exists=True,
    )
