"""Built-in vision inspection capability.

Bridges engineering drawing and visual inspection tasks to multimodal model inference.
Accepts a local image path and query, constructs normalized multimodal messages with
MediaAttachment, and returns structured inspection observations with cryptographic provenance.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from connectors.inference import InferenceConnector
from core.inference.types import (
    GenerationOptions,
    InferenceRequest,
    MediaAttachment,
    Message,
)
from orchestration.capabilities.base import CapabilityContext
from orchestration.domain.references import DataReference
from orchestration.domain.results import TaskResult


DEFAULT_VISION_PROMPT = (
    "Carefully analyze this engineering drawing or inspection image in detail. "
    "Identify all visible equipment tags, symbols, instrument tags, and flow directions. "
    "Provide clear, grounded observations based strictly on what is visible."
)


class VisionInspectionCapability:
    """Capability executing single-shot multimodal vision inspection via InferenceConnector.

    Semantic contract:
        Required inputs/parameters:
            - 'image_path' (or 'path'): str pointing to local image file.
        Optional inputs/parameters:
            - 'query' (or 'prompt'): str inspection instruction.
            - 'model_id': str (default: 'default')
            - 'temperature': Optional[float]
            - 'max_tokens': Optional[int]
    """

    def __init__(self, connector: InferenceConnector) -> None:
        """Initialize with an InferenceConnector implementation.

        Args:
            connector: Object satisfying InferenceConnector protocol.
        """
        self._connector = connector

    @property
    def capability_id(self) -> str:
        """Canonical identifier for vision inspection."""
        return "vision.inspect"

    def execute(
        self,
        parameters: Dict[str, Any],
        inputs: Dict[str, Any],
        context: CapabilityContext,
    ) -> TaskResult:
        """Execute multimodal image inspection.

        Args:
            parameters: Configuration parameters (model_id, temperature, etc.).
            inputs: Data inputs (image_path, query, etc.).
            context: Invocation context with execution_id.

        Returns:
            TaskResult with visual analysis text, metadata, and DataReference provenance.

        Raises:
            ValueError: If image_path is missing, invalid, or unsupported.
            FileNotFoundError: If the image file does not exist.
        """
        image_path_raw = (
            inputs.get("image_path")
            or parameters.get("image_path")
            or inputs.get("path")
            or parameters.get("path")
        )
        if not image_path_raw or not isinstance(image_path_raw, (str, Path)):
            raise ValueError(
                f"Capability '{self.capability_id}' requires a valid 'image_path' "
                f"in inputs or parameters."
            )

        resolved_path = Path(image_path_raw).resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Image file not found for vision inspection: {resolved_path}")

        attachment = MediaAttachment.from_file(resolved_path)

        query = (
            inputs.get("query")
            or parameters.get("query")
            or inputs.get("prompt")
            or parameters.get("prompt")
            or DEFAULT_VISION_PROMPT
        )
        if not isinstance(query, str) or not query.strip():
            query = DEFAULT_VISION_PROMPT

        model_id = str(parameters.get("model_id") or inputs.get("model_id") or "default")
        temp = parameters.get("temperature", 0.1)
        max_tokens = parameters.get("max_tokens", 1024)

        options = GenerationOptions(
            temperature=float(temp) if temp is not None else 0.1,
            max_tokens=int(max_tokens) if max_tokens is not None else 1024,
        )

        user_message = Message.user(
            content=query.strip(),
            attachments=[attachment],
        )

        request = InferenceRequest(
            model_id=model_id,
            messages=[user_message],
            options=options,
            request_id=context.execution_id,
        )

        response = self._connector.infer(request)

        data_ref = DataReference(
            key="inspected_image",
            uri=f"file://{attachment.source_path}",
            mime_type=attachment.mime_type,
            metadata={
                "sha256": attachment.sha256,
                "size_bytes": attachment.size_bytes,
                "filename": attachment.name,
            },
        )

        return TaskResult(
            output=response.text,
            metadata={
                "model_id": response.model_id,
                "image_sha256": attachment.sha256,
                "image_path": attachment.source_path,
                "tokens_used": response.usage.total_tokens if response.usage else None,
                "latency_ms": response.latency_ms,
            },
            references=[data_ref],
        )
