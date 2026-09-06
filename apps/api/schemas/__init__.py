"""MRPL API Schemas package."""

from apps.api.schemas.common import (
    ArtifactReferenceSchema,
    DataReferenceSchema,
    ProblemDetail,
)
from apps.api.schemas.direct import (
    DirectArtifactRequest,
    DirectCapabilityResponse,
    DirectDocumentRequest,
    DirectTextAnalysisRequest,
    DirectVisionRequest,
)
from apps.api.schemas.files import FileMetadataResponse, FileUploadResponse
from apps.api.schemas.goals import (
    CandidatePlanResponse,
    CancelGoalResponse,
    CreateGoalRequest,
    GoalDetailResponse,
    GoalExecutionResponse,
    GoalResponse,
    TaskSchema,
)

__all__ = [
    "ArtifactReferenceSchema",
    "CandidatePlanResponse",
    "CancelGoalResponse",
    "CreateGoalRequest",
    "DataReferenceSchema",
    "DirectArtifactRequest",
    "DirectCapabilityResponse",
    "DirectDocumentRequest",
    "DirectTextAnalysisRequest",
    "DirectVisionRequest",
    "FileMetadataResponse",
    "FileUploadResponse",
    "GoalDetailResponse",
    "GoalExecutionResponse",
    "GoalResponse",
    "ProblemDetail",
    "TaskSchema",
]
