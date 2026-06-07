"""Data models for the Douyin Video Translator."""

from app.models.job import (
    CheckpointType,
    ErrorDetail,
    JobState,
    JobStatus,
    PipelineStep,
    VideoInfo,
    VoiceOption,
)
from app.models.pipeline import (
    DownloadResult,
    SegmentAudio,
    SynthesisResult,
    TranscriptionResult,
    TranscriptionSegment,
    TranslatedSegment,
    TranslationResult,
    VocalIsolationResult,
)
from app.models.schemas import (
    CancelResponse,
    ErrorResponse,
    JobStatusResponse,
    PreviewData,
    TranscriptionPreviewSegment,
    TranslateRequest,
    TranslateResponse,
    TranslationPreviewSegment,
    VoiceOptionResponse,
)
from app.models.confirmation_schemas import (
    ConfirmResponse,
    SegmentEdit,
    TranscriptionConfirmRequest,
    TranslationConfirmRequest,
    TranslationEdit,
    VoiceConfirmRequest,
)

__all__ = [
    # Job models and enums
    "JobStatus",
    "CheckpointType",
    "PipelineStep",
    "VideoInfo",
    "VoiceOption",
    "ErrorDetail",
    "JobState",
    # Pipeline models
    "DownloadResult",
    "TranscriptionSegment",
    "TranscriptionResult",
    "TranslatedSegment",
    "TranslationResult",
    "VocalIsolationResult",
    "SegmentAudio",
    "SynthesisResult",
    # API schemas
    "TranslateRequest",
    "TranslateResponse",
    "JobStatusResponse",
    "CancelResponse",
    "ErrorResponse",
    # Preview/checkpoint schemas
    "TranscriptionPreviewSegment",
    "TranslationPreviewSegment",
    "VoiceOptionResponse",
    "PreviewData",
    # Confirmation schemas
    "SegmentEdit",
    "TranscriptionConfirmRequest",
    "TranslationEdit",
    "TranslationConfirmRequest",
    "VoiceConfirmRequest",
    "ConfirmResponse",
]
