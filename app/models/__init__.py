"""Data models for the Douyin Video Translator."""

from app.models.job import ErrorDetail, JobState, JobStatus, PipelineStep, VideoInfo
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
    TranslateRequest,
    TranslateResponse,
)

__all__ = [
    # Job models and enums
    "JobStatus",
    "PipelineStep",
    "VideoInfo",
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
]
