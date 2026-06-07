"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.models.job import ErrorDetail, VideoInfo


class TranscriptionPreviewSegment(BaseModel):
    """A single transcription segment for checkpoint preview."""

    index: int
    start: float
    end: float
    text: str
    confidence: float


class TranslationPreviewSegment(BaseModel):
    """A single translation segment for checkpoint preview."""

    index: int
    start: float
    end: float
    original_text: str
    translated_text: str


class VoiceOptionResponse(BaseModel):
    """A voice option returned in the checkpoint preview."""

    voice_id: str
    voice_name: str
    preview_url: str


class PreviewData(BaseModel):
    """Preview data included in the job status response at checkpoints."""

    transcription_segments: Optional[list[TranscriptionPreviewSegment]] = None
    translation_segments: Optional[list[TranslationPreviewSegment]] = None
    voice_options: Optional[list[VoiceOptionResponse]] = None


class TranslateRequest(BaseModel):
    """Request body for POST /api/v1/translate."""

    url: str


class TranslateResponse(BaseModel):
    """Response body for POST /api/v1/translate (HTTP 202)."""

    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Response body for GET /api/v1/jobs/{job_id}."""

    job_id: str
    status: str
    current_step: Optional[str] = None
    progress_percent: int
    video_info: Optional[VideoInfo] = None
    download_url: Optional[str] = None
    error: Optional[ErrorDetail] = None
    created_at: datetime
    expires_at: Optional[datetime] = None
    checkpoint_type: Optional[str] = None
    preview_data: Optional[PreviewData] = None


class CancelResponse(BaseModel):
    """Response body for DELETE /api/v1/jobs/{job_id}."""

    job_id: str
    status: str


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str
    message: str
    step: Optional[str] = None
    retryable: bool
    retry_after: Optional[int] = None
