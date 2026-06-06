"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.models.job import ErrorDetail, VideoInfo


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
