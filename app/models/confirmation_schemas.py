"""Pydantic schemas for confirmation API request/response models."""

from typing import Optional

from pydantic import BaseModel, Field


class SegmentEdit(BaseModel):
    """A single transcription segment edit with index and revised text."""

    index: int = Field(ge=0)
    text: str = Field(max_length=500)


class TranscriptionConfirmRequest(BaseModel):
    """Request body for POST /api/v1/jobs/{job_id}/confirm/transcription.

    If edits is None, the transcription is confirmed without changes.
    """

    edits: Optional[list[SegmentEdit]] = None


class TranslationEdit(BaseModel):
    """A single translation segment edit with index and revised translated text."""

    index: int = Field(ge=0)
    translated_text: str = Field(max_length=5000)


class TranslationConfirmRequest(BaseModel):
    """Request body for POST /api/v1/jobs/{job_id}/confirm/translation.

    If edits is None, the translation is confirmed without changes.
    """

    edits: Optional[list[TranslationEdit]] = None


class VoiceConfirmRequest(BaseModel):
    """Request body for POST /api/v1/jobs/{job_id}/confirm/voice."""

    voice_id: str = Field(min_length=1)


class ConfirmResponse(BaseModel):
    """Response body for confirmation endpoints."""

    job_id: str
    status: str
    next_step: str
    message: str
