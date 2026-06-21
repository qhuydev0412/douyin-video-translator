"""Job state models and enums for the Douyin Video Translator pipeline."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of a translation job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class CheckpointType(str, Enum):
    """Types of checkpoints where the pipeline pauses for user confirmation."""

    TRANSCRIPTION = "transcription"
    TRANSLATION = "translation"
    VOICE_SELECTION = "voice_selection"
    AUDIO_PREVIEW = "audio_preview"


class PipelineStep(str, Enum):
    """Steps in the translation pipeline."""

    DOWNLOADING = "downloading"
    EXTRACTING_AUDIO = "extracting_audio"
    ISOLATING_VOCALS = "isolating_vocals"
    RECOGNIZING_SPEECH = "recognizing_speech"
    TRANSLATING = "translating"
    SYNTHESIZING_VOICE = "synthesizing_voice"
    COMPOSING_VIDEO = "composing_video"


class VideoInfo(BaseModel):
    """Metadata about the downloaded video."""

    duration_seconds: float
    file_size_bytes: int
    resolution: str
    title: Optional[str] = None


class VoiceOption(BaseModel):
    """A voice option for TTS preview with name, ID, and audio preview URL."""

    voice_id: str
    voice_name: str
    preview_url: str


class ErrorDetail(BaseModel):
    """Details about an error that occurred during pipeline execution."""

    step: PipelineStep
    message: str
    retryable: bool
    retry_count: int


class JobState(BaseModel):
    """Full state of a translation job, stored in Redis."""

    job_id: str
    url: str
    status: JobStatus = JobStatus.QUEUED
    current_step: Optional[PipelineStep] = None
    progress_percent: int = Field(default=0, ge=0, le=100)
    video_info: Optional[VideoInfo] = None
    download_url: Optional[str] = None
    error: Optional[ErrorDetail] = None
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    work_dir: str
    artifacts: dict[str, str] = Field(default_factory=dict)
    checkpoint_type: Optional[CheckpointType] = None
    checkpoint_entered_at: Optional[datetime] = None
    confirmation_lock: bool = False
    voice_options: Optional[list[VoiceOption]] = None
