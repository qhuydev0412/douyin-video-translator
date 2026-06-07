"""FastAPI REST API routes for the Douyin Video Translator."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.status import HTTP_202_ACCEPTED, HTTP_404_NOT_FOUND

from app.api.dependencies import RateLimiter
from app.models.job import CheckpointType, JobState, JobStatus
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
from app.services.downloader import VideoDownloader

if TYPE_CHECKING:
    from app.services.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["translate"])


class JobStoreProtocol(Protocol):
    """Protocol for job store operations used by routes."""

    def get_job(self, job_id: str) -> JobState: ...

    def create_job(self, job_id: str, url: str, client_ip: str, work_dir: str) -> JobState: ...

    def update_job(self, job_id: str, **kwargs: object) -> None: ...

    def count_active_jobs(self, client_ip: str) -> int: ...


class TaskEnqueuer(Protocol):
    """Protocol for enqueueing Celery tasks (allows testing without Celery)."""

    def enqueue(self, job_id: str, url: str) -> str: ...


class CeleryTaskEnqueuer:
    """Production task enqueuer using Celery."""

    def enqueue(self, job_id: str, url: str) -> str:
        """Enqueue a translate_video task via Celery.

        Returns:
            The Celery task ID.
        """
        from app.tasks.translation_task import translate_video_task

        result = translate_video_task.delay(job_id, url)
        return result.id


# Module-level state (set during app setup via configure_routes)
_job_store: JobStoreProtocol | None = None
_rate_limiter: RateLimiter | None = None
_task_enqueuer: TaskEnqueuer | None = None
_downloader: VideoDownloader | None = None
_checkpoint_manager: "CheckpointManager | None" = None


def configure_routes(
    job_store: JobStoreProtocol,
    task_enqueuer: TaskEnqueuer | None = None,
    checkpoint_manager: "CheckpointManager | None" = None,
) -> None:
    """Configure route dependencies. Called during app startup.

    Args:
        job_store: Job state store implementation.
        task_enqueuer: Task enqueuer (defaults to CeleryTaskEnqueuer).
        checkpoint_manager: Optional CheckpointManager for checkpoint operations.
    """
    global _job_store, _rate_limiter, _task_enqueuer, _downloader, _checkpoint_manager  # noqa: PLW0603
    _job_store = job_store
    _rate_limiter = RateLimiter(job_store)
    _task_enqueuer = task_enqueuer or CeleryTaskEnqueuer()
    _downloader = VideoDownloader()
    _checkpoint_manager = checkpoint_manager


def _get_job_store() -> JobStoreProtocol:
    """Dependency to get the configured job store."""
    if _job_store is None:
        raise RuntimeError("Routes not configured. Call configure_routes() first.")
    return _job_store


def _get_rate_limiter() -> RateLimiter:
    """Dependency to get the configured rate limiter."""
    if _rate_limiter is None:
        raise RuntimeError("Routes not configured. Call configure_routes() first.")
    return _rate_limiter


def _get_task_enqueuer() -> TaskEnqueuer:
    """Dependency to get the configured task enqueuer."""
    if _task_enqueuer is None:
        raise RuntimeError("Routes not configured. Call configure_routes() first.")
    return _task_enqueuer


def _get_downloader() -> VideoDownloader:
    """Dependency to get the video downloader."""
    if _downloader is None:
        raise RuntimeError("Routes not configured. Call configure_routes() first.")
    return _downloader


def _get_checkpoint_manager_optional() -> "CheckpointManager | None":
    """Get the configured checkpoint manager (may be None)."""
    return _checkpoint_manager


@router.post(
    "/translate",
    response_model=TranslateResponse,
    status_code=HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid URL"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
def create_translation_job(
    body: TranslateRequest,
    request: Request,
    job_store: JobStoreProtocol = Depends(_get_job_store),
    rate_limiter: RateLimiter = Depends(_get_rate_limiter),
    task_enqueuer: TaskEnqueuer = Depends(_get_task_enqueuer),
    downloader: VideoDownloader = Depends(_get_downloader),
) -> TranslateResponse:
    """Accept a Douyin URL and create a translation job.

    Validates the URL, checks rate limits, creates a job record,
    and enqueues the translation task.

    Returns HTTP 202 with job_id on success.
    """
    # Validate URL
    if not downloader.validate_url(body.url):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_URL",
                "message": "URL không hợp lệ, vui lòng cung cấp link Douyin",
                "step": None,
                "retryable": False,
                "retry_after": None,
            },
        )

    # Check rate limit
    rate_limiter.check(request)

    # Create job
    job_id = str(uuid.uuid4())
    client_ip = _get_client_ip(request)
    work_dir = f"storage/jobs/{job_id}"

    job_store.create_job(
        job_id=job_id,
        url=body.url,
        client_ip=client_ip,
        work_dir=work_dir,
    )

    # Enqueue Celery task
    task_enqueuer.enqueue(job_id, body.url)

    logger.info("Created translation job %s for URL %s from IP %s", job_id, body.url, client_ip)

    return TranslateResponse(
        job_id=job_id,
        status=JobStatus.QUEUED.value,
        message="Đã tiếp nhận yêu cầu dịch video",
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    responses={404: {"model": ErrorResponse, "description": "Job not found"}},
)
def get_job_status(
    job_id: str,
    job_store: JobStoreProtocol = Depends(_get_job_store),
) -> JobStatusResponse:
    """Get the current status of a translation job.

    Returns job details including progress, current step, and download URL.
    When the job is at a checkpoint, includes preview data and resets the
    24-hour expiration timer.
    """
    try:
        job = job_store.get_job(job_id)
    except KeyError:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail={
                "error": "JOB_NOT_FOUND",
                "message": f"Không tìm thấy job với ID: {job_id}",
                "step": None,
                "retryable": False,
                "retry_after": None,
            },
        )

    # Handle checkpoint preview data
    checkpoint_type_value: str | None = None
    preview_data: PreviewData | None = None

    if job.status == JobStatus.AWAITING_CONFIRMATION and job.checkpoint_type:
        checkpoint_type_value = job.checkpoint_type.value

        # Reset expiration timer (Requirement 7.3)
        checkpoint_manager = _get_checkpoint_manager_optional()
        if checkpoint_manager:
            checkpoint_manager.reset_expiration(job_id)

        # Load preview data based on checkpoint type
        if job.checkpoint_type == CheckpointType.TRANSCRIPTION:
            preview_data = _load_transcription_preview(job)
        elif job.checkpoint_type == CheckpointType.TRANSLATION:
            preview_data = _load_translation_preview(job)
        elif job.checkpoint_type == CheckpointType.VOICE_SELECTION:
            preview_data = _load_voice_selection_preview(job)

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        current_step=job.current_step.value if job.current_step else None,
        progress_percent=job.progress_percent,
        video_info=job.video_info,
        download_url=job.download_url,
        error=job.error,
        created_at=job.created_at,
        expires_at=job.expires_at,
        checkpoint_type=checkpoint_type_value,
        preview_data=preview_data,
    )


@router.delete(
    "/jobs/{job_id}",
    response_model=CancelResponse,
    responses={404: {"model": ErrorResponse, "description": "Job not found"}},
)
def cancel_job(
    job_id: str,
    job_store: JobStoreProtocol = Depends(_get_job_store),
) -> CancelResponse:
    """Cancel a translation job.

    Updates job status to CANCELLED and revokes the Celery task.
    """
    try:
        job_store.get_job(job_id)
    except KeyError:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail={
                "error": "JOB_NOT_FOUND",
                "message": f"Không tìm thấy job với ID: {job_id}",
                "step": None,
                "retryable": False,
                "retry_after": None,
            },
        )

    # Update status to CANCELLED
    job_store.update_job(job_id, status=JobStatus.CANCELLED)

    # Revoke Celery task (best-effort, task may already have completed)
    try:
        from app.tasks.translation_task import revoke_task

        revoke_task(job_id, terminate=True)
    except Exception as exc:
        logger.warning("Failed to revoke task for job %s: %s", job_id, exc)

    return CancelResponse(
        job_id=job_id,
        status=JobStatus.CANCELLED.value,
    )


def _load_transcription_preview(job: JobState) -> PreviewData | None:
    """Load transcription segments from the job's artifact file.

    Reads the transcription JSON and maps segments to TranscriptionPreviewSegment.
    """
    transcription_path_str = job.artifacts.get("transcription_path")
    if not transcription_path_str:
        return None

    transcription_path = Path(transcription_path_str)
    if not transcription_path.exists():
        return None

    try:
        with open(transcription_path, "r", encoding="utf-8") as f:
            transcription_data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load transcription preview for job %s: %s", job.job_id, exc)
        return None

    segments = transcription_data.get("segments", [])
    preview_segments = [
        TranscriptionPreviewSegment(
            index=i,
            start=seg["start"],
            end=seg["end"],
            text=seg["text"],
            confidence=seg.get("confidence", 0.0),
        )
        for i, seg in enumerate(segments)
    ]

    return PreviewData(transcription_segments=preview_segments)


def _load_translation_preview(job: JobState) -> PreviewData | None:
    """Load translation segments from the job's artifact file.

    Reads the translation JSON and maps segments to TranslationPreviewSegment.
    """
    translation_path_str = job.artifacts.get("translation_path")
    if not translation_path_str:
        return None

    translation_path = Path(translation_path_str)
    if not translation_path.exists():
        return None

    try:
        with open(translation_path, "r", encoding="utf-8") as f:
            translation_data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load translation preview for job %s: %s", job.job_id, exc)
        return None

    segments = translation_data.get("segments", [])
    preview_segments = [
        TranslationPreviewSegment(
            index=i,
            start=seg["start"],
            end=seg["end"],
            original_text=seg["text"],
            translated_text=seg["translated_text"],
        )
        for i, seg in enumerate(segments)
    ]

    return PreviewData(translation_segments=preview_segments)


def _load_voice_selection_preview(job: JobState) -> PreviewData | None:
    """Build voice selection preview from the job's voice_options field."""
    if not job.voice_options:
        return None

    voice_options = [
        VoiceOptionResponse(
            voice_id=opt.voice_id,
            voice_name=opt.voice_name,
            preview_url=opt.preview_url,
        )
        for opt in job.voice_options
    ]

    return PreviewData(voice_options=voice_options)


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request, considering proxy headers."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
