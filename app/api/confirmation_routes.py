"""Confirmation API routes for pipeline checkpoint confirmations.

Provides endpoints for confirming transcription, translation, and voice selection
checkpoints during the pipeline preview & confirm workflow.
"""

import logging
from typing import Protocol

from fastapi import APIRouter, Depends, HTTPException

from app.models.confirmation_schemas import (
    AudioConfirmRequest,
    AudioPreviewEdit,
    ConfirmResponse,
    TranscriptionConfirmRequest,
    TranslationConfirmRequest,
    VoiceConfirmRequest,
)
from app.models.job import CheckpointType, JobState, JobStatus, PipelineStep
from app.services.checkpoint_manager import (
    CheckpointManager,
    ConfirmationInProgressError,
    InvalidSegmentIndexError,
    NotAwaitingConfirmationError,
    WrongCheckpointError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/jobs", tags=["confirmation"])


class ResumeTaskEnqueuer(Protocol):
    """Protocol for enqueueing pipeline resume tasks (allows testing without Celery)."""

    def enqueue(self, job_id: str, next_step: str) -> str: ...


class CeleryResumeEnqueuer:
    """Production resume task enqueuer using Celery."""

    def enqueue(self, job_id: str, next_step: str) -> str:
        """Enqueue a resume_pipeline_task via Celery.

        Returns:
            The Celery task ID.
        """
        from app.tasks.resume_task import resume_pipeline_task

        result = resume_pipeline_task.delay(job_id, next_step)
        return result.id


# Module-level state (set during app setup via configure_confirmation_routes)
_checkpoint_manager: CheckpointManager | None = None
_task_enqueuer: ResumeTaskEnqueuer | None = None


def configure_confirmation_routes(
    checkpoint_manager: CheckpointManager,
    task_enqueuer: ResumeTaskEnqueuer | None = None,
) -> None:
    """Configure confirmation route dependencies. Called during app startup.

    Args:
        checkpoint_manager: CheckpointManager instance (contains job_store internally).
        task_enqueuer: Task enqueuer (defaults to CeleryResumeEnqueuer).
    """
    global _checkpoint_manager, _task_enqueuer  # noqa: PLW0603
    _checkpoint_manager = checkpoint_manager
    _task_enqueuer = task_enqueuer or CeleryResumeEnqueuer()


def _get_checkpoint_manager() -> CheckpointManager:
    """Dependency to get the configured checkpoint manager."""
    if _checkpoint_manager is None:
        raise RuntimeError("Confirmation routes not configured. Call configure_confirmation_routes() first.")
    return _checkpoint_manager


def _get_task_enqueuer() -> ResumeTaskEnqueuer:
    """Dependency to get the configured task enqueuer."""
    if _task_enqueuer is None:
        raise RuntimeError("Confirmation routes not configured. Call configure_confirmation_routes() first.")
    return _task_enqueuer


@router.post(
    "/{job_id}/confirm/transcription",
    response_model=ConfirmResponse,
    responses={
        404: {"description": "Job not found"},
        409: {"description": "Job not awaiting confirmation or wrong checkpoint"},
        410: {"description": "Job expired"},
        422: {"description": "Validation error (invalid segment index)"},
    },
)
def confirm_transcription(
    job_id: str,
    body: TranscriptionConfirmRequest,
    checkpoint_manager: CheckpointManager = Depends(_get_checkpoint_manager),
    task_enqueuer: ResumeTaskEnqueuer = Depends(_get_task_enqueuer),
) -> ConfirmResponse:
    """Confirm the transcription checkpoint, optionally applying edits.

    Flow:
    1. Get job from store (404 if not found)
    2. Check if expired (410 if expired)
    3. Validate confirmation (catches 409 errors)
    4. Apply transcription edits if provided (catches 422 errors)
    5. Confirm and resume (returns next_step)
    6. Enqueue resume task
    7. Return ConfirmResponse
    """
    # Step 1: Get job (404 if not found)
    try:
        job = checkpoint_manager._job_store.get_job(job_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "JOB_NOT_FOUND",
                "message": f"Khong tim thay job voi ID: {job_id}",
            },
        )

    # Step 2: Check if expired (410)
    if job.status == JobStatus.EXPIRED:
        raise HTTPException(
            status_code=410,
            detail={
                "error": "JOB_EXPIRED",
                "message": f"Job '{job_id}' has expired",
            },
        )

    # Step 3: Validate confirmation (409 errors, 410 for expired fallback)
    try:
        checkpoint_manager.validate_confirmation(job_id, CheckpointType.TRANSCRIPTION)
    except NotAwaitingConfirmationError as exc:
        # Fallback: if job became expired between step 2 and step 3, return 410
        if exc.current_status == JobStatus.EXPIRED:
            raise HTTPException(
                status_code=410,
                detail={
                    "error": "JOB_EXPIRED",
                    "message": f"Job '{job_id}' has expired",
                },
            )
        raise HTTPException(
            status_code=409,
            detail={
                "error": "NOT_AWAITING_CONFIRMATION",
                "message": f"Job '{job_id}' is not awaiting confirmation (current status: {exc.current_status.value})",
                "current_status": exc.current_status.value,
            },
        )
    except WrongCheckpointError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "WRONG_CHECKPOINT",
                "message": f"Job '{job_id}' is at checkpoint '{exc.actual.value}', not '{exc.expected.value}'",
                "current_checkpoint": exc.actual.value,
                "expected_checkpoint": exc.expected.value,
            },
        )
    except ConfirmationInProgressError:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "CONFIRMATION_IN_PROGRESS",
                "message": f"Confirmation already in progress for job '{job_id}'",
            },
        )

    # Step 4: Apply transcription edits if provided (422 errors)
    if body.edits:
        try:
            checkpoint_manager.apply_transcription_edits(job_id, body.edits)
        except InvalidSegmentIndexError as exc:
            # Release the confirmation lock on failure
            checkpoint_manager._job_store.update_job(job_id, confirmation_lock=False)
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "INVALID_SEGMENT_INDEX",
                    "message": f"Segment index {exc.index} is out of bounds (valid range: 0 to {exc.segment_count - 1})",
                    "index": exc.index,
                    "segment_count": exc.segment_count,
                },
            )

    # Step 5: Confirm and resume (returns next_step)
    next_step = checkpoint_manager.confirm_and_resume(job_id)

    # Step 6: Enqueue resume task
    task_enqueuer.enqueue(job_id, next_step.value)

    logger.info(
        "Transcription confirmed for job %s, resuming at step '%s'",
        job_id,
        next_step.value,
    )

    # Step 7: Return ConfirmResponse
    return ConfirmResponse(
        job_id=job_id,
        status=JobStatus.PROCESSING.value,
        next_step=next_step.value,
        message="Transcription confirmed, pipeline resuming",
    )


@router.post(
    "/{job_id}/confirm/translation",
    response_model=ConfirmResponse,
    responses={
        404: {"description": "Job not found"},
        409: {"description": "Job not awaiting confirmation or wrong checkpoint"},
        410: {"description": "Job expired"},
        422: {"description": "Validation error (invalid segment index)"},
    },
)
def confirm_translation(
    job_id: str,
    body: TranslationConfirmRequest,
    checkpoint_manager: CheckpointManager = Depends(_get_checkpoint_manager),
    task_enqueuer: ResumeTaskEnqueuer = Depends(_get_task_enqueuer),
) -> ConfirmResponse:
    """Confirm the translation checkpoint, optionally applying edits.

    Flow:
    1. Get job from store (404 if not found)
    2. Check if expired (410 if expired)
    3. Validate confirmation (catches 409 errors)
    4. Apply translation edits if provided (catches 422 errors)
    5. Confirm and resume (returns next_step)
    6. Enqueue resume task
    7. Return ConfirmResponse
    """
    # Step 1: Get job (404 if not found)
    try:
        job = checkpoint_manager._job_store.get_job(job_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "JOB_NOT_FOUND",
                "message": f"Khong tim thay job voi ID: {job_id}",
            },
        )

    # Step 2: Check if expired (410)
    if job.status == JobStatus.EXPIRED:
        raise HTTPException(
            status_code=410,
            detail={
                "error": "JOB_EXPIRED",
                "message": f"Job '{job_id}' has expired",
            },
        )

    # Step 3: Validate confirmation (409 errors, 410 for expired fallback)
    try:
        checkpoint_manager.validate_confirmation(job_id, CheckpointType.TRANSLATION)
    except NotAwaitingConfirmationError as exc:
        # Fallback: if job became expired between step 2 and step 3, return 410
        if exc.current_status == JobStatus.EXPIRED:
            raise HTTPException(
                status_code=410,
                detail={
                    "error": "JOB_EXPIRED",
                    "message": f"Job '{job_id}' has expired",
                },
            )
        raise HTTPException(
            status_code=409,
            detail={
                "error": "NOT_AWAITING_CONFIRMATION",
                "message": f"Job '{job_id}' is not awaiting confirmation (current status: {exc.current_status.value})",
                "current_status": exc.current_status.value,
            },
        )
    except WrongCheckpointError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "WRONG_CHECKPOINT",
                "message": f"Job '{job_id}' is at checkpoint '{exc.actual.value}', not '{exc.expected.value}'",
                "current_checkpoint": exc.actual.value,
                "expected_checkpoint": exc.expected.value,
            },
        )
    except ConfirmationInProgressError:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "CONFIRMATION_IN_PROGRESS",
                "message": f"Confirmation already in progress for job '{job_id}'",
            },
        )

    # Step 4: Apply translation edits if provided (422 errors)
    if body.edits:
        try:
            checkpoint_manager.apply_translation_edits(job_id, body.edits)
        except InvalidSegmentIndexError as exc:
            # Release the confirmation lock on failure
            checkpoint_manager._job_store.update_job(job_id, confirmation_lock=False)
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "INVALID_SEGMENT_INDEX",
                    "message": f"Segment index {exc.index} is out of bounds (valid range: 0 to {exc.segment_count - 1})",
                    "index": exc.index,
                    "segment_count": exc.segment_count,
                },
            )

    # Step 5: Confirm and resume (returns next_step)
    next_step = checkpoint_manager.confirm_and_resume(job_id)

    # Step 6: Enqueue resume task
    task_enqueuer.enqueue(job_id, next_step.value)

    logger.info(
        "Translation confirmed for job %s, resuming at step '%s'",
        job_id,
        next_step.value,
    )

    # Step 7: Return ConfirmResponse
    return ConfirmResponse(
        job_id=job_id,
        status=JobStatus.PROCESSING.value,
        next_step=next_step.value,
        message="Translation confirmed, pipeline resuming",
    )


@router.post(
    "/{job_id}/confirm/voice",
    response_model=ConfirmResponse,
    responses={
        404: {"description": "Job not found"},
        409: {"description": "Job not awaiting confirmation or wrong checkpoint"},
        410: {"description": "Job expired"},
        422: {"description": "Voice unavailable"},
    },
)
def confirm_voice(
    job_id: str,
    body: VoiceConfirmRequest,
    checkpoint_manager: CheckpointManager = Depends(_get_checkpoint_manager),
    task_enqueuer: ResumeTaskEnqueuer = Depends(_get_task_enqueuer),
) -> ConfirmResponse:
    """Confirm the voice selection checkpoint.

    Flow:
    1. Get job from store (404 if not found)
    2. Check if expired (410 if expired)
    3. Validate confirmation (catches 409 errors)
    4. Validate voice_id against available options (422 if unavailable)
    5. Store selected voice_id in artifacts
    6. Confirm and resume (returns next_step)
    7. Enqueue resume task
    8. Return ConfirmResponse
    """
    # Step 1: Get job (404 if not found)
    try:
        job = checkpoint_manager._job_store.get_job(job_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "JOB_NOT_FOUND",
                "message": f"Khong tim thay job voi ID: {job_id}",
            },
        )

    # Step 2: Check if expired (410)
    if job.status == JobStatus.EXPIRED:
        raise HTTPException(
            status_code=410,
            detail={
                "error": "JOB_EXPIRED",
                "message": f"Job '{job_id}' has expired",
            },
        )

    # Step 3: Validate confirmation (409 errors, 410 for expired fallback)
    try:
        checkpoint_manager.validate_confirmation(job_id, CheckpointType.VOICE_SELECTION)
    except NotAwaitingConfirmationError as exc:
        # Fallback: if job became expired between step 2 and step 3, return 410
        if exc.current_status == JobStatus.EXPIRED:
            raise HTTPException(
                status_code=410,
                detail={
                    "error": "JOB_EXPIRED",
                    "message": f"Job '{job_id}' has expired",
                },
            )
        raise HTTPException(
            status_code=409,
            detail={
                "error": "NOT_AWAITING_CONFIRMATION",
                "message": f"Job '{job_id}' is not awaiting confirmation (current status: {exc.current_status.value})",
                "current_status": exc.current_status.value,
            },
        )
    except WrongCheckpointError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "WRONG_CHECKPOINT",
                "message": f"Job '{job_id}' is at checkpoint '{exc.actual.value}', not '{exc.expected.value}'",
                "current_checkpoint": exc.actual.value,
                "expected_checkpoint": exc.expected.value,
            },
        )
    except ConfirmationInProgressError:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "CONFIRMATION_IN_PROGRESS",
                "message": f"Confirmation already in progress for job '{job_id}'",
            },
        )

    # Step 4: Validate voice_id against available voice options (422 if unavailable)
    available_voices = [opt.voice_id for opt in (job.voice_options or [])]
    if body.voice_id not in available_voices:
        # Release the confirmation lock on failure
        checkpoint_manager._job_store.update_job(job_id, confirmation_lock=False)
        raise HTTPException(
            status_code=422,
            detail={
                "error": "VOICE_UNAVAILABLE",
                "message": f"Voice '{body.voice_id}' is not available",
                "available_voices": available_voices,
            },
        )

    # Step 5: Store selected voice_id in job artifacts
    existing_artifacts = dict(job.artifacts)
    existing_artifacts["selected_voice_id"] = body.voice_id
    checkpoint_manager._job_store.update_job(job_id, artifacts=existing_artifacts)

    # Step 6: Confirm and resume (returns next_step)
    next_step = checkpoint_manager.confirm_and_resume(job_id)

    # Step 7: Enqueue resume task
    task_enqueuer.enqueue(job_id, next_step.value)

    logger.info(
        "Voice '%s' confirmed for job %s, resuming at step '%s'",
        body.voice_id,
        job_id,
        next_step.value,
    )

    # Step 8: Return ConfirmResponse
    return ConfirmResponse(
        job_id=job_id,
        status=JobStatus.PROCESSING.value,
        next_step=next_step.value,
        message="Voice selection confirmed, pipeline resuming",
    )


@router.post(
    "/{job_id}/confirm/audio",
    response_model=ConfirmResponse,
    responses={
        404: {"description": "Job not found"},
        409: {"description": "Job not awaiting confirmation or wrong checkpoint"},
        410: {"description": "Job expired"},
    },
)
def confirm_audio(
    job_id: str,
    body: AudioConfirmRequest,
    checkpoint_manager: CheckpointManager = Depends(_get_checkpoint_manager),
    task_enqueuer: ResumeTaskEnqueuer = Depends(_get_task_enqueuer),
) -> ConfirmResponse:
    """Confirm the audio preview checkpoint, applying any text edits.

    Flow:
    1. Get job (404)
    2. Check expiry (410)
    3. Validate checkpoint is AUDIO_PREVIEW (409)
    4. Apply translation text edits if provided
    5. Set audio_preview_confirmed flag in artifacts
    6. Confirm and resume at SYNTHESIZING_VOICE (re-synthesize with edited texts)
    """
    try:
        job = checkpoint_manager._job_store.get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail={"error": "JOB_NOT_FOUND", "message": f"Khong tim thay job voi ID: {job_id}"})

    if job.status == JobStatus.EXPIRED:
        raise HTTPException(status_code=410, detail={"error": "JOB_EXPIRED", "message": f"Job '{job_id}' has expired"})

    try:
        checkpoint_manager.validate_confirmation(job_id, CheckpointType.AUDIO_PREVIEW)
    except NotAwaitingConfirmationError as exc:
        if exc.current_status == JobStatus.EXPIRED:
            raise HTTPException(status_code=410, detail={"error": "JOB_EXPIRED", "message": f"Job '{job_id}' has expired"})
        raise HTTPException(status_code=409, detail={"error": "NOT_AWAITING_CONFIRMATION", "message": f"Job is not awaiting confirmation", "current_status": exc.current_status.value})
    except WrongCheckpointError as exc:
        raise HTTPException(status_code=409, detail={"error": "WRONG_CHECKPOINT", "message": f"Job is at checkpoint '{exc.actual.value}', not 'audio_preview'", "current_checkpoint": exc.actual.value})
    except ConfirmationInProgressError:
        raise HTTPException(status_code=409, detail={"error": "CONFIRMATION_IN_PROGRESS", "message": f"Confirmation already in progress for job '{job_id}'"})

    # Apply text and/or voice edits
    if body.edits:
        try:
            checkpoint_manager.apply_audio_preview_edits(job_id, body.edits)
        except Exception as exc:
            checkpoint_manager._job_store.update_job(job_id, confirmation_lock=False)
            raise HTTPException(status_code=422, detail={"error": "EDIT_FAILED", "message": str(exc)})

    # Mark audio preview as confirmed — synthesis will not pause again
    existing_artifacts = dict(job.artifacts)
    existing_artifacts["audio_preview_confirmed"] = "true"
    checkpoint_manager._job_store.update_job(job_id, artifacts=existing_artifacts)

    # Resume at SYNTHESIZING_VOICE to re-synthesize with edited texts
    checkpoint_manager.confirm_and_resume(job_id)
    task_enqueuer.enqueue(job_id, PipelineStep.SYNTHESIZING_VOICE.value)

    logger.info("Audio preview confirmed for job %s, re-synthesizing", job_id)

    return ConfirmResponse(
        job_id=job_id,
        status=JobStatus.PROCESSING.value,
        next_step=PipelineStep.SYNTHESIZING_VOICE.value,
        message="Audio confirmed, re-synthesizing and composing video",
    )


@router.get("/{job_id}/preview/voice/{voice_id}")
def get_voice_preview(
    job_id: str,
    voice_id: str,
    checkpoint_manager: CheckpointManager = Depends(_get_checkpoint_manager),
) -> "FileResponse":
    """Serve a voice preview audio file for a job at the voice_selection checkpoint.

    Validates:
    - Job exists (404 if not found)
    - Job is at voice_selection checkpoint (409 if not)
    - Preview audio file exists on disk (404 if not found)

    Returns the MP3 audio file with appropriate content-type header.
    """
    from pathlib import Path

    from fastapi.responses import FileResponse

    # 1. Get job (404 if not found)
    try:
        job = checkpoint_manager._job_store.get_job(job_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "JOB_NOT_FOUND",
                "message": f"Khong tim thay job voi ID: {job_id}",
            },
        )

    # 2. Verify job is at VOICE_SELECTION checkpoint (409 if not)
    if (
        job.status != JobStatus.AWAITING_CONFIRMATION
        or job.checkpoint_type != CheckpointType.VOICE_SELECTION
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "NOT_AT_VOICE_SELECTION",
                "message": "Job khong o trang thai cho xac nhan chon giong noi",
            },
        )

    # 3. Locate the preview file at {work_dir}/voice_previews/{voice_id}_preview.mp3
    preview_path = Path(job.work_dir) / "voice_previews" / f"{voice_id}_preview.mp3"

    # 4. Verify file exists (404 if not)
    if not preview_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "PREVIEW_NOT_FOUND",
                "message": f"Khong tim thay file preview cho giong: {voice_id}",
            },
        )

    # 5. Return FileResponse with media_type="audio/mpeg"
    return FileResponse(
        path=str(preview_path),
        media_type="audio/mpeg",
        filename=f"{voice_id}_preview.mp3",
    )
