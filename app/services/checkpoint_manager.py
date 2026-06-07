"""Checkpoint manager for pipeline pause/resume and confirmation logic.

Encapsulates checkpoint state transitions, validation, and optimistic locking
for the confirmation workflow.
"""

import json
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.models.confirmation_schemas import SegmentEdit, TranslationEdit
from app.models.job import CheckpointType, JobState, JobStatus, PipelineStep
from app.services.pipeline import JobStoreProtocol

logger = logging.getLogger(__name__)

# Expiration duration for awaiting_confirmation jobs
CHECKPOINT_EXPIRATION_HOURS = 24


# Mapping from checkpoint type to the next pipeline step after confirmation
CHECKPOINT_NEXT_STEP: dict[CheckpointType, PipelineStep] = {
    CheckpointType.TRANSCRIPTION: PipelineStep.TRANSLATING,
    CheckpointType.TRANSLATION: PipelineStep.SYNTHESIZING_VOICE,
    CheckpointType.VOICE_SELECTION: PipelineStep.COMPOSING_VIDEO,
}


# --- Custom Exceptions ---


class CheckpointError(Exception):
    """Base exception for checkpoint-related errors."""

    def __init__(self, message: str, job_id: str | None = None):
        super().__init__(message)
        self.job_id = job_id


class NotAwaitingConfirmationError(CheckpointError):
    """Raised when a confirmation is attempted on a job not in awaiting_confirmation status."""

    def __init__(self, job_id: str, current_status: JobStatus):
        self.current_status = current_status
        super().__init__(
            f"Job {job_id} is not awaiting confirmation (current status: {current_status.value})",
            job_id=job_id,
        )


class WrongCheckpointError(CheckpointError):
    """Raised when a confirmation targets a checkpoint that doesn't match the job's current checkpoint."""

    def __init__(
        self, job_id: str, expected: CheckpointType, actual: CheckpointType
    ):
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Job {job_id} is at checkpoint '{actual.value}', not '{expected.value}'",
            job_id=job_id,
        )


class ConfirmationInProgressError(CheckpointError):
    """Raised when another confirmation for the same job is already being processed."""

    def __init__(self, job_id: str):
        super().__init__(
            f"Confirmation already in progress for job {job_id}",
            job_id=job_id,
        )


class InvalidSegmentIndexError(CheckpointError):
    """Raised when an edit references a segment index that is out of bounds."""

    def __init__(self, job_id: str, index: int, segment_count: int):
        self.index = index
        self.segment_count = segment_count
        super().__init__(
            f"Segment index {index} is out of bounds for job {job_id} "
            f"(valid range: 0 to {segment_count - 1})",
            job_id=job_id,
        )


# --- CheckpointManager ---


class CheckpointManager:
    """Manages checkpoint pause/resume lifecycle for pipeline jobs.

    Handles:
    - Pausing the pipeline at a checkpoint (status → awaiting_confirmation)
    - Validating confirmation requests (correct status, checkpoint match, locking)
    - Resuming the pipeline after confirmation (status → processing, return next step)
    """

    def __init__(self, job_store: JobStoreProtocol) -> None:
        self._job_store = job_store

    def pause_at_checkpoint(
        self, job_id: str, checkpoint_type: CheckpointType
    ) -> None:
        """Transition job to awaiting_confirmation at the given checkpoint.

        Sets the job status to AWAITING_CONFIRMATION, stores the checkpoint type,
        and records the time the checkpoint was entered.

        Args:
            job_id: Unique job identifier.
            checkpoint_type: The type of checkpoint being paused at.
        """
        now = datetime.now(timezone.utc)
        self._job_store.update_job(
            job_id,
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=checkpoint_type,
            checkpoint_entered_at=now,
        )
        logger.info(
            "Job %s paused at checkpoint '%s'", job_id, checkpoint_type.value
        )

    def validate_confirmation(
        self, job_id: str, expected_checkpoint: CheckpointType
    ) -> JobState:
        """Validate job is at expected checkpoint and acquire confirmation lock.

        Checks that:
        1. The job is in AWAITING_CONFIRMATION status
        2. The job's checkpoint type matches the expected checkpoint
        3. No other confirmation is already in progress (confirmation_lock is False)

        If all checks pass, acquires the confirmation lock.

        Args:
            job_id: Unique job identifier.
            expected_checkpoint: The checkpoint type the caller expects the job to be at.

        Returns:
            The validated JobState with confirmation_lock acquired.

        Raises:
            NotAwaitingConfirmationError: If job is not in awaiting_confirmation status.
            WrongCheckpointError: If checkpoint type doesn't match.
            ConfirmationInProgressError: If another confirmation is already in progress.
        """
        job = self._job_store.get_job(job_id)

        # Check 1: Job must be in AWAITING_CONFIRMATION status
        if job.status != JobStatus.AWAITING_CONFIRMATION:
            raise NotAwaitingConfirmationError(
                job_id=job_id, current_status=job.status
            )

        # Check 2: Checkpoint type must match
        if job.checkpoint_type != expected_checkpoint:
            raise WrongCheckpointError(
                job_id=job_id,
                expected=expected_checkpoint,
                actual=job.checkpoint_type,  # type: ignore[arg-type]
            )

        # Check 3: No concurrent confirmation in progress
        if job.confirmation_lock:
            raise ConfirmationInProgressError(job_id=job_id)

        # Acquire confirmation lock
        self._job_store.update_job(job_id, confirmation_lock=True)
        logger.info(
            "Confirmation lock acquired for job %s at checkpoint '%s'",
            job_id,
            expected_checkpoint.value,
        )

        # Re-fetch to return the updated state
        return self._job_store.get_job(job_id)

    def confirm_and_resume(self, job_id: str) -> PipelineStep:
        """Release lock, set status to processing, and return the next pipeline step.

        Args:
            job_id: Unique job identifier.

        Returns:
            The next PipelineStep to execute after this checkpoint.
        """
        job = self._job_store.get_job(job_id)
        checkpoint_type = job.checkpoint_type

        # Determine next step from checkpoint type
        next_step = CHECKPOINT_NEXT_STEP[checkpoint_type]  # type: ignore[index]

        # Release lock, set status to processing, clear checkpoint fields
        self._job_store.update_job(
            job_id,
            status=JobStatus.PROCESSING,
            confirmation_lock=False,
            checkpoint_type=None,
            checkpoint_entered_at=None,
        )
        logger.info(
            "Job %s confirmed and resuming at step '%s'",
            job_id,
            next_step.value,
        )

        return next_step

    def apply_transcription_edits(
        self, job_id: str, edits: list[SegmentEdit]
    ) -> None:
        """Apply user edits to transcription artifact segments.

        Loads the transcription JSON from the job's artifacts, applies each edit
        by segment index, filters out whitespace-only segments, and saves back.

        Args:
            job_id: Unique job identifier.
            edits: List of SegmentEdit objects with index and replacement text.

        Raises:
            InvalidSegmentIndexError: If any edit references an out-of-bounds index.
        """
        job = self._job_store.get_job(job_id)
        transcription_path = Path(job.artifacts["transcription_path"])

        # Load transcription JSON
        with open(transcription_path, "r", encoding="utf-8") as f:
            transcription_data = json.load(f)

        segments = transcription_data["segments"]
        segment_count = len(segments)

        # Validate all indices first before applying any edits
        for edit in edits:
            if edit.index < 0 or edit.index >= segment_count:
                raise InvalidSegmentIndexError(
                    job_id=job_id,
                    index=edit.index,
                    segment_count=segment_count,
                )

        # Apply edits by index
        for edit in edits:
            segments[edit.index]["text"] = edit.text

        # Filter out segments where text is whitespace-only
        segments = [seg for seg in segments if seg["text"].strip()]
        transcription_data["segments"] = segments

        # Save back to the same path
        with open(transcription_path, "w", encoding="utf-8") as f:
            json.dump(transcription_data, f, ensure_ascii=False, indent=2)

        logger.info(
            "Applied %d transcription edits for job %s (%d segments remaining)",
            len(edits),
            job_id,
            len(segments),
        )

    def apply_translation_edits(
        self, job_id: str, edits: list[TranslationEdit]
    ) -> None:
        """Apply user edits to translation artifact segments.

        Loads the translation JSON from the job's artifacts, applies each edit
        by segment index, filters out whitespace-only segments, and saves back.

        Args:
            job_id: Unique job identifier.
            edits: List of TranslationEdit objects with index and replacement translated_text.

        Raises:
            InvalidSegmentIndexError: If any edit references an out-of-bounds index.
        """
        job = self._job_store.get_job(job_id)
        translation_path = Path(job.artifacts["translation_path"])

        # Load translation JSON
        with open(translation_path, "r", encoding="utf-8") as f:
            translation_data = json.load(f)

        segments = translation_data["segments"]
        segment_count = len(segments)

        # Validate all indices first before applying any edits
        for edit in edits:
            if edit.index < 0 or edit.index >= segment_count:
                raise InvalidSegmentIndexError(
                    job_id=job_id,
                    index=edit.index,
                    segment_count=segment_count,
                )

        # Apply edits by index
        for edit in edits:
            segments[edit.index]["translated_text"] = edit.translated_text

        # Filter out segments where translated_text is whitespace-only
        segments = [seg for seg in segments if seg["translated_text"].strip()]
        translation_data["segments"] = segments

        # Save back to the same path
        with open(translation_path, "w", encoding="utf-8") as f:
            json.dump(translation_data, f, ensure_ascii=False, indent=2)

        logger.info(
            "Applied %d translation edits for job %s (%d segments remaining)",
            len(edits),
            job_id,
            len(segments),
        )

    def check_expired_jobs(self) -> list[str]:
        """Find and expire jobs that have been awaiting confirmation > 24h.

        Scans all jobs with AWAITING_CONFIRMATION status, checks if their
        checkpoint_entered_at timestamp is older than 24 hours from now.
        For expired jobs: transitions status to EXPIRED, deletes the working
        directory, and removes the job from the store.

        Returns:
            List of job IDs that were expired.
        """
        now = datetime.now(timezone.utc)
        expiration_threshold = now - timedelta(hours=CHECKPOINT_EXPIRATION_HOURS)
        expired_ids: list[str] = []

        awaiting_job_ids = self._job_store.list_awaiting_confirmation_job_ids()

        for job_id in awaiting_job_ids:
            try:
                job = self._job_store.get_job(job_id)
            except KeyError:
                # Job was deleted between scan and get — skip
                continue

            # Skip jobs that don't have checkpoint_entered_at set
            if job.checkpoint_entered_at is None:
                continue

            if job.checkpoint_entered_at < expiration_threshold:
                # Expire the job
                self._job_store.update_job(job_id, status=JobStatus.EXPIRED)

                # Delete working directory
                work_dir = Path(job.work_dir)
                if work_dir.exists():
                    try:
                        shutil.rmtree(work_dir)
                        logger.info(
                            "Deleted working directory for expired job %s: %s",
                            job_id,
                            work_dir,
                        )
                    except OSError as exc:
                        logger.warning(
                            "Failed to delete working directory for job %s: %s",
                            job_id,
                            exc,
                        )

                # Remove from store
                self._job_store.delete_job(job_id)

                expired_ids.append(job_id)
                logger.info("Expired job %s (awaiting since %s)", job_id, job.checkpoint_entered_at)

        return expired_ids

    def reset_expiration(self, job_id: str) -> None:
        """Reset the 24h expiration timer for a job at a checkpoint.

        Called when the user queries the status of a job that is in
        AWAITING_CONFIRMATION status. Resets checkpoint_entered_at to now,
        giving the user another 24 hours to confirm.

        Args:
            job_id: Unique job identifier.
        """
        now = datetime.now(timezone.utc)
        self._job_store.update_job(job_id, checkpoint_entered_at=now)
        logger.debug("Reset expiration timer for job %s", job_id)
