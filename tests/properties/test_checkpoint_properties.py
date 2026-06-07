"""Property-based tests for CheckpointManager pause behavior.

Feature: pipeline-preview-confirm

Tests cover:
- Property 1: Checkpoint pause preserves results
- Property 2: No progression while awaiting confirmation
"""

from datetime import datetime, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.models.job import CheckpointType, JobState, JobStatus, PipelineStep
from app.services.checkpoint_manager import (
    CheckpointManager,
    NotAwaitingConfirmationError,
)


# --- Test Helpers ---


class FakeJobStore:
    """In-memory job store for property testing CheckpointManager."""

    def __init__(self, jobs: dict[str, JobState] | None = None):
        self._jobs: dict[str, JobState] = jobs or {}
        self.updates: list[dict] = []

    def add_job(self, job: JobState) -> None:
        self._jobs[job.job_id] = job

    def get_job(self, job_id: str) -> JobState:
        return self._jobs[job_id]

    def update_job(self, job_id: str, **kwargs: object) -> None:
        self.updates.append({"job_id": job_id, **kwargs})
        job = self._jobs[job_id]
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)

    def delete_job(self, job_id: str) -> None:
        del self._jobs[job_id]

    def list_awaiting_confirmation_job_ids(self) -> list[str]:
        return [
            jid
            for jid, job in self._jobs.items()
            if job.status == JobStatus.AWAITING_CONFIRMATION
        ]


def _make_job_state(
    job_id: str,
    status: JobStatus = JobStatus.PROCESSING,
    checkpoint_type: CheckpointType | None = None,
    checkpoint_entered_at: datetime | None = None,
    confirmation_lock: bool = False,
) -> JobState:
    """Create a JobState for property testing."""
    now = datetime.now(timezone.utc)
    return JobState(
        job_id=job_id,
        url="https://www.douyin.com/video/123",
        status=status,
        created_at=now,
        updated_at=now,
        work_dir=f"/tmp/jobs/{job_id}",
        checkpoint_type=checkpoint_type,
        checkpoint_entered_at=checkpoint_entered_at,
        confirmation_lock=confirmation_lock,
    )


# --- Strategies ---

# Generate valid job_id strings (non-empty alphanumeric with hyphens)
job_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="-_"),
    min_size=1,
    max_size=50,
)

checkpoint_type_strategy = st.sampled_from(CheckpointType)


# --- Property 1: Checkpoint pause preserves results ---


# Feature: pipeline-preview-confirm, Property 1: Checkpoint pause preserves results
@pytest.mark.property
@settings(max_examples=100)
@given(
    job_id=job_id_strategy,
    checkpoint_type=checkpoint_type_strategy,
)
def test_checkpoint_pause_transitions_to_awaiting_confirmation(
    job_id: str,
    checkpoint_type: CheckpointType,
):
    """For any pipeline job and any checkpoint step, when that step completes,
    the job status SHALL transition to "awaiting_confirmation" with the correct
    checkpoint_type, and checkpoint_entered_at SHALL be set.

    **Validates: Requirements 1.1, 1.2, 1.3**
    """
    # Arrange: create a job in PROCESSING status
    job = _make_job_state(job_id=job_id, status=JobStatus.PROCESSING)
    store = FakeJobStore()
    store.add_job(job)
    manager = CheckpointManager(store)

    # Act: pause at the given checkpoint
    manager.pause_at_checkpoint(job_id, checkpoint_type)

    # Assert: status transitioned to AWAITING_CONFIRMATION
    updated_job = store.get_job(job_id)
    assert updated_job.status == JobStatus.AWAITING_CONFIRMATION, (
        f"Expected status AWAITING_CONFIRMATION, got {updated_job.status}"
    )

    # Assert: checkpoint_type matches what was requested
    assert updated_job.checkpoint_type == checkpoint_type, (
        f"Expected checkpoint_type {checkpoint_type}, got {updated_job.checkpoint_type}"
    )

    # Assert: checkpoint_entered_at is set to a valid datetime
    assert updated_job.checkpoint_entered_at is not None, (
        "checkpoint_entered_at should be set after pause_at_checkpoint"
    )
    assert isinstance(updated_job.checkpoint_entered_at, datetime), (
        "checkpoint_entered_at should be a datetime instance"
    )


# Feature: pipeline-preview-confirm, Property 1: Checkpoint pause preserves results
@pytest.mark.property
@settings(max_examples=100)
@given(
    job_id=job_id_strategy,
    checkpoint_type=checkpoint_type_strategy,
)
def test_checkpoint_pause_records_update_in_store(
    job_id: str,
    checkpoint_type: CheckpointType,
):
    """For any pause_at_checkpoint call, the job store update SHALL contain
    the correct status, checkpoint_type, and a checkpoint_entered_at timestamp.

    **Validates: Requirements 1.1, 1.2, 1.3**
    """
    # Arrange
    job = _make_job_state(job_id=job_id, status=JobStatus.PROCESSING)
    store = FakeJobStore()
    store.add_job(job)
    manager = CheckpointManager(store)

    # Act
    manager.pause_at_checkpoint(job_id, checkpoint_type)

    # Assert: exactly one update was made
    assert len(store.updates) == 1, (
        f"Expected 1 update call, got {len(store.updates)}"
    )

    update = store.updates[0]
    assert update["job_id"] == job_id
    assert update["status"] == JobStatus.AWAITING_CONFIRMATION
    assert update["checkpoint_type"] == checkpoint_type
    assert "checkpoint_entered_at" in update
    assert update["checkpoint_entered_at"] is not None


# --- Property 2: No progression while awaiting confirmation ---


# Feature: pipeline-preview-confirm, Property 2: No progression while awaiting confirmation
@pytest.mark.property
@settings(max_examples=100)
@given(
    job_id=job_id_strategy,
    checkpoint_type=checkpoint_type_strategy,
)
def test_no_progression_while_awaiting_confirmation(
    job_id: str,
    checkpoint_type: CheckpointType,
):
    """For any job in "awaiting_confirmation" status, attempting to validate a
    confirmation for a WRONG checkpoint type SHALL raise WrongCheckpointError,
    leaving job artifacts and status unchanged.

    **Validates: Requirements 1.4, 1.5**
    """
    from app.services.checkpoint_manager import WrongCheckpointError

    # Arrange: create a job already at a checkpoint
    job = _make_job_state(
        job_id=job_id,
        status=JobStatus.AWAITING_CONFIRMATION,
        checkpoint_type=checkpoint_type,
        checkpoint_entered_at=datetime.now(timezone.utc),
    )
    store = FakeJobStore()
    store.add_job(job)
    manager = CheckpointManager(store)

    # Determine a different checkpoint type to attempt
    other_checkpoints = [ct for ct in CheckpointType if ct != checkpoint_type]
    if not other_checkpoints:
        return  # Only one checkpoint type (shouldn't happen with current enum)

    wrong_checkpoint = other_checkpoints[0]

    # Record state before attempt
    status_before = job.status
    checkpoint_type_before = job.checkpoint_type
    entered_at_before = job.checkpoint_entered_at

    # Act: attempt to validate with wrong checkpoint type
    with pytest.raises(WrongCheckpointError):
        manager.validate_confirmation(job_id, wrong_checkpoint)

    # Assert: job status and checkpoint fields are unchanged
    assert job.status == status_before, (
        "Job status should not change when wrong checkpoint confirmation is attempted"
    )
    assert job.checkpoint_type == checkpoint_type_before, (
        "checkpoint_type should not change on failed confirmation attempt"
    )
    assert job.checkpoint_entered_at == entered_at_before, (
        "checkpoint_entered_at should not change on failed confirmation attempt"
    )


# Feature: pipeline-preview-confirm, Property 2: No progression while awaiting confirmation
@pytest.mark.property
@settings(max_examples=100)
@given(
    job_id=job_id_strategy,
    checkpoint_type=checkpoint_type_strategy,
)
def test_locked_job_prevents_progression(
    job_id: str,
    checkpoint_type: CheckpointType,
):
    """For any job in "awaiting_confirmation" status with confirmation_lock=True,
    attempting to validate a confirmation SHALL raise ConfirmationInProgressError,
    and the job state SHALL remain unchanged.

    **Validates: Requirements 1.4, 1.5**
    """
    from app.services.checkpoint_manager import ConfirmationInProgressError

    # Arrange: create a locked job at a checkpoint
    job = _make_job_state(
        job_id=job_id,
        status=JobStatus.AWAITING_CONFIRMATION,
        checkpoint_type=checkpoint_type,
        checkpoint_entered_at=datetime.now(timezone.utc),
        confirmation_lock=True,
    )
    store = FakeJobStore()
    store.add_job(job)
    manager = CheckpointManager(store)

    # Record state before attempt
    status_before = job.status
    checkpoint_type_before = job.checkpoint_type
    lock_before = job.confirmation_lock

    # Act: attempt to validate (should be blocked by lock)
    with pytest.raises(ConfirmationInProgressError):
        manager.validate_confirmation(job_id, checkpoint_type)

    # Assert: job state is completely unchanged
    assert job.status == status_before, (
        "Job status should not change when confirmation is locked"
    )
    assert job.checkpoint_type == checkpoint_type_before, (
        "checkpoint_type should not change when confirmation is locked"
    )
    assert job.confirmation_lock == lock_before, (
        "confirmation_lock should remain True when blocked"
    )


# Feature: pipeline-preview-confirm, Property 2: No progression while awaiting confirmation
@pytest.mark.property
@settings(max_examples=100)
@given(
    job_id=job_id_strategy,
    checkpoint_type=checkpoint_type_strategy,
    non_awaiting_status=st.sampled_from([
        JobStatus.QUEUED,
        JobStatus.PROCESSING,
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.EXPIRED,
    ]),
)
def test_non_awaiting_job_rejects_confirmation(
    job_id: str,
    checkpoint_type: CheckpointType,
    non_awaiting_status: JobStatus,
):
    """For any job NOT in "awaiting_confirmation" status, attempting to
    validate a confirmation SHALL raise NotAwaitingConfirmationError,
    and job state SHALL remain unchanged.

    **Validates: Requirements 1.4, 1.5**
    """
    # Arrange: create a job in a non-awaiting status
    job = _make_job_state(
        job_id=job_id,
        status=non_awaiting_status,
    )
    store = FakeJobStore()
    store.add_job(job)
    manager = CheckpointManager(store)

    # Record state before attempt
    status_before = job.status

    # Act: attempt to validate confirmation
    with pytest.raises(NotAwaitingConfirmationError) as exc_info:
        manager.validate_confirmation(job_id, checkpoint_type)

    # Assert: error reports the correct current status
    assert exc_info.value.current_status == non_awaiting_status, (
        f"Error should report status {non_awaiting_status}, got {exc_info.value.current_status}"
    )

    # Assert: job status is unchanged
    assert job.status == status_before, (
        "Job status should not change when confirmation is rejected"
    )
