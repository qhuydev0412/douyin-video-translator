"""Property-based tests for confirmation API behavior.

Feature: pipeline-preview-confirm

Tests cover:
- Property 11: Confirmation for wrong job status returns 409
- Property 12: Confirmation for wrong checkpoint type returns 409
- Property 13: Valid confirmation returns updated status and enqueues resumption
"""

import json
import os
import tempfile
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from app.api.confirmation_routes import configure_confirmation_routes, router
from app.models.job import CheckpointType, JobState, JobStatus, PipelineStep, VoiceOption
from app.services.checkpoint_manager import CheckpointManager


# --- Test Helpers ---


class FakeJobStore:
    """In-memory fake job store for property testing confirmation API."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}

    def add_job(self, job: JobState) -> None:
        self._jobs[job.job_id] = job

    def create_job(self, job_id: str, url: str, client_ip: str, work_dir: str) -> JobState:
        now = datetime.now(timezone.utc)
        job = JobState(
            job_id=job_id,
            url=url,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            work_dir=work_dir,
        )
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> JobState:
        if job_id not in self._jobs:
            raise KeyError(f"Job not found: {job_id}")
        return self._jobs[job_id]

    def update_job(self, job_id: str, **kwargs: object) -> None:
        job = self.get_job(job_id)
        for field, value in kwargs.items():
            setattr(job, field, value)

    def delete_job(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    def list_awaiting_confirmation_job_ids(self) -> list[str]:
        return [
            jid
            for jid, job in self._jobs.items()
            if job.status == JobStatus.AWAITING_CONFIRMATION
        ]


class FakeResumeTaskEnqueuer:
    """Fake resume task enqueuer that records enqueued tasks."""

    def __init__(self) -> None:
        self.enqueued: list[tuple[str, str]] = []

    def enqueue(self, job_id: str, next_step: str) -> str:
        self.enqueued.append((job_id, next_step))
        return f"resume-task-{job_id}"


# Mapping from checkpoint type to the expected next pipeline step
CHECKPOINT_NEXT_STEP: dict[CheckpointType, PipelineStep] = {
    CheckpointType.TRANSCRIPTION: PipelineStep.TRANSLATING,
    CheckpointType.TRANSLATION: PipelineStep.SYNTHESIZING_VOICE,
    CheckpointType.VOICE_SELECTION: PipelineStep.SYNTHESIZING_VOICE,
}

# Mapping from checkpoint type to confirmation endpoint path suffix
CHECKPOINT_ENDPOINT: dict[CheckpointType, str] = {
    CheckpointType.TRANSCRIPTION: "transcription",
    CheckpointType.TRANSLATION: "translation",
    CheckpointType.VOICE_SELECTION: "voice",
}


def _build_test_client(
    fake_store: FakeJobStore, fake_enqueuer: FakeResumeTaskEnqueuer
) -> TestClient:
    """Create a FastAPI TestClient with configured confirmation routes."""
    app = FastAPI()
    checkpoint_manager = CheckpointManager(fake_store)
    configure_confirmation_routes(
        checkpoint_manager=checkpoint_manager,
        task_enqueuer=fake_enqueuer,
    )
    app.include_router(router)
    return TestClient(app)


def _make_job_state(
    job_id: str,
    status: JobStatus,
    checkpoint_type: CheckpointType | None = None,
    work_dir: str = "/tmp/test",
    artifacts: dict[str, str] | None = None,
    voice_options: list[VoiceOption] | None = None,
) -> JobState:
    """Create a JobState for property testing."""
    now = datetime.now(timezone.utc)
    return JobState(
        job_id=job_id,
        url="https://www.douyin.com/video/123",
        status=status,
        created_at=now,
        updated_at=now,
        work_dir=work_dir,
        checkpoint_type=checkpoint_type,
        checkpoint_entered_at=now if status == JobStatus.AWAITING_CONFIRMATION else None,
        confirmation_lock=False,
        artifacts=artifacts or {},
        voice_options=voice_options,
    )


def _create_transcription_artifact(tmp_dir: str) -> str:
    """Create a transcription JSON artifact and return the path."""
    path = os.path.join(tmp_dir, "transcription.json")
    data = {
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "你好世界", "confidence": 0.95},
            {"start": 1.5, "end": 3.0, "text": "测试段落", "confidence": 0.88},
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return path


def _create_translation_artifact(tmp_dir: str) -> str:
    """Create a translation JSON artifact and return the path."""
    path = os.path.join(tmp_dir, "translation.json")
    data = {
        "segments": [
            {"start": 0.0, "end": 1.5, "original_text": "你好世界", "translated_text": "Xin chào thế giới"},
            {"start": 1.5, "end": 3.0, "original_text": "测试段落", "translated_text": "Đoạn kiểm tra"},
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return path


# --- Strategies ---

# Non-awaiting statuses (all statuses except AWAITING_CONFIRMATION)
non_awaiting_status_strategy = st.sampled_from([
    JobStatus.QUEUED,
    JobStatus.PROCESSING,
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
])

# All confirmation endpoints
confirmation_endpoint_strategy = st.sampled_from(list(CheckpointType))

# Checkpoint types
checkpoint_type_strategy = st.sampled_from(list(CheckpointType))


# --- Property 11: Confirmation for wrong job status returns 409 ---


# Feature: pipeline-preview-confirm, Property 11: Confirmation for wrong job status returns 409
@pytest.mark.property
@settings(max_examples=100)
@given(
    non_awaiting_status=non_awaiting_status_strategy,
    target_endpoint=confirmation_endpoint_strategy,
)
def test_confirmation_for_wrong_job_status_returns_409(
    non_awaiting_status: JobStatus,
    target_endpoint: CheckpointType,
):
    """For any job in a status other than "awaiting_confirmation", submitting
    any confirmation request SHALL return HTTP 409.

    **Validates: Requirements 5.4**
    """
    # Arrange: create a job in a non-awaiting status
    fake_store = FakeJobStore()
    fake_enqueuer = FakeResumeTaskEnqueuer()

    job = _make_job_state(
        job_id="test-job",
        status=non_awaiting_status,
    )
    fake_store.add_job(job)

    client = _build_test_client(fake_store, fake_enqueuer)

    # Build the request body based on endpoint type
    endpoint_path = CHECKPOINT_ENDPOINT[target_endpoint]
    if target_endpoint == CheckpointType.VOICE_SELECTION:
        body = {"voice_id": "voice-1"}
    else:
        body = {}

    # Act: submit confirmation request
    response = client.post(
        f"/api/v1/jobs/test-job/confirm/{endpoint_path}",
        json=body,
    )

    # Assert: HTTP 409
    assert response.status_code == 409, (
        f"Expected 409 for job status '{non_awaiting_status.value}' "
        f"at endpoint '{endpoint_path}', got {response.status_code}"
    )

    # Assert: error indicates not awaiting confirmation
    detail = response.json()["detail"]
    assert detail["error"] == "NOT_AWAITING_CONFIRMATION", (
        f"Expected error code 'NOT_AWAITING_CONFIRMATION', got '{detail['error']}'"
    )

    # Assert: no resume task was enqueued
    assert len(fake_enqueuer.enqueued) == 0, (
        "No resume task should be enqueued for a rejected confirmation"
    )


# --- Property 12: Confirmation for wrong checkpoint type returns 409 ---


# Feature: pipeline-preview-confirm, Property 12: Confirmation for wrong checkpoint type returns 409
@pytest.mark.property
@settings(max_examples=100)
@given(
    actual_checkpoint=checkpoint_type_strategy,
    data=st.data(),
)
def test_confirmation_for_wrong_checkpoint_type_returns_409(
    actual_checkpoint: CheckpointType,
    data: st.DataObject,
):
    """For any job at checkpoint X, submitting a confirmation targeting
    checkpoint Y (where Y ≠ X) SHALL return HTTP 409 with WRONG_CHECKPOINT error.

    **Validates: Requirements 5.7**
    """
    # Select a different endpoint than the actual checkpoint
    other_endpoints = [ct for ct in CheckpointType if ct != actual_checkpoint]
    target_endpoint = data.draw(st.sampled_from(other_endpoints))

    # Arrange: create a job at the actual checkpoint with required artifacts
    fake_store = FakeJobStore()
    fake_enqueuer = FakeResumeTaskEnqueuer()

    tmp_dir = tempfile.mkdtemp()

    # Set up artifacts depending on actual checkpoint type
    artifacts: dict[str, str] = {}
    voice_options: list[VoiceOption] | None = None

    if actual_checkpoint == CheckpointType.TRANSCRIPTION:
        artifacts["transcription_path"] = _create_transcription_artifact(tmp_dir)
    elif actual_checkpoint == CheckpointType.TRANSLATION:
        artifacts["translation_path"] = _create_translation_artifact(tmp_dir)
    elif actual_checkpoint == CheckpointType.VOICE_SELECTION:
        voice_options = [
            VoiceOption(voice_id="voice-1", voice_name="Voice 1", preview_url="/preview/1"),
            VoiceOption(voice_id="voice-2", voice_name="Voice 2", preview_url="/preview/2"),
        ]

    job = _make_job_state(
        job_id="test-job",
        status=JobStatus.AWAITING_CONFIRMATION,
        checkpoint_type=actual_checkpoint,
        work_dir=tmp_dir,
        artifacts=artifacts,
        voice_options=voice_options,
    )
    fake_store.add_job(job)

    client = _build_test_client(fake_store, fake_enqueuer)

    # Build the request body for the WRONG endpoint
    endpoint_path = CHECKPOINT_ENDPOINT[target_endpoint]
    if target_endpoint == CheckpointType.VOICE_SELECTION:
        body = {"voice_id": "voice-1"}
    else:
        body = {}

    # Act: submit confirmation to wrong endpoint
    response = client.post(
        f"/api/v1/jobs/test-job/confirm/{endpoint_path}",
        json=body,
    )

    # Assert: HTTP 409
    assert response.status_code == 409, (
        f"Expected 409 for job at checkpoint '{actual_checkpoint.value}' "
        f"with confirmation to '{endpoint_path}', got {response.status_code}"
    )

    # Assert: error indicates wrong checkpoint
    detail = response.json()["detail"]
    assert detail["error"] == "WRONG_CHECKPOINT", (
        f"Expected error code 'WRONG_CHECKPOINT', got '{detail['error']}'"
    )

    # Assert: no resume task was enqueued
    assert len(fake_enqueuer.enqueued) == 0, (
        "No resume task should be enqueued for a wrong checkpoint confirmation"
    )


# --- Property 13: Valid confirmation returns updated status and enqueues resumption ---


# Feature: pipeline-preview-confirm, Property 13: Valid confirmation returns updated status and enqueues resumption
@pytest.mark.property
@settings(max_examples=100)
@given(
    checkpoint_type=checkpoint_type_strategy,
)
def test_valid_confirmation_returns_processing_status_and_enqueues_resume(
    checkpoint_type: CheckpointType,
):
    """For any valid confirmation request, the API response SHALL include
    status "processing" and correct next_step, and a resume task SHALL be enqueued.

    **Validates: Requirements 5.6**
    """
    # Arrange: create a job at the matching checkpoint with proper artifacts
    fake_store = FakeJobStore()
    fake_enqueuer = FakeResumeTaskEnqueuer()

    tmp_dir = tempfile.mkdtemp()

    artifacts: dict[str, str] = {}
    voice_options: list[VoiceOption] | None = None

    if checkpoint_type == CheckpointType.TRANSCRIPTION:
        artifacts["transcription_path"] = _create_transcription_artifact(tmp_dir)
    elif checkpoint_type == CheckpointType.TRANSLATION:
        artifacts["translation_path"] = _create_translation_artifact(tmp_dir)
    elif checkpoint_type == CheckpointType.VOICE_SELECTION:
        voice_options = [
            VoiceOption(voice_id="voice-1", voice_name="Voice 1", preview_url="/preview/1"),
            VoiceOption(voice_id="voice-2", voice_name="Voice 2", preview_url="/preview/2"),
        ]

    job = _make_job_state(
        job_id="test-job",
        status=JobStatus.AWAITING_CONFIRMATION,
        checkpoint_type=checkpoint_type,
        work_dir=tmp_dir,
        artifacts=artifacts,
        voice_options=voice_options,
    )
    fake_store.add_job(job)

    client = _build_test_client(fake_store, fake_enqueuer)

    # Build the request body for the matching endpoint (no edits for simplicity)
    endpoint_path = CHECKPOINT_ENDPOINT[checkpoint_type]
    if checkpoint_type == CheckpointType.VOICE_SELECTION:
        body = {"voice_id": "voice-1"}
    else:
        body = {}

    # Act: submit valid confirmation
    response = client.post(
        f"/api/v1/jobs/test-job/confirm/{endpoint_path}",
        json=body,
    )

    # Assert: HTTP 200
    assert response.status_code == 200, (
        f"Expected 200 for valid confirmation at '{endpoint_path}', "
        f"got {response.status_code}: {response.json()}"
    )

    # Assert: response contains status "processing"
    data = response.json()
    assert data["status"] == "processing", (
        f"Expected status 'processing', got '{data['status']}'"
    )

    # Assert: response contains correct next_step
    expected_next_step = CHECKPOINT_NEXT_STEP[checkpoint_type].value
    assert data["next_step"] == expected_next_step, (
        f"Expected next_step '{expected_next_step}', got '{data['next_step']}'"
    )

    # Assert: job_id is correct
    assert data["job_id"] == "test-job", (
        f"Expected job_id 'test-job', got '{data['job_id']}'"
    )

    # Assert: a resume task was enqueued with correct parameters
    assert len(fake_enqueuer.enqueued) == 1, (
        f"Expected exactly 1 enqueued resume task, got {len(fake_enqueuer.enqueued)}"
    )
    enqueued_job_id, enqueued_next_step = fake_enqueuer.enqueued[0]
    assert enqueued_job_id == "test-job", (
        f"Expected enqueued job_id 'test-job', got '{enqueued_job_id}'"
    )
    assert enqueued_next_step == expected_next_step, (
        f"Expected enqueued next_step '{expected_next_step}', got '{enqueued_next_step}'"
    )
