"""Property-based tests for status response correctness.

Feature: pipeline-preview-confirm

Tests cover:
- Property 9: Status response includes correct preview data at checkpoint
- Property 10: Non-awaiting jobs omit checkpoint fields
- Property 14: Status query resets expiration timer
"""

import json
import tempfile
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from app.api.routes import configure_routes, router
from app.models.job import (
    CheckpointType,
    JobState,
    JobStatus,
    VoiceOption,
)


# --- Test Helpers ---


class FakeJobStore:
    """In-memory fake job store for property testing."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}

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

    def count_active_jobs(self, client_ip: str) -> int:
        return 0


class FakeTaskEnqueuer:
    """Fake task enqueuer for test setup."""

    def enqueue(self, job_id: str, url: str) -> str:
        return f"task-{job_id}"


class FakeCheckpointManager:
    """Fake checkpoint manager that records reset_expiration calls."""

    def __init__(self) -> None:
        self.reset_calls: list[str] = []

    def reset_expiration(self, job_id: str) -> None:
        self.reset_calls.append(job_id)


# --- Strategies ---

checkpoint_type_strategy = st.sampled_from(CheckpointType)

non_awaiting_status_strategy = st.sampled_from([
    JobStatus.QUEUED,
    JobStatus.PROCESSING,
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
])

# Generate transcription segments with valid data
transcription_segment_strategy = st.fixed_dictionaries({
    "start": st.floats(min_value=0.0, max_value=3600.0, allow_nan=False, allow_infinity=False),
    "end": st.floats(min_value=0.0, max_value=3600.0, allow_nan=False, allow_infinity=False),
    "text": st.text(min_size=1, max_size=100, alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        blacklist_characters="\x00",
    )),
    "confidence": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
})

# Generate translation segments with valid data
translation_segment_strategy = st.fixed_dictionaries({
    "start": st.floats(min_value=0.0, max_value=3600.0, allow_nan=False, allow_infinity=False),
    "end": st.floats(min_value=0.0, max_value=3600.0, allow_nan=False, allow_infinity=False),
    "text": st.text(min_size=1, max_size=100, alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        blacklist_characters="\x00",
    )),
    "translated_text": st.text(min_size=1, max_size=100, alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        blacklist_characters="\x00",
    )),
})

# Generate voice options
voice_option_strategy = st.fixed_dictionaries({
    "voice_id": st.text(min_size=1, max_size=30, alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="-_"
    )),
    "voice_name": st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=("L", "N", "Z"),
        blacklist_characters="\x00",
    )),
})


# --- Helpers for creating test apps ---


def _create_test_app(
    fake_store: FakeJobStore,
    fake_checkpoint_manager: FakeCheckpointManager,
) -> TestClient:
    """Create a FastAPI TestClient with fakes."""
    app = FastAPI()
    configure_routes(
        job_store=fake_store,
        task_enqueuer=FakeTaskEnqueuer(),
        checkpoint_manager=fake_checkpoint_manager,
    )
    app.include_router(router)
    return TestClient(app)


# --- Property 9: Status response includes correct preview data at checkpoint ---


# Feature: pipeline-preview-confirm, Property 9: Status response includes correct preview data at checkpoint
@pytest.mark.property
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    checkpoint_type=checkpoint_type_strategy,
    transcription_segments=st.lists(transcription_segment_strategy, min_size=1, max_size=5),
    translation_segments=st.lists(translation_segment_strategy, min_size=1, max_size=5),
    voice_options=st.lists(voice_option_strategy, min_size=1, max_size=5),
)
def test_status_response_includes_correct_preview_data_at_checkpoint(
    checkpoint_type: CheckpointType,
    transcription_segments: list[dict],
    translation_segments: list[dict],
    voice_options: list[dict],
):
    """For any job in "awaiting_confirmation" status, the job status response SHALL
    include the checkpoint_type and preview_data fields with the correct data type
    for that checkpoint.

    **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
    """
    fake_store = FakeJobStore()
    fake_checkpoint_manager = FakeCheckpointManager()
    client = _create_test_app(fake_store, fake_checkpoint_manager)

    job_id = "test-job-preview"

    # Create job and set up appropriate artifacts based on checkpoint type
    fake_store.create_job(
        job_id=job_id,
        url="https://www.douyin.com/video/123",
        client_ip="127.0.0.1",
        work_dir=tempfile.mkdtemp(),
    )

    if checkpoint_type == CheckpointType.TRANSCRIPTION:
        # Write transcription JSON to a temp file
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump({"segments": transcription_segments}, tmp)
        tmp.close()
        fake_store.update_job(
            job_id,
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=checkpoint_type,
            checkpoint_entered_at=datetime.now(timezone.utc),
            artifacts={"transcription_path": tmp.name},
        )
    elif checkpoint_type == CheckpointType.TRANSLATION:
        # Write translation JSON to a temp file
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump({"segments": translation_segments}, tmp)
        tmp.close()
        fake_store.update_job(
            job_id,
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=checkpoint_type,
            checkpoint_entered_at=datetime.now(timezone.utc),
            artifacts={"translation_path": tmp.name},
        )
    elif checkpoint_type == CheckpointType.VOICE_SELECTION:
        # Set voice_options on the job
        voice_opt_models = [
            VoiceOption(
                voice_id=vo["voice_id"],
                voice_name=vo["voice_name"],
                preview_url=f"/api/v1/jobs/{job_id}/preview/voice/{vo['voice_id']}",
            )
            for vo in voice_options
        ]
        fake_store.update_job(
            job_id,
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=checkpoint_type,
            checkpoint_entered_at=datetime.now(timezone.utc),
            voice_options=voice_opt_models,
        )

    # Act: Query job status
    response = client.get(f"/api/v1/jobs/{job_id}")

    # Assert: response is 200
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()

    # Assert: checkpoint_type matches
    assert data["checkpoint_type"] == checkpoint_type.value, (
        f"Expected checkpoint_type '{checkpoint_type.value}', got '{data['checkpoint_type']}'"
    )

    # Assert: preview_data is not null
    assert data["preview_data"] is not None, (
        "preview_data should not be null for a job in awaiting_confirmation status"
    )

    # Assert: correct preview data type for the checkpoint
    if checkpoint_type == CheckpointType.TRANSCRIPTION:
        assert data["preview_data"]["transcription_segments"] is not None, (
            "transcription_segments should be present for transcription checkpoint"
        )
        assert len(data["preview_data"]["transcription_segments"]) == len(transcription_segments), (
            "Number of transcription segments should match input"
        )
        # Verify segment structure
        for seg in data["preview_data"]["transcription_segments"]:
            assert "index" in seg
            assert "start" in seg
            assert "end" in seg
            assert "text" in seg
            assert "confidence" in seg

    elif checkpoint_type == CheckpointType.TRANSLATION:
        assert data["preview_data"]["translation_segments"] is not None, (
            "translation_segments should be present for translation checkpoint"
        )
        assert len(data["preview_data"]["translation_segments"]) == len(translation_segments), (
            "Number of translation segments should match input"
        )
        # Verify segment structure
        for seg in data["preview_data"]["translation_segments"]:
            assert "index" in seg
            assert "start" in seg
            assert "end" in seg
            assert "original_text" in seg
            assert "translated_text" in seg

    elif checkpoint_type == CheckpointType.VOICE_SELECTION:
        assert data["preview_data"]["voice_options"] is not None, (
            "voice_options should be present for voice_selection checkpoint"
        )
        assert len(data["preview_data"]["voice_options"]) == len(voice_options), (
            "Number of voice options should match input"
        )
        # Verify option structure
        for opt in data["preview_data"]["voice_options"]:
            assert "voice_id" in opt
            assert "voice_name" in opt
            assert "preview_url" in opt


# --- Property 10: Non-awaiting jobs omit checkpoint fields ---


# Feature: pipeline-preview-confirm, Property 10: Non-awaiting jobs omit checkpoint fields
@pytest.mark.property
@settings(max_examples=100)
@given(
    non_awaiting_status=non_awaiting_status_strategy,
)
def test_non_awaiting_jobs_omit_checkpoint_fields(
    non_awaiting_status: JobStatus,
):
    """For any job NOT in "awaiting_confirmation" status, the status response SHALL
    return checkpoint_type as null and preview_data as null.

    **Validates: Requirements 6.6**
    """
    fake_store = FakeJobStore()
    fake_checkpoint_manager = FakeCheckpointManager()
    client = _create_test_app(fake_store, fake_checkpoint_manager)

    job_id = "test-job-non-awaiting"

    fake_store.create_job(
        job_id=job_id,
        url="https://www.douyin.com/video/123",
        client_ip="127.0.0.1",
        work_dir="/tmp/test-job",
    )
    fake_store.update_job(job_id, status=non_awaiting_status)

    # Act: Query job status
    response = client.get(f"/api/v1/jobs/{job_id}")

    # Assert
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()

    assert data["checkpoint_type"] is None, (
        f"checkpoint_type should be null for status '{non_awaiting_status.value}', "
        f"got '{data['checkpoint_type']}'"
    )
    assert data["preview_data"] is None, (
        f"preview_data should be null for status '{non_awaiting_status.value}', "
        f"got '{data['preview_data']}'"
    )

    # Assert: reset_expiration was NOT called for non-awaiting jobs
    assert job_id not in fake_checkpoint_manager.reset_calls, (
        "reset_expiration should not be called for non-awaiting jobs"
    )


# --- Property 14: Status query resets expiration timer ---


# Feature: pipeline-preview-confirm, Property 14: Status query resets expiration timer
@pytest.mark.property
@settings(max_examples=100)
@given(
    checkpoint_type=checkpoint_type_strategy,
)
def test_status_query_resets_expiration_timer(
    checkpoint_type: CheckpointType,
):
    """For any job in "awaiting_confirmation" status, querying its status SHALL
    reset the expiration timer.

    **Validates: Requirements 7.3**
    """
    fake_store = FakeJobStore()
    fake_checkpoint_manager = FakeCheckpointManager()
    client = _create_test_app(fake_store, fake_checkpoint_manager)

    job_id = "test-job-expiry-reset"

    fake_store.create_job(
        job_id=job_id,
        url="https://www.douyin.com/video/123",
        client_ip="127.0.0.1",
        work_dir=tempfile.mkdtemp(),
    )

    # Set up appropriate artifacts so the endpoint doesn't fail loading preview data
    if checkpoint_type == CheckpointType.TRANSCRIPTION:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump({"segments": [{"start": 0.0, "end": 1.0, "text": "test", "confidence": 0.9}]}, tmp)
        tmp.close()
        fake_store.update_job(
            job_id,
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=checkpoint_type,
            checkpoint_entered_at=datetime.now(timezone.utc),
            artifacts={"transcription_path": tmp.name},
        )
    elif checkpoint_type == CheckpointType.TRANSLATION:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump({"segments": [{"start": 0.0, "end": 1.0, "text": "hello", "translated_text": "xin chào"}]}, tmp)
        tmp.close()
        fake_store.update_job(
            job_id,
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=checkpoint_type,
            checkpoint_entered_at=datetime.now(timezone.utc),
            artifacts={"translation_path": tmp.name},
        )
    elif checkpoint_type == CheckpointType.VOICE_SELECTION:
        fake_store.update_job(
            job_id,
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=checkpoint_type,
            checkpoint_entered_at=datetime.now(timezone.utc),
            voice_options=[
                VoiceOption(
                    voice_id="v1",
                    voice_name="Voice 1",
                    preview_url=f"/api/v1/jobs/{job_id}/preview/voice/v1",
                ),
            ],
        )

    # Act: Query job status
    response = client.get(f"/api/v1/jobs/{job_id}")

    # Assert: response succeeded
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    # Assert: reset_expiration was called with the correct job_id
    assert job_id in fake_checkpoint_manager.reset_calls, (
        f"reset_expiration should have been called for job '{job_id}' "
        f"at checkpoint '{checkpoint_type.value}'"
    )
