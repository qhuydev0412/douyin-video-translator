"""Unit tests for GET /jobs/{id} checkpoint preview data."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import configure_routes, router
from app.models.job import (
    CheckpointType,
    JobState,
    JobStatus,
    PipelineStep,
    VoiceOption,
)


class FakeJobStore:
    """In-memory fake job store for testing."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._ip_jobs: dict[str, list[str]] = {}

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
        self._ip_jobs.setdefault(client_ip, []).append(job_id)
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
        job_ids = self._ip_jobs.get(client_ip, [])
        return sum(
            1
            for jid in job_ids
            if jid in self._jobs
            and self._jobs[jid].status in (JobStatus.QUEUED, JobStatus.PROCESSING)
        )


class FakeTaskEnqueuer:
    """Fake task enqueuer."""

    def enqueue(self, job_id: str, url: str) -> str:
        return f"task-{job_id}"


class FakeCheckpointManager:
    """Fake checkpoint manager that records reset_expiration calls."""

    def __init__(self) -> None:
        self.reset_calls: list[str] = []

    def reset_expiration(self, job_id: str) -> None:
        self.reset_calls.append(job_id)


@pytest.fixture
def fake_store() -> FakeJobStore:
    return FakeJobStore()


@pytest.fixture
def fake_checkpoint_manager() -> FakeCheckpointManager:
    return FakeCheckpointManager()


@pytest.fixture
def client(
    fake_store: FakeJobStore, fake_checkpoint_manager: FakeCheckpointManager
) -> TestClient:
    """Create a test client with fake dependencies including checkpoint manager."""
    app = FastAPI()
    configure_routes(
        job_store=fake_store,
        task_enqueuer=FakeTaskEnqueuer(),
        checkpoint_manager=fake_checkpoint_manager,
    )
    app.include_router(router)
    return TestClient(app)


class TestGetJobStatusCheckpointPreview:
    """Tests for GET /api/v1/jobs/{job_id} with checkpoint preview data."""

    def test_non_awaiting_job_returns_null_checkpoint_fields(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        """Jobs not in AWAITING_CONFIRMATION should return null checkpoint fields."""
        fake_store.create_job(
            job_id="job-processing",
            url="https://www.douyin.com/video/123",
            client_ip="127.0.0.1",
            work_dir="storage/jobs/job-processing",
        )
        fake_store.update_job("job-processing", status=JobStatus.PROCESSING)

        response = client.get("/api/v1/jobs/job-processing")
        assert response.status_code == 200
        data = response.json()
        assert data["checkpoint_type"] is None
        assert data["preview_data"] is None

    def test_transcription_checkpoint_returns_segments(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        """Job at transcription checkpoint returns transcription segments."""
        # Create a temp transcription JSON file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "segments": [
                        {"start": 0.0, "end": 2.5, "text": "你好", "confidence": 0.95},
                        {"start": 2.5, "end": 5.0, "text": "世界", "confidence": 0.88},
                    ]
                },
                f,
            )
            transcription_path = f.name

        fake_store.create_job(
            job_id="job-transcription",
            url="https://www.douyin.com/video/123",
            client_ip="127.0.0.1",
            work_dir="storage/jobs/job-transcription",
        )
        fake_store.update_job(
            "job-transcription",
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSCRIPTION,
            artifacts={"transcription_path": transcription_path},
        )

        response = client.get("/api/v1/jobs/job-transcription")
        assert response.status_code == 200
        data = response.json()

        assert data["checkpoint_type"] == "transcription"
        assert data["preview_data"] is not None
        assert data["preview_data"]["transcription_segments"] is not None
        assert len(data["preview_data"]["transcription_segments"]) == 2

        seg0 = data["preview_data"]["transcription_segments"][0]
        assert seg0["index"] == 0
        assert seg0["start"] == 0.0
        assert seg0["end"] == 2.5
        assert seg0["text"] == "你好"
        assert seg0["confidence"] == 0.95

        seg1 = data["preview_data"]["transcription_segments"][1]
        assert seg1["index"] == 1
        assert seg1["start"] == 2.5
        assert seg1["end"] == 5.0
        assert seg1["text"] == "世界"
        assert seg1["confidence"] == 0.88

        # Cleanup
        Path(transcription_path).unlink(missing_ok=True)

    def test_translation_checkpoint_returns_segments(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        """Job at translation checkpoint returns translation segments."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 2.5,
                            "text": "你好",
                            "translated_text": "Xin chào",
                        },
                        {
                            "start": 2.5,
                            "end": 5.0,
                            "text": "世界",
                            "translated_text": "Thế giới",
                        },
                    ]
                },
                f,
            )
            translation_path = f.name

        fake_store.create_job(
            job_id="job-translation",
            url="https://www.douyin.com/video/123",
            client_ip="127.0.0.1",
            work_dir="storage/jobs/job-translation",
        )
        fake_store.update_job(
            "job-translation",
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSLATION,
            artifacts={"translation_path": translation_path},
        )

        response = client.get("/api/v1/jobs/job-translation")
        assert response.status_code == 200
        data = response.json()

        assert data["checkpoint_type"] == "translation"
        assert data["preview_data"] is not None
        assert data["preview_data"]["translation_segments"] is not None
        assert len(data["preview_data"]["translation_segments"]) == 2

        seg0 = data["preview_data"]["translation_segments"][0]
        assert seg0["index"] == 0
        assert seg0["start"] == 0.0
        assert seg0["end"] == 2.5
        assert seg0["original_text"] == "你好"
        assert seg0["translated_text"] == "Xin chào"

        # Cleanup
        Path(translation_path).unlink(missing_ok=True)

    def test_voice_selection_checkpoint_returns_options(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        """Job at voice_selection checkpoint returns voice options."""
        fake_store.create_job(
            job_id="job-voice",
            url="https://www.douyin.com/video/123",
            client_ip="127.0.0.1",
            work_dir="storage/jobs/job-voice",
        )
        fake_store.update_job(
            "job-voice",
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.VOICE_SELECTION,
            voice_options=[
                VoiceOption(
                    voice_id="voice-1",
                    voice_name="Female Voice A",
                    preview_url="/api/v1/jobs/job-voice/preview/voice/voice-1",
                ),
                VoiceOption(
                    voice_id="voice-2",
                    voice_name="Male Voice B",
                    preview_url="/api/v1/jobs/job-voice/preview/voice/voice-2",
                ),
                VoiceOption(
                    voice_id="voice-3",
                    voice_name="Female Voice C",
                    preview_url="/api/v1/jobs/job-voice/preview/voice/voice-3",
                ),
            ],
        )

        response = client.get("/api/v1/jobs/job-voice")
        assert response.status_code == 200
        data = response.json()

        assert data["checkpoint_type"] == "voice_selection"
        assert data["preview_data"] is not None
        assert data["preview_data"]["voice_options"] is not None
        assert len(data["preview_data"]["voice_options"]) == 3

        opt0 = data["preview_data"]["voice_options"][0]
        assert opt0["voice_id"] == "voice-1"
        assert opt0["voice_name"] == "Female Voice A"
        assert opt0["preview_url"] == "/api/v1/jobs/job-voice/preview/voice/voice-1"

    def test_checkpoint_resets_expiration_timer(
        self,
        client: TestClient,
        fake_store: FakeJobStore,
        fake_checkpoint_manager: FakeCheckpointManager,
    ) -> None:
        """Querying status at checkpoint should call reset_expiration."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"segments": [{"start": 0.0, "end": 1.0, "text": "hello", "confidence": 0.9}]}, f)
            transcription_path = f.name

        fake_store.create_job(
            job_id="job-expiry",
            url="https://www.douyin.com/video/123",
            client_ip="127.0.0.1",
            work_dir="storage/jobs/job-expiry",
        )
        fake_store.update_job(
            "job-expiry",
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSCRIPTION,
            artifacts={"transcription_path": transcription_path},
        )

        response = client.get("/api/v1/jobs/job-expiry")
        assert response.status_code == 200

        # Verify reset_expiration was called
        assert "job-expiry" in fake_checkpoint_manager.reset_calls

        # Cleanup
        Path(transcription_path).unlink(missing_ok=True)

    def test_transcription_default_confidence_zero(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        """Segments without confidence field should default to 0.0."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(
                {"segments": [{"start": 0.0, "end": 1.0, "text": "test"}]},
                f,
            )
            transcription_path = f.name

        fake_store.create_job(
            job_id="job-noconf",
            url="https://www.douyin.com/video/123",
            client_ip="127.0.0.1",
            work_dir="storage/jobs/job-noconf",
        )
        fake_store.update_job(
            "job-noconf",
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSCRIPTION,
            artifacts={"transcription_path": transcription_path},
        )

        response = client.get("/api/v1/jobs/job-noconf")
        data = response.json()
        seg = data["preview_data"]["transcription_segments"][0]
        assert seg["confidence"] == 0.0

        Path(transcription_path).unlink(missing_ok=True)

    def test_missing_artifact_file_returns_null_preview(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        """If the artifact file doesn't exist, preview_data should be null."""
        fake_store.create_job(
            job_id="job-missing",
            url="https://www.douyin.com/video/123",
            client_ip="127.0.0.1",
            work_dir="storage/jobs/job-missing",
        )
        fake_store.update_job(
            "job-missing",
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSCRIPTION,
            artifacts={"transcription_path": "/nonexistent/path/transcription.json"},
        )

        response = client.get("/api/v1/jobs/job-missing")
        assert response.status_code == 200
        data = response.json()
        assert data["checkpoint_type"] == "transcription"
        assert data["preview_data"] is None

    def test_queued_job_returns_null_checkpoint(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        """Queued jobs should not include checkpoint data."""
        fake_store.create_job(
            job_id="job-queued",
            url="https://www.douyin.com/video/123",
            client_ip="127.0.0.1",
            work_dir="storage/jobs/job-queued",
        )

        response = client.get("/api/v1/jobs/job-queued")
        data = response.json()
        assert data["checkpoint_type"] is None
        assert data["preview_data"] is None

    def test_completed_job_returns_null_checkpoint(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        """Completed jobs should not include checkpoint data."""
        fake_store.create_job(
            job_id="job-done",
            url="https://www.douyin.com/video/123",
            client_ip="127.0.0.1",
            work_dir="storage/jobs/job-done",
        )
        fake_store.update_job("job-done", status=JobStatus.COMPLETED)

        response = client.get("/api/v1/jobs/job-done")
        data = response.json()
        assert data["checkpoint_type"] is None
        assert data["preview_data"] is None
