"""Unit tests for GET /{job_id}/preview/voice/{voice_id} endpoint."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.confirmation_routes import configure_confirmation_routes, router
from app.models.job import CheckpointType, JobState, JobStatus
from app.services.checkpoint_manager import CheckpointManager


class FakeJobStore:
    """In-memory fake job store for testing."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}

    def create_job(self, job_id: str, url: str, work_dir: str) -> JobState:
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


@pytest.fixture
def fake_store() -> FakeJobStore:
    return FakeJobStore()


@pytest.fixture
def checkpoint_manager(fake_store: FakeJobStore) -> CheckpointManager:
    return CheckpointManager(fake_store)


@pytest.fixture
def client(fake_store: FakeJobStore, checkpoint_manager: CheckpointManager) -> TestClient:
    """Create a test client with fake dependencies."""
    app = FastAPI()
    configure_confirmation_routes(
        checkpoint_manager=checkpoint_manager,
        task_enqueuer=FakeResumeTaskEnqueuer(),
    )
    app.include_router(router)
    return TestClient(app)


class TestGetVoicePreview:
    """Tests for GET /api/v1/jobs/{job_id}/preview/voice/{voice_id}."""

    def test_returns_audio_file_when_valid(
        self, client: TestClient, fake_store: FakeJobStore, tmp_path: Path
    ) -> None:
        """Serve MP3 file when job is at voice_selection checkpoint and file exists."""
        # Setup: create job at voice_selection checkpoint with a preview file
        work_dir = str(tmp_path / "workdir")
        job = fake_store.create_job(
            job_id="job-1",
            url="https://douyin.com/video/123",
            work_dir=work_dir,
        )
        fake_store.update_job(
            "job-1",
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.VOICE_SELECTION,
        )

        # Create the preview audio file
        preview_dir = Path(work_dir) / "voice_previews"
        preview_dir.mkdir(parents=True)
        preview_file = preview_dir / "voice_abc_preview.mp3"
        preview_file.write_bytes(b"fake mp3 content")

        response = client.get("/api/v1/jobs/job-1/preview/voice/voice_abc")
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.content == b"fake mp3 content"

    def test_returns_404_when_job_not_found(self, client: TestClient) -> None:
        """Return 404 when job does not exist."""
        response = client.get("/api/v1/jobs/nonexistent/preview/voice/voice_abc")
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "JOB_NOT_FOUND"

    def test_returns_409_when_not_at_voice_selection(
        self, client: TestClient, fake_store: FakeJobStore, tmp_path: Path
    ) -> None:
        """Return 409 when job is not at the voice_selection checkpoint."""
        work_dir = str(tmp_path / "workdir")
        fake_store.create_job(
            job_id="job-2",
            url="https://douyin.com/video/123",
            work_dir=work_dir,
        )
        fake_store.update_job(
            "job-2",
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSCRIPTION,
        )

        response = client.get("/api/v1/jobs/job-2/preview/voice/voice_abc")
        assert response.status_code == 409
        data = response.json()
        assert data["detail"]["error"] == "NOT_AT_VOICE_SELECTION"

    def test_returns_409_when_job_not_awaiting_confirmation(
        self, client: TestClient, fake_store: FakeJobStore, tmp_path: Path
    ) -> None:
        """Return 409 when job is not in awaiting_confirmation status."""
        work_dir = str(tmp_path / "workdir")
        fake_store.create_job(
            job_id="job-3",
            url="https://douyin.com/video/123",
            work_dir=work_dir,
        )
        fake_store.update_job("job-3", status=JobStatus.PROCESSING)

        response = client.get("/api/v1/jobs/job-3/preview/voice/voice_abc")
        assert response.status_code == 409
        data = response.json()
        assert data["detail"]["error"] == "NOT_AT_VOICE_SELECTION"

    def test_returns_404_when_preview_file_not_found(
        self, client: TestClient, fake_store: FakeJobStore, tmp_path: Path
    ) -> None:
        """Return 404 with PREVIEW_NOT_FOUND when the audio file doesn't exist on disk."""
        work_dir = str(tmp_path / "workdir")
        fake_store.create_job(
            job_id="job-4",
            url="https://douyin.com/video/123",
            work_dir=work_dir,
        )
        fake_store.update_job(
            "job-4",
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.VOICE_SELECTION,
        )

        # Don't create the preview file
        response = client.get("/api/v1/jobs/job-4/preview/voice/nonexistent_voice")
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "PREVIEW_NOT_FOUND"

    def test_returns_409_for_completed_job(
        self, client: TestClient, fake_store: FakeJobStore, tmp_path: Path
    ) -> None:
        """Return 409 when job is in a terminal state (completed)."""
        work_dir = str(tmp_path / "workdir")
        fake_store.create_job(
            job_id="job-5",
            url="https://douyin.com/video/123",
            work_dir=work_dir,
        )
        fake_store.update_job("job-5", status=JobStatus.COMPLETED)

        response = client.get("/api/v1/jobs/job-5/preview/voice/voice_abc")
        assert response.status_code == 409
