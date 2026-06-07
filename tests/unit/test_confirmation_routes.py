"""Unit tests for the confirmation API endpoints (transcription)."""

import json
import os
import tempfile
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.confirmation_routes import configure_confirmation_routes, router, ResumeTaskEnqueuer
from app.models.job import CheckpointType, JobState, JobStatus, PipelineStep
from app.services.checkpoint_manager import CheckpointManager


class FakeJobStore:
    """In-memory fake job store for testing confirmation routes."""

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
def fake_enqueuer() -> FakeResumeTaskEnqueuer:
    return FakeResumeTaskEnqueuer()


@pytest.fixture
def checkpoint_manager(fake_store: FakeJobStore) -> CheckpointManager:
    return CheckpointManager(fake_store)


@pytest.fixture
def client(
    fake_store: FakeJobStore,
    fake_enqueuer: FakeResumeTaskEnqueuer,
    checkpoint_manager: CheckpointManager,
) -> TestClient:
    """Create a test client with fake dependencies."""
    app = FastAPI()
    configure_confirmation_routes(
        checkpoint_manager=checkpoint_manager,
        task_enqueuer=fake_enqueuer,
    )
    app.include_router(router)
    return TestClient(app)


def _create_awaiting_job(
    fake_store: FakeJobStore,
    job_id: str = "test-job",
    checkpoint: CheckpointType = CheckpointType.TRANSCRIPTION,
    transcription_segments: list | None = None,
) -> tuple[JobState, str]:
    """Helper to create a job at transcription checkpoint with artifacts."""
    # Create a temp directory with transcription file
    tmp_dir = tempfile.mkdtemp()
    transcription_path = os.path.join(tmp_dir, "transcription.json")

    if transcription_segments is None:
        transcription_segments = [
            {"start": 0.0, "end": 1.5, "text": "你好世界", "confidence": 0.95},
            {"start": 1.5, "end": 3.0, "text": "测试段落", "confidence": 0.88},
            {"start": 3.0, "end": 4.5, "text": "第三句话", "confidence": 0.92},
        ]

    with open(transcription_path, "w", encoding="utf-8") as f:
        json.dump({"segments": transcription_segments}, f, ensure_ascii=False)

    job = fake_store.create_job(
        job_id=job_id,
        url="https://www.douyin.com/video/123",
        client_ip="127.0.0.1",
        work_dir=tmp_dir,
    )
    fake_store.update_job(
        job_id,
        status=JobStatus.AWAITING_CONFIRMATION,
        checkpoint_type=checkpoint,
        checkpoint_entered_at=datetime.now(timezone.utc),
        artifacts={"transcription_path": transcription_path},
    )
    return job, transcription_path


class TestConfirmTranscription:
    """Tests for POST /api/v1/jobs/{job_id}/confirm/transcription."""

    def test_confirm_without_edits_returns_200(
        self,
        client: TestClient,
        fake_store: FakeJobStore,
        fake_enqueuer: FakeResumeTaskEnqueuer,
    ) -> None:
        _create_awaiting_job(fake_store)

        response = client.post(
            "/api/v1/jobs/test-job/confirm/transcription",
            json={},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job"
        assert data["status"] == "processing"
        assert data["next_step"] == "translating"
        assert len(fake_enqueuer.enqueued) == 1
        assert fake_enqueuer.enqueued[0] == ("test-job", "translating")

    def test_confirm_with_edits_applies_changes(
        self,
        client: TestClient,
        fake_store: FakeJobStore,
        fake_enqueuer: FakeResumeTaskEnqueuer,
    ) -> None:
        _, transcription_path = _create_awaiting_job(fake_store)

        response = client.post(
            "/api/v1/jobs/test-job/confirm/transcription",
            json={"edits": [{"index": 0, "text": "修改后的文本"}]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert data["next_step"] == "translating"

        # Verify edit was applied
        with open(transcription_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["segments"][0]["text"] == "修改后的文本"

    def test_nonexistent_job_returns_404(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/jobs/nonexistent/confirm/transcription",
            json={},
        )
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "JOB_NOT_FOUND"

    def test_expired_job_returns_410(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        fake_store.create_job(
            job_id="expired-job",
            url="https://www.douyin.com/video/123",
            client_ip="127.0.0.1",
            work_dir="/tmp/expired",
        )
        fake_store.update_job("expired-job", status=JobStatus.EXPIRED)

        response = client.post(
            "/api/v1/jobs/expired-job/confirm/transcription",
            json={},
        )
        assert response.status_code == 410
        data = response.json()
        assert data["detail"]["error"] == "JOB_EXPIRED"

    def test_wrong_status_returns_409(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        fake_store.create_job(
            job_id="processing-job",
            url="https://www.douyin.com/video/123",
            client_ip="127.0.0.1",
            work_dir="/tmp/processing",
        )
        fake_store.update_job("processing-job", status=JobStatus.PROCESSING)

        response = client.post(
            "/api/v1/jobs/processing-job/confirm/transcription",
            json={},
        )
        assert response.status_code == 409
        data = response.json()
        assert data["detail"]["error"] == "NOT_AWAITING_CONFIRMATION"

    def test_wrong_checkpoint_returns_409(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        _create_awaiting_job(
            fake_store,
            job_id="translation-job",
            checkpoint=CheckpointType.TRANSLATION,
        )

        response = client.post(
            "/api/v1/jobs/translation-job/confirm/transcription",
            json={},
        )
        assert response.status_code == 409
        data = response.json()
        assert data["detail"]["error"] == "WRONG_CHECKPOINT"

    def test_concurrent_confirmation_returns_409(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        _create_awaiting_job(fake_store, job_id="locked-job")
        fake_store.update_job("locked-job", confirmation_lock=True)

        response = client.post(
            "/api/v1/jobs/locked-job/confirm/transcription",
            json={},
        )
        assert response.status_code == 409
        data = response.json()
        assert data["detail"]["error"] == "CONFIRMATION_IN_PROGRESS"

    def test_invalid_segment_index_returns_422(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        _create_awaiting_job(fake_store, job_id="index-job")

        response = client.post(
            "/api/v1/jobs/index-job/confirm/transcription",
            json={"edits": [{"index": 99, "text": "out of bounds"}]},
        )
        assert response.status_code == 422
        data = response.json()
        assert data["detail"]["error"] == "INVALID_SEGMENT_INDEX"

        # Verify lock was released
        job = fake_store.get_job("index-job")
        assert job.confirmation_lock is False

    def test_segment_text_exceeds_500_chars_returns_422(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        """Pydantic validation rejects text > 500 chars."""
        _create_awaiting_job(fake_store, job_id="long-text-job")

        long_text = "a" * 501
        response = client.post(
            "/api/v1/jobs/long-text-job/confirm/transcription",
            json={"edits": [{"index": 0, "text": long_text}]},
        )
        # Pydantic returns 422 for field validation errors
        assert response.status_code == 422
