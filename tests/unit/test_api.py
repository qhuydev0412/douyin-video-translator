"""Unit tests for the FastAPI API endpoints."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.routes import configure_routes, router
from app.models.job import JobState, JobStatus, PipelineStep


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

    def delete_job(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)


class FakeTaskEnqueuer:
    """Fake task enqueuer that records enqueued tasks."""

    def __init__(self) -> None:
        self.enqueued: list[tuple[str, str]] = []

    def enqueue(self, job_id: str, url: str) -> str:
        self.enqueued.append((job_id, url))
        return f"task-{job_id}"


@pytest.fixture
def fake_store() -> FakeJobStore:
    return FakeJobStore()


@pytest.fixture
def fake_enqueuer() -> FakeTaskEnqueuer:
    return FakeTaskEnqueuer()


@pytest.fixture
def client(fake_store: FakeJobStore, fake_enqueuer: FakeTaskEnqueuer) -> TestClient:
    """Create a test client with fake dependencies."""
    from fastapi import FastAPI

    app = FastAPI()
    configure_routes(job_store=fake_store, task_enqueuer=fake_enqueuer)
    app.include_router(router)
    return TestClient(app)


class TestCreateTranslationJob:
    """Tests for POST /api/v1/translate."""

    def test_valid_douyin_url_returns_202(
        self, client: TestClient, fake_enqueuer: FakeTaskEnqueuer
    ) -> None:
        response = client.post(
            "/api/v1/translate",
            json={"url": "https://www.douyin.com/video/123456"},
        )
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "queued"
        assert data["message"] == "Đã tiếp nhận yêu cầu dịch video"
        assert "job_id" in data
        assert len(fake_enqueuer.enqueued) == 1

    def test_invalid_url_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/translate",
            json={"url": "https://www.youtube.com/watch?v=abc"},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error"] == "INVALID_URL"

    def test_empty_url_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/translate",
            json={"url": ""},
        )
        assert response.status_code == 400

    def test_rate_limit_exceeded_returns_429(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        # Create 5 active jobs for the test client IP (testclient)
        for i in range(5):
            fake_store.create_job(
                job_id=f"job-{i}",
                url="https://www.douyin.com/video/123",
                client_ip="testclient",
                work_dir=f"storage/jobs/job-{i}",
            )

        response = client.post(
            "/api/v1/translate",
            json={"url": "https://www.douyin.com/video/999"},
        )
        assert response.status_code == 429
        assert response.headers.get("Retry-After") == "60"

    def test_rate_limit_not_exceeded_when_jobs_completed(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        # Create 5 jobs but mark them as completed
        for i in range(5):
            fake_store.create_job(
                job_id=f"job-{i}",
                url="https://www.douyin.com/video/123",
                client_ip="testclient",
                work_dir=f"storage/jobs/job-{i}",
            )
            fake_store.update_job(f"job-{i}", status=JobStatus.COMPLETED)

        response = client.post(
            "/api/v1/translate",
            json={"url": "https://www.douyin.com/video/999"},
        )
        assert response.status_code == 202


class TestGetJobStatus:
    """Tests for GET /api/v1/jobs/{job_id}."""

    def test_existing_job_returns_status(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        job = fake_store.create_job(
            job_id="test-job-1",
            url="https://www.douyin.com/video/123",
            client_ip="127.0.0.1",
            work_dir="storage/jobs/test-job-1",
        )

        response = client.get("/api/v1/jobs/test-job-1")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job-1"
        assert data["status"] == "queued"
        assert data["progress_percent"] == 0

    def test_nonexistent_job_returns_404(self, client: TestClient) -> None:
        response = client.get("/api/v1/jobs/nonexistent-id")
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "JOB_NOT_FOUND"

    def test_processing_job_shows_current_step(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        fake_store.create_job(
            job_id="processing-job",
            url="https://www.douyin.com/video/123",
            client_ip="127.0.0.1",
            work_dir="storage/jobs/processing-job",
        )
        fake_store.update_job(
            "processing-job",
            status=JobStatus.PROCESSING,
            current_step=PipelineStep.EXTRACTING_AUDIO,
            progress_percent=20,
        )

        response = client.get("/api/v1/jobs/processing-job")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert data["current_step"] == "extracting_audio"
        assert data["progress_percent"] == 20


class TestCancelJob:
    """Tests for DELETE /api/v1/jobs/{job_id}."""

    def test_cancel_existing_job(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        fake_store.create_job(
            job_id="cancel-job",
            url="https://www.douyin.com/video/123",
            client_ip="127.0.0.1",
            work_dir="storage/jobs/cancel-job",
        )

        response = client.delete("/api/v1/jobs/cancel-job")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "cancel-job"
        assert data["status"] == "cancelled"

        # Verify job was updated
        job = fake_store.get_job("cancel-job")
        assert job.status == JobStatus.CANCELLED

    def test_cancel_nonexistent_job_returns_404(self, client: TestClient) -> None:
        response = client.delete("/api/v1/jobs/nonexistent-id")
        assert response.status_code == 404
