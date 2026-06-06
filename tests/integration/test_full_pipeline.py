"""Integration tests for the full translation pipeline.

Tests end-to-end job creation, status tracking, file cleanup, and error propagation
using FastAPI TestClient with fake job store and task enqueuer (no external services).

Validates: Requirements 7.1, 7.2, 7.5, 8.1, 8.2
"""

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import configure_routes, router
from app.models.job import ErrorDetail, JobState, JobStatus, PipelineStep


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeJobStore:
    """In-memory fake job store for integration testing."""

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
        job.updated_at = datetime.now(timezone.utc)

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
    """Fake task enqueuer that records enqueued tasks without invoking Celery."""

    def __init__(self) -> None:
        self.enqueued: list[tuple[str, str]] = []

    def enqueue(self, job_id: str, url: str) -> str:
        self.enqueued.append((job_id, url))
        return f"task-{job_id}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_store() -> FakeJobStore:
    return FakeJobStore()


@pytest.fixture
def fake_enqueuer() -> FakeTaskEnqueuer:
    return FakeTaskEnqueuer()


@pytest.fixture
def client(fake_store: FakeJobStore, fake_enqueuer: FakeTaskEnqueuer) -> TestClient:
    """Create a FastAPI TestClient with fake dependencies."""
    app = FastAPI()
    configure_routes(job_store=fake_store, task_enqueuer=fake_enqueuer)
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Integration Tests: End-to-End Job Creation and Status Tracking
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEndToEndJobLifecycle:
    """Test the full job lifecycle: create → queued → processing → completed."""

    def test_create_job_returns_202_with_job_id(
        self, client: TestClient, fake_enqueuer: FakeTaskEnqueuer
    ) -> None:
        """POST /translate with valid Douyin URL returns 202 and a job_id."""
        response = client.post(
            "/api/v1/translate",
            json={"url": "https://www.douyin.com/video/7301234567890"},
        )

        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert len(fake_enqueuer.enqueued) == 1
        assert fake_enqueuer.enqueued[0][1] == "https://www.douyin.com/video/7301234567890"

    def test_new_job_starts_in_queued_state(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        """After creation, GET /jobs/{id} shows status 'queued'."""
        response = client.post(
            "/api/v1/translate",
            json={"url": "https://www.douyin.com/video/7301234567890"},
        )
        job_id = response.json()["job_id"]

        status_response = client.get(f"/api/v1/jobs/{job_id}")
        assert status_response.status_code == 200
        data = status_response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "queued"
        assert data["progress_percent"] == 0
        assert data["current_step"] is None

    def test_job_transitions_to_processing(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        """Simulate pipeline starting: status changes to 'processing' with step info."""
        response = client.post(
            "/api/v1/translate",
            json={"url": "https://www.douyin.com/video/7301234567890"},
        )
        job_id = response.json()["job_id"]

        # Simulate pipeline updating the job (as the Celery worker would)
        fake_store.update_job(
            job_id,
            status=JobStatus.PROCESSING,
            current_step=PipelineStep.DOWNLOADING,
            progress_percent=5,
        )

        status_response = client.get(f"/api/v1/jobs/{job_id}")
        data = status_response.json()
        assert data["status"] == "processing"
        assert data["current_step"] == "downloading"
        assert data["progress_percent"] == 5

    def test_job_progresses_through_steps(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        """Simulate pipeline progress through multiple steps."""
        response = client.post(
            "/api/v1/translate",
            json={"url": "https://www.douyin.com/video/7301234567890"},
        )
        job_id = response.json()["job_id"]

        # Simulate step progression
        steps_with_progress = [
            (PipelineStep.DOWNLOADING, 15),
            (PipelineStep.EXTRACTING_AUDIO, 25),
            (PipelineStep.ISOLATING_VOCALS, 40),
            (PipelineStep.RECOGNIZING_SPEECH, 60),
            (PipelineStep.TRANSLATING, 75),
            (PipelineStep.SYNTHESIZING_VOICE, 90),
        ]

        for step, progress in steps_with_progress:
            fake_store.update_job(
                job_id,
                status=JobStatus.PROCESSING,
                current_step=step,
                progress_percent=progress,
            )

            status_response = client.get(f"/api/v1/jobs/{job_id}")
            data = status_response.json()
            assert data["status"] == "processing"
            assert data["current_step"] == step.value
            assert data["progress_percent"] == progress

    def test_job_completes_with_download_url(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        """When pipeline completes, status is 'completed' with download_url set."""
        response = client.post(
            "/api/v1/translate",
            json={"url": "https://www.douyin.com/video/7301234567890"},
        )
        job_id = response.json()["job_id"]

        # Simulate pipeline completion
        expires = datetime.now(timezone.utc) + timedelta(hours=24)
        fake_store.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress_percent=100,
            current_step=PipelineStep.COMPOSING_VIDEO,
            download_url=f"/api/v1/jobs/{job_id}/download",
            expires_at=expires,
        )

        status_response = client.get(f"/api/v1/jobs/{job_id}")
        data = status_response.json()
        assert data["status"] == "completed"
        assert data["progress_percent"] == 100
        assert data["download_url"] == f"/api/v1/jobs/{job_id}/download"
        assert data["expires_at"] is not None

    def test_multiple_jobs_are_independent(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        """Multiple concurrent jobs track independently."""
        # Create two jobs
        resp1 = client.post(
            "/api/v1/translate",
            json={"url": "https://www.douyin.com/video/111"},
        )
        resp2 = client.post(
            "/api/v1/translate",
            json={"url": "https://www.douyin.com/video/222"},
        )
        job_id_1 = resp1.json()["job_id"]
        job_id_2 = resp2.json()["job_id"]

        # Progress job 1 but not job 2
        fake_store.update_job(
            job_id_1,
            status=JobStatus.PROCESSING,
            current_step=PipelineStep.TRANSLATING,
            progress_percent=60,
        )

        status1 = client.get(f"/api/v1/jobs/{job_id_1}").json()
        status2 = client.get(f"/api/v1/jobs/{job_id_2}").json()

        assert status1["status"] == "processing"
        assert status1["progress_percent"] == 60
        assert status2["status"] == "queued"
        assert status2["progress_percent"] == 0


# ---------------------------------------------------------------------------
# Integration Tests: File Cleanup After Expiry
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFileCleanupAfterExpiry:
    """Test that expired job directories are cleaned up properly."""

    def test_expired_directory_is_removed(self, tmp_path: Path) -> None:
        """Job directories older than FILE_EXPIRY_HOURS are removed."""
        # Create a fake job directory with old modification time
        job_dir = tmp_path / "old-job-123"
        job_dir.mkdir()
        (job_dir / "output.mp4").write_text("fake video content")

        # Set modification time to 25 hours ago (beyond 24h expiry)
        old_time = time.time() - (25 * 3600)
        os.utime(job_dir, (old_time, old_time))

        # Mock settings and job store (JobStore is lazily imported inside the task)
        fake_store = FakeJobStore()

        with (
            patch("app.tasks.cleanup_task.settings") as mock_settings,
            patch("app.services.job_store.JobStore", return_value=fake_store),
        ):
            mock_settings.STORAGE_PATH = str(tmp_path)
            mock_settings.FILE_EXPIRY_HOURS = 24

            from app.tasks.cleanup_task import cleanup_expired_jobs

            result = cleanup_expired_jobs()

        assert result["cleaned_dirs"] == 1
        assert not job_dir.exists()

    def test_non_expired_directory_is_kept(self, tmp_path: Path) -> None:
        """Job directories newer than FILE_EXPIRY_HOURS are preserved."""
        # Create a recent job directory (fresh modification time)
        job_dir = tmp_path / "recent-job-456"
        job_dir.mkdir()
        (job_dir / "output.mp4").write_text("fake video content")

        fake_store = FakeJobStore()

        with (
            patch("app.tasks.cleanup_task.settings") as mock_settings,
            patch("app.services.job_store.JobStore", return_value=fake_store),
        ):
            mock_settings.STORAGE_PATH = str(tmp_path)
            mock_settings.FILE_EXPIRY_HOURS = 24

            from app.tasks.cleanup_task import cleanup_expired_jobs

            result = cleanup_expired_jobs()

        assert result["cleaned_dirs"] == 0
        assert job_dir.exists()

    def test_mixed_expired_and_fresh_directories(self, tmp_path: Path) -> None:
        """Only expired directories are removed; fresh ones remain."""
        # Create expired directory
        expired_dir = tmp_path / "expired-job"
        expired_dir.mkdir()
        (expired_dir / "video.mp4").write_text("old content")
        old_time = time.time() - (48 * 3600)  # 48 hours old
        os.utime(expired_dir, (old_time, old_time))

        # Create fresh directory
        fresh_dir = tmp_path / "fresh-job"
        fresh_dir.mkdir()
        (fresh_dir / "video.mp4").write_text("new content")

        fake_store = FakeJobStore()

        with (
            patch("app.tasks.cleanup_task.settings") as mock_settings,
            patch("app.services.job_store.JobStore", return_value=fake_store),
        ):
            mock_settings.STORAGE_PATH = str(tmp_path)
            mock_settings.FILE_EXPIRY_HOURS = 24

            from app.tasks.cleanup_task import cleanup_expired_jobs

            result = cleanup_expired_jobs()

        assert result["cleaned_dirs"] == 1
        assert not expired_dir.exists()
        assert fresh_dir.exists()

    def test_cleanup_handles_empty_storage_path(self, tmp_path: Path) -> None:
        """Cleanup works gracefully when storage path doesn't exist."""
        nonexistent_path = tmp_path / "nonexistent"

        with patch("app.tasks.cleanup_task.settings") as mock_settings:
            mock_settings.STORAGE_PATH = str(nonexistent_path)
            mock_settings.FILE_EXPIRY_HOURS = 24

            from app.tasks.cleanup_task import cleanup_expired_jobs

            result = cleanup_expired_jobs()

        assert result["cleaned_dirs"] == 0
        assert result["cleaned_records"] == 0

    def test_dotfiles_are_skipped_during_cleanup(self, tmp_path: Path) -> None:
        """Directories starting with '.' (e.g., .gitkeep) are skipped."""
        # Create a dotfile directory that looks old
        dot_dir = tmp_path / ".gitkeep"
        dot_dir.mkdir()
        old_time = time.time() - (48 * 3600)
        os.utime(dot_dir, (old_time, old_time))

        fake_store = FakeJobStore()

        with (
            patch("app.tasks.cleanup_task.settings") as mock_settings,
            patch("app.services.job_store.JobStore", return_value=fake_store),
        ):
            mock_settings.STORAGE_PATH = str(tmp_path)
            mock_settings.FILE_EXPIRY_HOURS = 24

            from app.tasks.cleanup_task import cleanup_expired_jobs

            result = cleanup_expired_jobs()

        assert result["cleaned_dirs"] == 0
        assert dot_dir.exists()


# ---------------------------------------------------------------------------
# Integration Tests: Error Propagation from Pipeline to API Response
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestErrorPropagation:
    """Test that pipeline errors are properly propagated to API responses."""

    def test_failed_job_shows_error_details(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        """When pipeline fails, GET /jobs/{id} shows 'failed' with error info."""
        response = client.post(
            "/api/v1/translate",
            json={"url": "https://www.douyin.com/video/7301234567890"},
        )
        job_id = response.json()["job_id"]

        # Simulate pipeline failure at the downloading step
        error = ErrorDetail(
            step=PipelineStep.DOWNLOADING,
            message="Video không tồn tại hoặc đã bị xóa",
            retryable=False,
            retry_count=0,
        )
        fake_store.update_job(
            job_id,
            status=JobStatus.FAILED,
            current_step=PipelineStep.DOWNLOADING,
            error=error,
        )

        status_response = client.get(f"/api/v1/jobs/{job_id}")
        data = status_response.json()
        assert data["status"] == "failed"
        assert data["error"] is not None
        assert data["error"]["step"] == "downloading"
        assert "không tồn tại" in data["error"]["message"]
        assert data["error"]["retryable"] is False

    def test_error_at_speech_recognition_step(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        """Error at recognizing_speech step is correctly reported."""
        response = client.post(
            "/api/v1/translate",
            json={"url": "https://www.douyin.com/video/7301234567890"},
        )
        job_id = response.json()["job_id"]

        # Simulate failure at speech recognition
        error = ErrorDetail(
            step=PipelineStep.RECOGNIZING_SPEECH,
            message="Không nhận dạng được giọng nói trong file âm thanh",
            retryable=False,
            retry_count=0,
        )
        fake_store.update_job(
            job_id,
            status=JobStatus.FAILED,
            current_step=PipelineStep.RECOGNIZING_SPEECH,
            progress_percent=40,
            error=error,
        )

        status_response = client.get(f"/api/v1/jobs/{job_id}")
        data = status_response.json()
        assert data["status"] == "failed"
        assert data["error"]["step"] == "recognizing_speech"
        assert data["progress_percent"] == 40

    def test_retryable_error_at_translation_step(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        """Retryable error at translation step includes retry information."""
        response = client.post(
            "/api/v1/translate",
            json={"url": "https://www.douyin.com/video/7301234567890"},
        )
        job_id = response.json()["job_id"]

        # Simulate retryable network failure at translation
        error = ErrorDetail(
            step=PipelineStep.TRANSLATING,
            message="Google Translate API timeout after 3 retries",
            retryable=True,
            retry_count=3,
        )
        fake_store.update_job(
            job_id,
            status=JobStatus.FAILED,
            current_step=PipelineStep.TRANSLATING,
            progress_percent=60,
            error=error,
        )

        status_response = client.get(f"/api/v1/jobs/{job_id}")
        data = status_response.json()
        assert data["status"] == "failed"
        assert data["error"]["step"] == "translating"
        assert data["error"]["retryable"] is True
        assert data["error"]["retry_count"] == 3

    def test_error_at_composing_video_step(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        """Error at the final composing_video step is correctly reported."""
        response = client.post(
            "/api/v1/translate",
            json={"url": "https://www.douyin.com/video/7301234567890"},
        )
        job_id = response.json()["job_id"]

        # Simulate failure at video composition
        error = ErrorDetail(
            step=PipelineStep.COMPOSING_VIDEO,
            message="FFmpeg failed: insufficient disk space",
            retryable=True,
            retry_count=1,
        )
        fake_store.update_job(
            job_id,
            status=JobStatus.FAILED,
            current_step=PipelineStep.COMPOSING_VIDEO,
            progress_percent=90,
            error=error,
        )

        status_response = client.get(f"/api/v1/jobs/{job_id}")
        data = status_response.json()
        assert data["status"] == "failed"
        assert data["error"]["step"] == "composing_video"
        assert data["error"]["retryable"] is True
        assert data["progress_percent"] == 90

    def test_failed_job_has_no_download_url(
        self, client: TestClient, fake_store: FakeJobStore
    ) -> None:
        """A failed job should not have a download_url."""
        response = client.post(
            "/api/v1/translate",
            json={"url": "https://www.douyin.com/video/7301234567890"},
        )
        job_id = response.json()["job_id"]

        error = ErrorDetail(
            step=PipelineStep.ISOLATING_VOCALS,
            message="Demucs model failed to process audio",
            retryable=True,
            retry_count=2,
        )
        fake_store.update_job(
            job_id,
            status=JobStatus.FAILED,
            error=error,
        )

        status_response = client.get(f"/api/v1/jobs/{job_id}")
        data = status_response.json()
        assert data["status"] == "failed"
        assert data["download_url"] is None
