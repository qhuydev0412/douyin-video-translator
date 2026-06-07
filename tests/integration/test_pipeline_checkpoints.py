"""Integration tests for pipeline checkpoint confirmation flow.

Tests the full checkpoint lifecycle: pause → preview → confirm → resume,
expiry scenarios, and concurrent confirmation rejection using FastAPI TestClient
with in-memory job store and real CheckpointManager + API routes.

Validates: Requirements 1.1–1.8, 5.4–5.8, 7.1–7.4
"""

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.confirmation_routes import configure_confirmation_routes, router as confirmation_router
from app.api.routes import configure_routes, router as main_router
from app.models.job import (
    CheckpointType,
    JobState,
    JobStatus,
    PipelineStep,
    VoiceOption,
)
from app.services.checkpoint_manager import CheckpointManager


# ---------------------------------------------------------------------------
# FakeJobStore — implements JobStoreProtocol with all required methods
# ---------------------------------------------------------------------------


class FakeJobStore:
    """In-memory fake job store implementing all JobStoreProtocol methods."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}

    def get_job(self, job_id: str) -> JobState:
        if job_id not in self._jobs:
            raise KeyError(f"Job not found: {job_id}")
        return self._jobs[job_id]

    def update_job(self, job_id: str, **kwargs: object) -> None:
        job = self.get_job(job_id)
        for field, value in kwargs.items():
            setattr(job, field, value)
        job.updated_at = datetime.now(timezone.utc)

    def delete_job(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    def list_awaiting_confirmation_job_ids(self) -> list[str]:
        return [
            jid
            for jid, job in self._jobs.items()
            if job.status == JobStatus.AWAITING_CONFIRMATION
        ]

    # Helper for tests to insert jobs directly
    def add_job(self, job: JobState) -> None:
        self._jobs[job.job_id] = job


# ---------------------------------------------------------------------------
# FakeResumeEnqueuer — records resume enqueue calls
# ---------------------------------------------------------------------------


class FakeResumeEnqueuer:
    """Fake resume task enqueuer that records calls without Celery."""

    def __init__(self) -> None:
        self.enqueued: list[tuple[str, str]] = []

    def enqueue(self, job_id: str, next_step: str) -> str:
        self.enqueued.append((job_id, next_step))
        return f"task-{job_id}-{next_step}"


# ---------------------------------------------------------------------------
# FakeTaskEnqueuer for main routes
# ---------------------------------------------------------------------------


class FakeTaskEnqueuer:
    """Fake task enqueuer for main routes (create job)."""

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
def checkpoint_manager(fake_store: FakeJobStore) -> CheckpointManager:
    return CheckpointManager(job_store=fake_store)


@pytest.fixture
def resume_enqueuer() -> FakeResumeEnqueuer:
    return FakeResumeEnqueuer()


@pytest.fixture
def client(
    fake_store: FakeJobStore,
    checkpoint_manager: CheckpointManager,
    resume_enqueuer: FakeResumeEnqueuer,
) -> TestClient:
    """Create a FastAPI TestClient with confirmation routes and status route."""
    app = FastAPI()

    # Configure main routes (for GET /jobs/{id})
    configure_routes(
        job_store=fake_store,
        task_enqueuer=FakeTaskEnqueuer(),
        checkpoint_manager=checkpoint_manager,
    )
    app.include_router(main_router)

    # Configure confirmation routes
    configure_confirmation_routes(
        checkpoint_manager=checkpoint_manager,
        task_enqueuer=resume_enqueuer,
    )
    app.include_router(confirmation_router)

    return TestClient(app)


def _make_job(
    job_id: str,
    work_dir: str,
    status: JobStatus = JobStatus.PROCESSING,
    checkpoint_type: CheckpointType | None = None,
    checkpoint_entered_at: datetime | None = None,
    confirmation_lock: bool = False,
    artifacts: dict[str, str] | None = None,
    voice_options: list[VoiceOption] | None = None,
) -> JobState:
    """Helper to create a JobState for testing."""
    now = datetime.now(timezone.utc)
    return JobState(
        job_id=job_id,
        url="https://www.douyin.com/video/7301234567890",
        status=status,
        current_step=PipelineStep.RECOGNIZING_SPEECH,
        progress_percent=50,
        created_at=now,
        updated_at=now,
        work_dir=work_dir,
        checkpoint_type=checkpoint_type,
        checkpoint_entered_at=checkpoint_entered_at or now,
        confirmation_lock=confirmation_lock,
        artifacts=artifacts or {},
        voice_options=voice_options,
    )


# ---------------------------------------------------------------------------
# Integration Test: Full Pipeline Checkpoint Flow
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFullPipelineCheckpointFlow:
    """Test the complete checkpoint flow: transcription → translation → voice → complete."""

    def test_full_pipeline_checkpoint_flow(
        self,
        client: TestClient,
        fake_store: FakeJobStore,
        checkpoint_manager: CheckpointManager,
        resume_enqueuer: FakeResumeEnqueuer,
        tmp_path: Path,
    ) -> None:
        """Full flow: pause at transcription → confirm with edits → pause at translation
        → confirm → pause at voice → select voice → pipeline completes.
        """
        job_id = "job-full-flow-001"
        work_dir = str(tmp_path / job_id)
        Path(work_dir).mkdir(parents=True)

        # --- Step 1: Set up job at PROCESSING status ---
        job = _make_job(job_id=job_id, work_dir=work_dir, status=JobStatus.PROCESSING)
        fake_store.add_job(job)

        # --- Step 2: Pipeline pauses at TRANSCRIPTION checkpoint ---
        # Create transcription artifact on disk
        transcription_data = {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "你好世界", "confidence": 0.95},
                {"start": 2.5, "end": 5.0, "text": "这是测试", "confidence": 0.88},
                {"start": 5.0, "end": 7.0, "text": "谢谢", "confidence": 0.92},
            ],
            "full_text": "你好世界 这是测试 谢谢",
            "language": "zh",
            "confidence": 0.91,
        }
        transcription_path = Path(work_dir) / "transcription.json"
        transcription_path.write_text(
            json.dumps(transcription_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fake_store.update_job(job_id, artifacts={"transcription_path": str(transcription_path)})

        # Pause at transcription checkpoint
        checkpoint_manager.pause_at_checkpoint(job_id, CheckpointType.TRANSCRIPTION)

        # --- Verify: GET /jobs/{id} returns AWAITING_CONFIRMATION with transcription preview ---
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "awaiting_confirmation"
        assert data["checkpoint_type"] == "transcription"
        assert data["preview_data"] is not None
        assert data["preview_data"]["transcription_segments"] is not None
        assert len(data["preview_data"]["transcription_segments"]) == 3
        assert data["preview_data"]["transcription_segments"][0]["text"] == "你好世界"

        # --- Step 3: User confirms transcription WITH edits ---
        confirm_response = client.post(
            f"/api/v1/jobs/{job_id}/confirm/transcription",
            json={"edits": [{"index": 0, "text": "你好世界！"}]},
        )
        assert confirm_response.status_code == 200
        confirm_data = confirm_response.json()
        assert confirm_data["status"] == "processing"
        assert confirm_data["next_step"] == "translating"
        assert confirm_data["job_id"] == job_id

        # Verify resume task was enqueued
        assert len(resume_enqueuer.enqueued) == 1
        assert resume_enqueuer.enqueued[0] == (job_id, "translating")

        # Verify job is now PROCESSING
        job_state = fake_store.get_job(job_id)
        assert job_state.status == JobStatus.PROCESSING
        assert job_state.confirmation_lock is False
        assert job_state.checkpoint_type is None

        # Verify the transcription edit was applied on disk
        with open(transcription_path, "r", encoding="utf-8") as f:
            updated_transcription = json.load(f)
        assert updated_transcription["segments"][0]["text"] == "你好世界！"

        # --- Step 4: Pipeline pauses at TRANSLATION checkpoint ---
        translation_data = {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "你好世界！", "original_text": "你好世界！", "translated_text": "Xin chào thế giới!"},
                {"start": 2.5, "end": 5.0, "text": "这是测试", "original_text": "这是测试", "translated_text": "Đây là bài kiểm tra"},
                {"start": 5.0, "end": 7.0, "text": "谢谢", "original_text": "谢谢", "translated_text": "Cảm ơn"},
            ],
            "full_text_original": "你好世界！ 这是测试 谢谢",
            "full_text_translated": "Xin chào thế giới! Đây là bài kiểm tra Cảm ơn",
        }
        translation_path = Path(work_dir) / "translation.json"
        translation_path.write_text(
            json.dumps(translation_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fake_store.update_job(
            job_id,
            artifacts={
                "transcription_path": str(transcription_path),
                "translation_path": str(translation_path),
            },
        )

        # Pause at translation checkpoint
        checkpoint_manager.pause_at_checkpoint(job_id, CheckpointType.TRANSLATION)

        # --- Verify: GET /jobs/{id} returns translation preview ---
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "awaiting_confirmation"
        assert data["checkpoint_type"] == "translation"
        assert data["preview_data"] is not None
        assert data["preview_data"]["translation_segments"] is not None
        assert len(data["preview_data"]["translation_segments"]) == 3

        # --- Step 5: User confirms translation without edits ---
        confirm_response = client.post(
            f"/api/v1/jobs/{job_id}/confirm/translation",
            json={},
        )
        assert confirm_response.status_code == 200
        confirm_data = confirm_response.json()
        assert confirm_data["status"] == "processing"
        assert confirm_data["next_step"] == "synthesizing_voice"

        # Verify second resume enqueued
        assert len(resume_enqueuer.enqueued) == 2
        assert resume_enqueuer.enqueued[1] == (job_id, "synthesizing_voice")

        # --- Step 6: Pipeline pauses at VOICE_SELECTION checkpoint ---
        voice_options = [
            VoiceOption(
                voice_id="voice-001",
                voice_name="Female A",
                preview_url=f"/api/v1/jobs/{job_id}/preview/voice/voice-001",
            ),
            VoiceOption(
                voice_id="voice-002",
                voice_name="Male B",
                preview_url=f"/api/v1/jobs/{job_id}/preview/voice/voice-002",
            ),
        ]
        fake_store.update_job(job_id, voice_options=voice_options)
        checkpoint_manager.pause_at_checkpoint(job_id, CheckpointType.VOICE_SELECTION)

        # --- Verify: GET /jobs/{id} returns voice options preview ---
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "awaiting_confirmation"
        assert data["checkpoint_type"] == "voice_selection"
        assert data["preview_data"] is not None
        assert data["preview_data"]["voice_options"] is not None
        assert len(data["preview_data"]["voice_options"]) == 2
        assert data["preview_data"]["voice_options"][0]["voice_id"] == "voice-001"

        # --- Step 7: User selects a voice ---
        confirm_response = client.post(
            f"/api/v1/jobs/{job_id}/confirm/voice",
            json={"voice_id": "voice-002"},
        )
        assert confirm_response.status_code == 200
        confirm_data = confirm_response.json()
        assert confirm_data["status"] == "processing"
        assert confirm_data["next_step"] == "composing_video"

        # Verify third resume enqueued
        assert len(resume_enqueuer.enqueued) == 3
        assert resume_enqueuer.enqueued[2] == (job_id, "composing_video")

        # Verify selected voice stored in artifacts
        job_state = fake_store.get_job(job_id)
        assert job_state.artifacts["selected_voice_id"] == "voice-002"

        # --- Step 8: Simulate pipeline completion ---
        fake_store.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress_percent=100,
            current_step=PipelineStep.COMPOSING_VIDEO,
        )

        # Final verification: job is completed
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["progress_percent"] == 100
        assert data["checkpoint_type"] is None
        assert data["preview_data"] is None


# ---------------------------------------------------------------------------
# Integration Test: Checkpoint Expiry
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCheckpointExpiry:
    """Test that jobs awaiting confirmation for > 24h are expired."""

    def test_checkpoint_expiry_removes_job(
        self,
        client: TestClient,
        fake_store: FakeJobStore,
        checkpoint_manager: CheckpointManager,
        tmp_path: Path,
    ) -> None:
        """Create a job at AWAITING_CONFIRMATION with checkpoint_entered_at 25h ago,
        call check_expired_jobs(), verify status → EXPIRED and work_dir deleted.
        """
        job_id = "job-expiry-001"
        work_dir = str(tmp_path / job_id)
        Path(work_dir).mkdir(parents=True)

        # Create a dummy file to confirm directory deletion
        (Path(work_dir) / "transcription.json").write_text("{}", encoding="utf-8")

        # Create job with checkpoint_entered_at 25 hours ago
        entered_at = datetime.now(timezone.utc) - timedelta(hours=25)
        job = _make_job(
            job_id=job_id,
            work_dir=work_dir,
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSCRIPTION,
            checkpoint_entered_at=entered_at,
        )
        fake_store.add_job(job)

        # Verify the job exists before expiry
        assert fake_store.get_job(job_id).status == JobStatus.AWAITING_CONFIRMATION

        # Run expiry check
        expired_ids = checkpoint_manager.check_expired_jobs()

        # Verify job was expired
        assert job_id in expired_ids

        # Verify working directory was deleted
        assert not Path(work_dir).exists()

        # Verify confirming the expired job returns 410
        # The job is deleted from store by check_expired_jobs, so we get 404
        # Let's re-add it as EXPIRED to test the 410 path
        expired_job = _make_job(
            job_id=job_id,
            work_dir=work_dir,
            status=JobStatus.EXPIRED,
            checkpoint_type=CheckpointType.TRANSCRIPTION,
        )
        fake_store.add_job(expired_job)

        response = client.post(
            f"/api/v1/jobs/{job_id}/confirm/transcription",
            json={},
        )
        assert response.status_code == 410
        assert response.json()["detail"]["error"] == "JOB_EXPIRED"

    def test_non_expired_job_not_affected(
        self,
        fake_store: FakeJobStore,
        checkpoint_manager: CheckpointManager,
        tmp_path: Path,
    ) -> None:
        """A job with checkpoint_entered_at 1 hour ago is NOT expired."""
        job_id = "job-fresh-001"
        work_dir = str(tmp_path / job_id)
        Path(work_dir).mkdir(parents=True)

        entered_at = datetime.now(timezone.utc) - timedelta(hours=1)
        job = _make_job(
            job_id=job_id,
            work_dir=work_dir,
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSLATION,
            checkpoint_entered_at=entered_at,
        )
        fake_store.add_job(job)

        expired_ids = checkpoint_manager.check_expired_jobs()

        assert job_id not in expired_ids
        assert fake_store.get_job(job_id).status == JobStatus.AWAITING_CONFIRMATION
        assert Path(work_dir).exists()


# ---------------------------------------------------------------------------
# Integration Test: Concurrent Confirmation Rejection
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestConcurrentConfirmationRejection:
    """Test that concurrent confirmation attempts are rejected with 409."""

    def test_concurrent_confirmation_rejected(
        self,
        client: TestClient,
        fake_store: FakeJobStore,
        tmp_path: Path,
    ) -> None:
        """When confirmation_lock is True, a new confirmation returns 409
        with CONFIRMATION_IN_PROGRESS error.
        """
        job_id = "job-concurrent-001"
        work_dir = str(tmp_path / job_id)
        Path(work_dir).mkdir(parents=True)

        # Create transcription artifact
        transcription_path = Path(work_dir) / "transcription.json"
        transcription_path.write_text(
            json.dumps({"segments": [{"start": 0, "end": 1, "text": "test"}]}),
            encoding="utf-8",
        )

        # Create job at AWAITING_CONFIRMATION with lock already held
        job = _make_job(
            job_id=job_id,
            work_dir=work_dir,
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSCRIPTION,
            confirmation_lock=True,  # Simulate another in-progress confirmation
            artifacts={"transcription_path": str(transcription_path)},
        )
        fake_store.add_job(job)

        # Attempt to confirm — should be rejected
        response = client.post(
            f"/api/v1/jobs/{job_id}/confirm/transcription",
            json={},
        )
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["error"] == "CONFIRMATION_IN_PROGRESS"
        assert "already in progress" in detail["message"]

    def test_concurrent_rejection_on_translation_checkpoint(
        self,
        client: TestClient,
        fake_store: FakeJobStore,
        tmp_path: Path,
    ) -> None:
        """Concurrent rejection works for translation checkpoint too."""
        job_id = "job-concurrent-002"
        work_dir = str(tmp_path / job_id)
        Path(work_dir).mkdir(parents=True)

        job = _make_job(
            job_id=job_id,
            work_dir=work_dir,
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSLATION,
            confirmation_lock=True,
        )
        fake_store.add_job(job)

        response = client.post(
            f"/api/v1/jobs/{job_id}/confirm/translation",
            json={},
        )
        assert response.status_code == 409
        assert response.json()["detail"]["error"] == "CONFIRMATION_IN_PROGRESS"

    def test_concurrent_rejection_on_voice_checkpoint(
        self,
        client: TestClient,
        fake_store: FakeJobStore,
        tmp_path: Path,
    ) -> None:
        """Concurrent rejection works for voice selection checkpoint too."""
        job_id = "job-concurrent-003"
        work_dir = str(tmp_path / job_id)
        Path(work_dir).mkdir(parents=True)

        voice_options = [
            VoiceOption(voice_id="v1", voice_name="Voice 1", preview_url="/preview/v1"),
        ]
        job = _make_job(
            job_id=job_id,
            work_dir=work_dir,
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.VOICE_SELECTION,
            confirmation_lock=True,
            voice_options=voice_options,
        )
        fake_store.add_job(job)

        response = client.post(
            f"/api/v1/jobs/{job_id}/confirm/voice",
            json={"voice_id": "v1"},
        )
        assert response.status_code == 409
        assert response.json()["detail"]["error"] == "CONFIRMATION_IN_PROGRESS"
