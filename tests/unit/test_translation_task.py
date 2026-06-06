"""Unit tests for the translate_video_task Celery task."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.models.job import JobStatus, PipelineStep
from app.services.pipeline import CancellationError, PipelineError, PipelineResult
from app.tasks.translation_task import revoke_task, translate_video_task


@pytest.fixture
def mock_job_store():
    """Create a mock JobStore."""
    with patch("app.services.job_store.JobStore") as mock_cls:
        store = MagicMock()
        mock_cls.return_value = store
        yield store


@pytest.fixture
def mock_pipeline():
    """Create a mock TranslationPipeline."""
    with patch("app.tasks.translation_task._create_pipeline") as mock_create:
        pipeline = MagicMock()
        mock_create.return_value = pipeline
        yield pipeline


class TestTranslateVideoTask:
    """Tests for translate_video_task."""

    def test_successful_execution(self, mock_job_store, mock_pipeline):
        """Task returns job_id and output_path on success."""
        output_path = Path("storage") / "jobs" / "job-123" / "output" / "video.mp4"
        expected_result = PipelineResult(
            output_path=output_path,
            job_id="job-123",
            artifacts={"output_video": str(output_path)},
        )

        async def mock_execute(job_id, url):
            return expected_result

        mock_pipeline.execute = mock_execute

        result = translate_video_task("job-123", "https://douyin.com/video/123")

        assert result["job_id"] == "job-123"
        assert result["output_path"] == str(output_path)

    def test_pipeline_error_updates_job_to_failed(self, mock_job_store, mock_pipeline):
        """Task updates job state to FAILED on PipelineError."""
        error = PipelineError(
            step=PipelineStep.DOWNLOADING,
            message="Download failed",
            retryable=True,
        )

        async def mock_execute(job_id, url):
            raise error

        mock_pipeline.execute = mock_execute

        with pytest.raises(PipelineError):
            translate_video_task("job-123", "https://douyin.com/video/123")

        mock_job_store.update_job.assert_called_once_with(
            "job-123",
            status=JobStatus.FAILED,
            error={
                "step": "downloading",
                "message": "Download failed",
                "retryable": True,
            },
        )

    def test_cancellation_error_does_not_update_job(self, mock_job_store, mock_pipeline):
        """Task just logs on CancellationError (job already marked CANCELLED)."""
        error = CancellationError(job_id="job-123", step=PipelineStep.TRANSLATING)

        async def mock_execute(job_id, url):
            raise error

        mock_pipeline.execute = mock_execute

        result = translate_video_task("job-123", "https://douyin.com/video/123")

        assert result == {"job_id": "job-123", "status": "cancelled"}
        mock_job_store.update_job.assert_not_called()

    def test_unexpected_error_updates_job_to_failed(self, mock_job_store, mock_pipeline):
        """Task updates job state to FAILED on unexpected exceptions."""

        async def mock_execute(job_id, url):
            raise RuntimeError("Something went wrong")

        mock_pipeline.execute = mock_execute

        with pytest.raises(RuntimeError):
            translate_video_task("job-123", "https://douyin.com/video/123")

        mock_job_store.update_job.assert_called_once_with(
            "job-123",
            status=JobStatus.FAILED,
            error={
                "step": "downloading",
                "message": "Unexpected error: Something went wrong",
                "retryable": False,
            },
        )


class TestRevokeTask:
    """Tests for revoke_task helper."""

    def test_revoke_calls_celery_control(self):
        """revoke_task sends revoke command to Celery."""
        with patch("app.tasks.translation_task.celery_app") as mock_app:
            revoke_task("abc-task-id", terminate=True)

            mock_app.control.revoke.assert_called_once_with(
                "abc-task-id", terminate=True, signal="SIGTERM"
            )

    def test_revoke_without_terminate(self):
        """revoke_task can be called without termination."""
        with patch("app.tasks.translation_task.celery_app") as mock_app:
            revoke_task("abc-task-id", terminate=False)

            mock_app.control.revoke.assert_called_once_with(
                "abc-task-id", terminate=False, signal="SIGTERM"
            )
