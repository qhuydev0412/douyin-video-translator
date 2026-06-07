"""Unit tests for the resume_pipeline_task Celery task."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.job import JobStatus, PipelineStep
from app.services.pipeline import CancellationError, PipelineError, PipelineResult
from app.tasks.resume_task import resume_pipeline_task


@pytest.fixture
def mock_services():
    """Mock the TranslationPipeline class and all lazy-imported dependencies.

    TranslationPipeline is imported at module level in resume_task.py, so we
    patch it there. All other dependencies (JobStore, CheckpointManager, etc.)
    are imported lazily inside the function body, so we patch them at their
    source modules.
    """
    with (
        patch("app.tasks.resume_task.TranslationPipeline") as mock_pipeline_cls,
        patch("app.services.job_store.JobStore") as mock_job_store_cls,
        patch("app.services.checkpoint_manager.CheckpointManager") as mock_cm_cls,
        patch("app.services.voice_preview.VoicePreviewGenerator") as mock_vpg_cls,
        patch("app.services.voice_synthesizer.VoiceSynthesizer") as mock_synth_cls,
        patch("app.services.downloader.VideoDownloader") as mock_dl_cls,
        patch("app.services.audio_extractor.AudioExtractor") as mock_ae_cls,
        patch("app.services.vocal_isolator.VocalIsolator") as mock_vi_cls,
        patch("app.services.speech_recognizer.SpeechRecognizer") as mock_sr_cls,
        patch("app.services.translator.Translator") as mock_tr_cls,
        patch("app.services.video_composer.VideoComposer") as mock_vc_cls,
    ):
        mock_store = MagicMock()
        mock_job_store_cls.return_value = mock_store

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        yield {
            "job_store": mock_store,
            "pipeline": mock_pipeline,
            "pipeline_cls": mock_pipeline_cls,
        }


class TestResumePipelineTask:
    """Tests for resume_pipeline_task."""

    def test_successful_completion(self, mock_services):
        """Task returns completed status when pipeline finishes."""
        output_path = Path("storage") / "jobs" / "job-123" / "output" / "video.mp4"
        expected_result = PipelineResult(
            output_path=output_path,
            job_id="job-123",
            artifacts={"output_video": str(output_path)},
        )

        async def mock_resume(job_id, from_step):
            return expected_result

        mock_services["pipeline"].resume = mock_resume

        result = resume_pipeline_task("job-123", "translating")

        assert result == {"job_id": "job-123", "status": "completed"}

    def test_pipeline_paused_at_checkpoint(self, mock_services):
        """Task returns paused status when pipeline hits another checkpoint."""

        async def mock_resume(job_id, from_step):
            return None

        mock_services["pipeline"].resume = mock_resume

        result = resume_pipeline_task("job-123", "translating")

        assert result == {"job_id": "job-123", "status": "paused"}

    def test_pipeline_error_updates_job_to_failed(self, mock_services):
        """Task updates job state to FAILED on PipelineError."""
        error = PipelineError(
            step=PipelineStep.TRANSLATING,
            message="Translation API error",
            retryable=True,
        )

        async def mock_resume(job_id, from_step):
            raise error

        mock_services["pipeline"].resume = mock_resume

        with pytest.raises(PipelineError):
            resume_pipeline_task("job-123", "translating")

        mock_services["job_store"].update_job.assert_called_once_with(
            "job-123",
            status=JobStatus.FAILED,
            error={
                "step": "translating",
                "message": "Translation API error",
                "retryable": True,
            },
        )

    def test_cancellation_returns_cancelled_status(self, mock_services):
        """Task returns cancelled status on CancellationError."""
        error = CancellationError(job_id="job-123", step=PipelineStep.TRANSLATING)

        async def mock_resume(job_id, from_step):
            raise error

        mock_services["pipeline"].resume = mock_resume

        result = resume_pipeline_task("job-123", "translating")

        assert result == {"job_id": "job-123", "status": "cancelled"}
        mock_services["job_store"].update_job.assert_not_called()

    def test_unexpected_error_updates_job_to_failed(self, mock_services):
        """Task updates job state to FAILED on unexpected exceptions."""

        async def mock_resume(job_id, from_step):
            raise RuntimeError("Something went wrong")

        mock_services["pipeline"].resume = mock_resume

        with pytest.raises(RuntimeError):
            resume_pipeline_task("job-123", "synthesizing_voice")

        mock_services["job_store"].update_job.assert_called_once_with(
            "job-123",
            status=JobStatus.FAILED,
            error={
                "step": "synthesizing_voice",
                "message": "Unexpected error: Something went wrong",
                "retryable": False,
            },
        )
