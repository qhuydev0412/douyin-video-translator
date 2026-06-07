"""Celery task for video translation pipeline execution."""

import asyncio
import logging

from celery import Task

from app.core.celery_app import celery_app
from app.models.job import JobStatus, PipelineStep
from app.services.audio_extractor import AudioExtractor
from app.services.downloader import VideoDownloader
from app.services.pipeline import (
    CancellationError,
    JobStoreProtocol,
    PipelineError,
    TranslationPipeline,
)
from app.services.speech_recognizer import SpeechRecognizer
from app.services.translator import Translator
from app.services.video_composer import VideoComposer
from app.services.vocal_isolator import VocalIsolator
from app.services.voice_synthesizer import VoiceSynthesizer

logger = logging.getLogger(__name__)


def _create_pipeline(job_store: JobStoreProtocol) -> TranslationPipeline:
    """Create a TranslationPipeline with all service dependencies.

    Args:
        job_store: Job state persistence backend.

    Returns:
        Configured TranslationPipeline instance.
    """
    from app.services.checkpoint_manager import CheckpointManager
    from app.services.voice_preview import VoicePreviewGenerator

    synthesizer = VoiceSynthesizer()
    checkpoint_manager = CheckpointManager(job_store)
    voice_preview_generator = VoicePreviewGenerator(synthesizer)

    return TranslationPipeline(
        downloader=VideoDownloader(),
        extractor=AudioExtractor(),
        isolator=VocalIsolator(),
        recognizer=SpeechRecognizer(),
        translator=Translator(),
        synthesizer=synthesizer,
        composer=VideoComposer(),
        job_store=job_store,
        checkpoint_manager=checkpoint_manager,
        voice_preview_generator=voice_preview_generator,
    )


@celery_app.task(
    name="translate_video",
    bind=True,
    max_retries=0,
    acks_late=True,
)
def translate_video_task(self: Task, job_id: str, url: str) -> dict[str, str]:
    """Execute the full video translation pipeline as a Celery task.

    Creates all service dependencies, builds the pipeline, and runs
    the async execute() method inside asyncio.run().

    Args:
        self: Bound Celery task instance.
        job_id: Unique job identifier.
        url: Douyin video URL to translate.

    Returns:
        Dict with job_id and output_path on success.
    """
    from app.services.job_store import JobStore  # noqa: WPS433 - lazy import

    logger.info("Starting translation task for job %s, url=%s", job_id, url)

    job_store = JobStore()
    pipeline = _create_pipeline(job_store)

    try:
        result = asyncio.run(pipeline.execute(job_id, url))

        if result is None:
            # Pipeline paused at a checkpoint — task completes cleanly
            logger.info("Pipeline paused at checkpoint for job %s", job_id)
            return {"job_id": job_id, "status": "paused"}

        logger.info("Translation completed for job %s: %s", job_id, result.output_path)
        return {"job_id": job_id, "output_path": str(result.output_path)}

    except PipelineError as exc:
        logger.error(
            "Pipeline error for job %s at step %s: %s",
            job_id,
            exc.step.value,
            exc.message,
        )
        job_store.update_job(
            job_id,
            status=JobStatus.FAILED,
            error={
                "step": exc.step.value,
                "message": exc.message,
                "retryable": exc.retryable,
            },
        )
        raise

    except CancellationError as exc:
        # Job already marked as CANCELLED in the store by the cancel endpoint.
        logger.info("Job %s was cancelled at step %s", exc.job_id, exc.step.value)
        return {"job_id": job_id, "status": "cancelled"}

    except Exception as exc:
        logger.exception("Unexpected error for job %s: %s", job_id, str(exc))
        job_store.update_job(
            job_id,
            status=JobStatus.FAILED,
            error={
                "step": PipelineStep.DOWNLOADING.value,
                "message": f"Unexpected error: {exc}",
                "retryable": False,
            },
        )
        raise


def revoke_task(task_id: str, terminate: bool = True) -> None:
    """Revoke a running Celery task for cancellation support.

    When the API cancels a job, it should:
    1. Update Redis job status to CANCELLED
    2. Call this function to revoke the Celery task

    The pipeline will detect the CANCELLED status at the next
    cancellation check point between steps.

    Args:
        task_id: The Celery task ID to revoke.
        terminate: Whether to send SIGTERM to the worker process.
    """
    logger.info("Revoking task %s (terminate=%s)", task_id, terminate)
    celery_app.control.revoke(task_id, terminate=terminate, signal="SIGTERM")
