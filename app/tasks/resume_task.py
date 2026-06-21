"""Celery task for resuming pipeline execution after checkpoint confirmation."""

import asyncio
import logging

from celery import Task

from app.core.celery_app import celery_app
from app.models.job import JobStatus, PipelineStep
from app.services.pipeline import (
    CancellationError,
    CheckpointPauseSignal,
    PipelineError,
    TranslationPipeline,
)

logger = logging.getLogger(__name__)


@celery_app.task(
    name="resume_pipeline",
    bind=True,
    max_retries=0,
    acks_late=True,
)
def resume_pipeline_task(self: Task, job_id: str, from_step: str) -> dict[str, str]:
    """Resume pipeline execution from a specific step after confirmation.

    Creates all service dependencies, builds the pipeline with checkpoint support,
    and runs the async resume() method inside asyncio.run().

    Args:
        self: Bound Celery task instance.
        job_id: Unique job identifier.
        from_step: PipelineStep value to resume from.

    Returns:
        Dict with job_id and status ("completed", "paused", or "cancelled").
    """
    from app.services.audio_extractor import AudioExtractor
    from app.services.checkpoint_manager import CheckpointManager
    from app.services.downloader import VideoDownloader
    from app.services.gender_detector import GenderDetector
    from app.services.job_store import JobStore
    from app.services.speech_recognizer import SpeechRecognizer
    from app.services.subtitle_extractor import SubtitleExtractor
    from app.services.translator import Translator
    from app.services.video_composer import VideoComposer
    from app.services.vocal_isolator import VocalIsolator
    from app.services.voice_preview import VoicePreviewGenerator
    from app.services.voice_synthesizer import VoiceSynthesizer

    logger.info("Resuming pipeline for job %s from step '%s'", job_id, from_step)

    job_store = JobStore()
    checkpoint_manager = CheckpointManager(job_store)
    synthesizer = VoiceSynthesizer()
    voice_preview_generator = VoicePreviewGenerator(synthesizer)

    pipeline = TranslationPipeline(
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
        gender_detector=GenderDetector(),
        subtitle_extractor=SubtitleExtractor(),
    )

    try:
        result = asyncio.run(pipeline.resume(job_id, from_step))

        if result is None:
            # Pipeline paused at another checkpoint
            logger.info("Pipeline paused at another checkpoint for job %s", job_id)
            return {"job_id": job_id, "status": "paused"}

        logger.info(
            "Pipeline completed for job %s: %s", job_id, result.output_path
        )
        return {"job_id": job_id, "status": "completed"}

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
                "retry_count": 0,
            },
        )
        raise

    except CancellationError as exc:
        logger.info("Job %s was cancelled at step %s", exc.job_id, exc.step.value)
        return {"job_id": job_id, "status": "cancelled"}

    except Exception as exc:
        logger.exception("Unexpected error resuming job %s: %s", job_id, str(exc))
        job_store.update_job(
            job_id,
            status=JobStatus.FAILED,
            error={
                "step": from_step,
                "message": f"Unexpected error: {exc}",
                "retryable": False,
                "retry_count": 0,
            },
        )
        raise
