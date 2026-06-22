"""Celery task for video translation pipeline execution."""

import asyncio
import logging

from celery import shared_task

logger = logging.getLogger(__name__)


def _create_pipeline(job_store):
    """Create a TranslationPipeline with all service dependencies."""
    from app.services.audio_extractor import AudioExtractor
    from app.services.checkpoint_manager import CheckpointManager
    from app.services.downloader import VideoDownloader
    from app.services.gender_detector import GenderDetector
    from app.services.pipeline import TranslationPipeline
    from app.services.speech_recognizer import SpeechRecognizer
    from app.services.subtitle_extractor import SubtitleExtractor
    from app.services.translator import Translator
    from app.services.video_composer import VideoComposer
    from app.services.vocal_isolator import VocalIsolator
    from app.services.voice_preview import VoicePreviewGenerator
    from app.services.voice_synthesizer import VoiceSynthesizer

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
        gender_detector=GenderDetector(),
        subtitle_extractor=SubtitleExtractor(),
    )


@shared_task(bind=True, name="translate_video")
def translate_video_task(self, job_id: str, url: str) -> dict[str, str]:
    """Execute the full video translation pipeline as a Celery task."""
    from app.models.job import JobStatus, PipelineStep
    from app.services.job_store import JobStore
    from app.services.pipeline import CancellationError, PipelineError

    logger.info("Starting translation task for job %s, url=%s", job_id, url)

    job_store = JobStore()
    pipeline = _create_pipeline(job_store)

    try:
        result = asyncio.run(pipeline.execute(job_id, url))

        if result is None:
            logger.info("Pipeline paused at checkpoint for job %s", job_id)
            return {"job_id": job_id, "status": "paused"}

        logger.info("Translation completed for job %s: %s", job_id, result.output_path)
        return {"job_id": job_id, "output_path": str(result.output_path)}

    except PipelineError as exc:
        logger.error(
            "Pipeline error for job %s at step %s: %s",
            job_id, exc.step.value, exc.message,
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
        logger.exception("Unexpected error for job %s: %s", job_id, str(exc))
        job_store.update_job(
            job_id,
            status=JobStatus.FAILED,
            error={
                "step": PipelineStep.DOWNLOADING.value,
                "message": f"Unexpected error: {exc}",
                "retryable": False,
                "retry_count": 0,
            },
        )
        raise


def revoke_task(task_id: str, terminate: bool = True) -> None:
    """Revoke a running Celery task."""
    from app.core.celery_app import celery_app

    logger.info("Revoking task %s (terminate=%s)", task_id, terminate)
    celery_app.control.revoke(task_id, terminate=terminate, signal="SIGTERM")
