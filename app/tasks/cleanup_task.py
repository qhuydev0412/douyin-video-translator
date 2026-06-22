"""Periodic Celery task for cleaning up expired job files and records."""

import logging
import shutil
import time
from pathlib import Path

from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(name="cleanup_expired_jobs")
def cleanup_expired_jobs() -> dict[str, int]:
    """Remove job directories and Redis records older than FILE_EXPIRY_HOURS."""
    from app.core.config import settings
    from app.services.job_store import JobStore

    storage_path = Path(settings.STORAGE_PATH)
    expiry_seconds = settings.FILE_EXPIRY_HOURS * 3600
    now = time.time()

    cleaned_dirs = 0
    cleaned_records = 0

    if not storage_path.exists():
        return {"cleaned_dirs": 0, "cleaned_records": 0}

    job_store = JobStore()

    for job_dir in storage_path.iterdir():
        if not job_dir.is_dir():
            continue
        if job_dir.name.startswith("."):
            continue

        try:
            mtime = job_dir.stat().st_mtime
        except OSError:
            continue

        age_seconds = now - mtime
        if age_seconds < expiry_seconds:
            continue

        job_id = job_dir.name
        logger.info("Cleaning expired job %s (age: %.1f hours)", job_id, age_seconds / 3600)

        try:
            shutil.rmtree(job_dir)
            cleaned_dirs += 1
        except OSError as exc:
            logger.error("Failed to remove directory %s: %s", job_dir, exc)
            continue

        try:
            job_store.delete_job(job_id)
            cleaned_records += 1
        except Exception as exc:
            logger.warning("Failed to delete Redis record for job %s: %s", job_id, exc)

    logger.info("Cleanup complete: removed %d directories, %d Redis records", cleaned_dirs, cleaned_records)
    return {"cleaned_dirs": cleaned_dirs, "cleaned_records": cleaned_records}
