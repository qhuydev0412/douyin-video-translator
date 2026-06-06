"""Periodic Celery task for cleaning up expired job files and records."""

import logging
import shutil
import time
from pathlib import Path

from app.core.celery_app import celery_app
from app.core.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(name="cleanup_expired_jobs")
def cleanup_expired_jobs() -> dict[str, int]:
    """Remove job directories and Redis records older than FILE_EXPIRY_HOURS.

    Scans the storage/jobs/ directory for expired job folders based on
    their last modification time. Also cleans up corresponding Redis
    job records.

    This task runs periodically via Celery Beat (every hour by default).

    Returns:
        Dict with counts of cleaned directories and Redis records.
    """
    from app.services.job_store import JobStore  # noqa: WPS433 - lazy import

    storage_path = Path(settings.STORAGE_PATH)
    expiry_seconds = settings.FILE_EXPIRY_HOURS * 3600
    now = time.time()

    cleaned_dirs = 0
    cleaned_records = 0

    if not storage_path.exists():
        logger.debug("Storage path %s does not exist, nothing to clean", storage_path)
        return {"cleaned_dirs": 0, "cleaned_records": 0}

    job_store = JobStore()

    for job_dir in storage_path.iterdir():
        if not job_dir.is_dir():
            continue

        # Skip non-job entries (e.g., .gitkeep)
        if job_dir.name.startswith("."):
            continue

        # Check directory age by modification time
        try:
            mtime = job_dir.stat().st_mtime
        except OSError:
            logger.warning("Cannot stat directory %s, skipping", job_dir)
            continue

        age_seconds = now - mtime
        if age_seconds < expiry_seconds:
            continue

        job_id = job_dir.name
        logger.info(
            "Cleaning expired job %s (age: %.1f hours)",
            job_id,
            age_seconds / 3600,
        )

        # Remove the job directory
        try:
            shutil.rmtree(job_dir)
            cleaned_dirs += 1
        except OSError as exc:
            logger.error("Failed to remove directory %s: %s", job_dir, exc)
            continue

        # Remove Redis record
        try:
            job_store.delete_job(job_id)
            cleaned_records += 1
        except Exception as exc:
            logger.warning("Failed to delete Redis record for job %s: %s", job_id, exc)

    logger.info(
        "Cleanup complete: removed %d directories, %d Redis records",
        cleaned_dirs,
        cleaned_records,
    )
    return {"cleaned_dirs": cleaned_dirs, "cleaned_records": cleaned_records}
