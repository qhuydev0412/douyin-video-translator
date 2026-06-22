"""Periodic Celery task for expiring stale awaiting_confirmation jobs."""

import logging

from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(name="check_checkpoint_expiry")
def check_checkpoint_expiry_task() -> dict[str, int | list[str]]:
    """Periodic task to expire stale awaiting_confirmation jobs."""
    from app.services.checkpoint_manager import CheckpointManager
    from app.services.job_store import JobStore

    job_store = JobStore()
    checkpoint_manager = CheckpointManager(job_store)

    expired_ids = checkpoint_manager.check_expired_jobs()

    logger.info("Checkpoint expiry check complete: %d jobs expired", len(expired_ids))
    return {"expired_count": len(expired_ids), "expired_job_ids": expired_ids}
