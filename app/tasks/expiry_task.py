"""Periodic Celery task for expiring stale awaiting_confirmation jobs."""

import logging

from app.core.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="check_checkpoint_expiry")
def check_checkpoint_expiry_task() -> dict[str, int | list[str]]:
    """Periodic task to expire stale awaiting_confirmation jobs.

    Instantiates a CheckpointManager and calls check_expired_jobs() to find
    and expire any jobs that have been in awaiting_confirmation status for
    longer than the configured expiration period (24 hours).

    This task runs every 30 seconds via Celery Beat, ensuring jobs are expired
    well within the 60-second requirement window.

    Returns:
        Dict with expired_count and expired_job_ids.
    """
    from app.services.checkpoint_manager import CheckpointManager  # noqa: WPS433
    from app.services.job_store import JobStore  # noqa: WPS433

    job_store = JobStore()
    checkpoint_manager = CheckpointManager(job_store)

    expired_ids = checkpoint_manager.check_expired_jobs()

    logger.info(
        "Checkpoint expiry check complete: %d jobs expired", len(expired_ids)
    )

    return {"expired_count": len(expired_ids), "expired_job_ids": expired_ids}
