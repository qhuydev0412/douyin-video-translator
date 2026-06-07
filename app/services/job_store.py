"""Redis-backed job state store.

Provides persistence for job state using Redis as the backend.
Implements JobStoreProtocol from app.services.pipeline.
"""

import logging
from datetime import datetime, timezone
from typing import Any

import redis

from app.core.config import settings
from app.models.job import JobState, JobStatus

logger = logging.getLogger(__name__)

# Redis key patterns
JOB_KEY_PREFIX = "job:"
ACTIVE_JOBS_KEY_PREFIX = "jobs:active:"

# Statuses considered "active" for rate limiting
ACTIVE_STATUSES = {JobStatus.QUEUED, JobStatus.PROCESSING}


class JobStoreError(Exception):
    """Raised when a Redis operation fails.

    Wraps underlying redis.RedisError to provide a service-level exception
    that API handlers can map to HTTP 503.
    """

    def __init__(self, message: str, original: Exception | None = None):
        super().__init__(message)
        self.original = original


class JobStore:
    """Redis-backed store for job state persistence.

    Implements JobStoreProtocol from app.services.pipeline.

    Key patterns:
        - ``job:{job_id}`` — serialized JobState JSON
        - ``jobs:active:{client_ip}`` — Redis set of active job IDs per IP
    """

    def __init__(self, redis_url: str | None = None) -> None:
        url = redis_url or settings.REDIS_URL
        try:
            self._redis = redis.Redis.from_url(url, decode_responses=True)
        except redis.RedisError as exc:
            raise JobStoreError(f"Failed to connect to Redis: {exc}", original=exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_job(
        self,
        job_id: str,
        url: str,
        client_ip: str,
        work_dir: str,
    ) -> JobState:
        """Create and store a new job.

        Args:
            job_id: Unique job identifier (UUID).
            url: Original Douyin URL.
            client_ip: Requesting client's IP address (for rate limiting).
            work_dir: Working directory path for this job.

        Returns:
            The newly created JobState.

        Raises:
            JobStoreError: If the Redis write fails.
        """
        now = datetime.now(timezone.utc)
        job = JobState(
            job_id=job_id,
            url=url,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            work_dir=work_dir,
        )

        try:
            pipe = self._redis.pipeline()
            pipe.set(self._job_key(job_id), job.model_dump_json())
            pipe.sadd(self._active_key(client_ip), job_id)
            pipe.execute()
        except redis.RedisError as exc:
            raise JobStoreError(
                f"Failed to create job {job_id}: {exc}", original=exc
            )

        logger.info("Created job %s for client %s", job_id, client_ip)
        return job

    def get_job(self, job_id: str) -> JobState:
        """Retrieve job state from Redis.

        Args:
            job_id: Unique job identifier.

        Returns:
            JobState for the given job.

        Raises:
            KeyError: If job_id does not exist.
            JobStoreError: If the Redis read fails.
        """
        try:
            data = self._redis.get(self._job_key(job_id))
        except redis.RedisError as exc:
            raise JobStoreError(
                f"Failed to read job {job_id}: {exc}", original=exc
            )

        if data is None:
            raise KeyError(f"Job not found: {job_id}")

        return JobState.model_validate_json(data)

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        """Update fields on an existing job record.

        Retrieves the current state, applies the field updates, bumps
        ``updated_at``, and writes back to Redis.

        Args:
            job_id: Unique job identifier.
            **kwargs: Fields to update on the JobState.

        Raises:
            KeyError: If job_id does not exist.
            JobStoreError: If the Redis operation fails.
        """
        job = self.get_job(job_id)

        for field_name, value in kwargs.items():
            setattr(job, field_name, value)

        job.updated_at = datetime.now(timezone.utc)

        try:
            self._redis.set(self._job_key(job_id), job.model_dump_json())
        except redis.RedisError as exc:
            raise JobStoreError(
                f"Failed to update job {job_id}: {exc}", original=exc
            )

    def cancel_job(self, job_id: str) -> None:
        """Set job status to CANCELLED.

        Args:
            job_id: Unique job identifier.

        Raises:
            KeyError: If job_id does not exist.
            JobStoreError: If the Redis operation fails.
        """
        self.update_job(job_id, status=JobStatus.CANCELLED)
        logger.info("Cancelled job %s", job_id)

    def count_active_jobs(self, client_ip: str) -> int:
        """Count active (QUEUED/PROCESSING) jobs for a given client IP.

        Iterates the IP's active set and verifies each job is still
        in an active status. Removes stale entries (completed/failed/cancelled)
        from the set as a side effect.

        Args:
            client_ip: Client IP address.

        Returns:
            Number of currently active jobs for the IP.

        Raises:
            JobStoreError: If the Redis operation fails.
        """
        try:
            job_ids = self._redis.smembers(self._active_key(client_ip))
        except redis.RedisError as exc:
            raise JobStoreError(
                f"Failed to read active jobs for {client_ip}: {exc}", original=exc
            )

        if not job_ids:
            return 0

        active_count = 0
        stale_ids: list[str] = []

        for jid in job_ids:
            try:
                job = self.get_job(jid)
                if job.status in ACTIVE_STATUSES:
                    active_count += 1
                else:
                    stale_ids.append(jid)
            except KeyError:
                # Job record no longer exists
                stale_ids.append(jid)

        # Clean up stale entries
        if stale_ids:
            try:
                self._redis.srem(self._active_key(client_ip), *stale_ids)
            except redis.RedisError:
                # Non-critical; log and continue
                logger.warning(
                    "Failed to remove stale job IDs from active set for %s",
                    client_ip,
                )

        return active_count

    def delete_job(self, job_id: str) -> None:
        """Delete a job record from Redis.

        Args:
            job_id: Unique job identifier.

        Raises:
            JobStoreError: If the Redis operation fails.
        """
        try:
            self._redis.delete(self._job_key(job_id))
        except redis.RedisError as exc:
            raise JobStoreError(
                f"Failed to delete job {job_id}: {exc}", original=exc
            )

        logger.debug("Deleted job record %s", job_id)

    def list_awaiting_confirmation_job_ids(self) -> list[str]:
        """List all job IDs with AWAITING_CONFIRMATION status.

        Uses Redis SCAN to iterate all job keys and filters by status.
        This is acceptable for periodic task usage (runs every 30s, not
        performance-critical).

        Returns:
            List of job IDs currently in AWAITING_CONFIRMATION status.

        Raises:
            JobStoreError: If the Redis operation fails.
        """
        awaiting_ids: list[str] = []
        cursor = 0
        pattern = f"{JOB_KEY_PREFIX}*"

        try:
            while True:
                cursor, keys = self._redis.scan(
                    cursor=cursor, match=pattern, count=100
                )
                for key in keys:
                    raw = self._redis.get(key)
                    if raw is None:
                        continue
                    try:
                        job = JobState.model_validate_json(raw)
                        if job.status == JobStatus.AWAITING_CONFIRMATION:
                            awaiting_ids.append(job.job_id)
                    except Exception:
                        # Skip malformed entries
                        logger.warning("Failed to parse job data for key %s", key)
                if cursor == 0:
                    break
        except redis.RedisError as exc:
            raise JobStoreError(
                f"Failed to scan awaiting confirmation jobs: {exc}", original=exc
            )

        return awaiting_ids

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _job_key(job_id: str) -> str:
        """Build Redis key for a job record."""
        return f"{JOB_KEY_PREFIX}{job_id}"

    @staticmethod
    def _active_key(client_ip: str) -> str:
        """Build Redis key for a client's active job set."""
        return f"{ACTIVE_JOBS_KEY_PREFIX}{client_ip}"
