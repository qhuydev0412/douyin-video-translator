"""FastAPI dependencies for the API layer."""

import logging
from typing import Protocol

from fastapi import HTTPException, Request
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from app.core.config import settings

logger = logging.getLogger(__name__)


class JobStoreProtocol(Protocol):
    """Protocol for counting active jobs (used by rate limiter)."""

    def count_active_jobs(self, client_ip: str) -> int: ...


class RateLimiter:
    """Rate limiting dependency that enforces max concurrent jobs per IP.

    Checks the number of active (queued + processing) jobs for the client's
    IP address. If the count exceeds the configured limit, returns HTTP 429.
    """

    def __init__(self, job_store: JobStoreProtocol) -> None:
        self._job_store = job_store
        self._max_concurrent = settings.MAX_CONCURRENT_JOBS

    def check(self, request: Request) -> None:
        """Check if the client IP has exceeded the concurrent job limit.

        Args:
            request: The incoming FastAPI request.

        Raises:
            HTTPException: 429 with Retry-After header if limit exceeded.
        """
        client_ip = self._get_client_ip(request)
        active_count = self._job_store.count_active_jobs(client_ip)

        if active_count >= self._max_concurrent:
            logger.warning(
                "Rate limit exceeded for IP %s: %d active jobs",
                client_ip,
                active_count,
            )
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": "Đã vượt quá giới hạn xử lý đồng thời (tối đa 5 jobs)",
                    "retry_after": 60,
                },
                headers={"Retry-After": "60"},
            )

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extract client IP from request, considering proxy headers."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
