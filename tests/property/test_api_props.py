"""Property-based tests for API endpoints.

Feature: douyin-video-translator, Property 12: Rate Limiting Enforces Maximum 5 Concurrent Jobs

Validates: Requirements 8.5

For any user, if they have N active jobs (status in [queued, processing]) where N >= 5,
any subsequent translate request SHALL be rejected with HTTP 429. If N < 5, request
SHALL be accepted with HTTP 202.
"""

import uuid
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.api.routes import configure_routes, router
from app.models.job import JobState, JobStatus


# --- Fake dependencies (mirrors tests/unit/test_api.py) ---


class FakeJobStore:
    """In-memory fake job store for property testing."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._ip_jobs: dict[str, list[str]] = {}

    def create_job(self, job_id: str, url: str, client_ip: str, work_dir: str) -> JobState:
        now = datetime.now(timezone.utc)
        job = JobState(
            job_id=job_id,
            url=url,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            work_dir=work_dir,
        )
        self._jobs[job_id] = job
        self._ip_jobs.setdefault(client_ip, []).append(job_id)
        return job

    def get_job(self, job_id: str) -> JobState:
        if job_id not in self._jobs:
            raise KeyError(f"Job not found: {job_id}")
        return self._jobs[job_id]

    def update_job(self, job_id: str, **kwargs: object) -> None:
        job = self.get_job(job_id)
        for field, value in kwargs.items():
            setattr(job, field, value)

    def count_active_jobs(self, client_ip: str) -> int:
        job_ids = self._ip_jobs.get(client_ip, [])
        return sum(
            1
            for jid in job_ids
            if jid in self._jobs
            and self._jobs[jid].status in (JobStatus.QUEUED, JobStatus.PROCESSING)
        )

    def delete_job(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)


class FakeTaskEnqueuer:
    """Fake task enqueuer that records enqueued tasks."""

    def __init__(self) -> None:
        self.enqueued: list[tuple[str, str]] = []

    def enqueue(self, job_id: str, url: str) -> str:
        self.enqueued.append((job_id, url))
        return f"task-{job_id}"


# --- Strategies ---

# Active statuses that count toward rate limit
ACTIVE_STATUSES = st.sampled_from([JobStatus.QUEUED, JobStatus.PROCESSING])

# Number of active jobs at or above the limit (N >= 5)
n_at_or_above_limit = st.integers(min_value=5, max_value=20)

# Number of active jobs below the limit (N < 5)
n_below_limit = st.integers(min_value=0, max_value=4)


# --- Helper ---


def _create_test_app_and_client(fake_store: FakeJobStore) -> TestClient:
    """Create a fresh FastAPI app and test client with the given fake store."""
    app = FastAPI()
    fake_enqueuer = FakeTaskEnqueuer()
    configure_routes(job_store=fake_store, task_enqueuer=fake_enqueuer)
    app.include_router(router)
    return TestClient(app)


def _populate_active_jobs(
    fake_store: FakeJobStore, n: int, client_ip: str, statuses: list[JobStatus]
) -> None:
    """Pre-populate N active jobs for the given client IP."""
    for i in range(n):
        job_id = f"active-job-{uuid.uuid4().hex[:8]}-{i}"
        fake_store.create_job(
            job_id=job_id,
            url="https://www.douyin.com/video/123456",
            client_ip=client_ip,
            work_dir=f"storage/jobs/{job_id}",
        )
        # Assign a status from the provided list (cycle through)
        status = statuses[i % len(statuses)]
        if status != JobStatus.QUEUED:
            fake_store.update_job(job_id, status=status)


# --- Property Tests ---


@pytest.mark.property
class TestRateLimitingProperty:
    """Property 12: Rate Limiting Enforces Maximum 5 Concurrent Jobs.

    **Validates: Requirements 8.5**
    """

    @given(n=n_at_or_above_limit, status=ACTIVE_STATUSES)
    @settings(max_examples=100)
    def test_rejects_when_at_or_above_limit(self, n: int, status: JobStatus) -> None:
        """Feature: douyin-video-translator, Property 12: Rate Limiting Enforces Maximum 5 Concurrent Jobs

        For any N >= 5 active jobs, POST /translate returns HTTP 429.
        """
        fake_store = FakeJobStore()
        client = _create_test_app_and_client(fake_store)

        # Pre-populate N active jobs for the test client IP
        _populate_active_jobs(
            fake_store, n, client_ip="testclient", statuses=[status]
        )

        # Verify rate limit is enforced
        response = client.post(
            "/api/v1/translate",
            json={"url": "https://www.douyin.com/video/999999"},
        )
        assert response.status_code == 429, (
            f"Expected 429 with {n} active jobs (status={status.value}), "
            f"got {response.status_code}"
        )
        # Verify Retry-After header is present
        assert "Retry-After" in response.headers

    @given(n=n_below_limit, status=ACTIVE_STATUSES)
    @settings(max_examples=100)
    def test_accepts_when_below_limit(self, n: int, status: JobStatus) -> None:
        """Feature: douyin-video-translator, Property 12: Rate Limiting Enforces Maximum 5 Concurrent Jobs

        For any N < 5 active jobs, POST /translate returns HTTP 202 (given valid URL).
        """
        fake_store = FakeJobStore()
        client = _create_test_app_and_client(fake_store)

        # Pre-populate N active jobs for the test client IP
        if n > 0:
            _populate_active_jobs(
                fake_store, n, client_ip="testclient", statuses=[status]
            )

        # Verify request is accepted
        response = client.post(
            "/api/v1/translate",
            json={"url": "https://www.douyin.com/video/999999"},
        )
        assert response.status_code == 202, (
            f"Expected 202 with {n} active jobs (status={status.value}), "
            f"got {response.status_code}"
        )
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"


# =============================================================================
# Property 11: API Response Contains Required Fields
# =============================================================================

from app.models.job import PipelineStep, VideoInfo, ErrorDetail
from app.models.schemas import JobStatusResponse


# --- Strategies for Property 11 ---

# Non-empty strings for job_id
_non_empty_job_ids = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip() != "")

# Valid Douyin URLs
_valid_urls = st.builds(
    lambda path: f"https://www.douyin.com/{path}",
    path=st.from_regex(r"[a-zA-Z0-9]{1,20}", fullmatch=True),
)

# Job statuses
_job_statuses = st.sampled_from(list(JobStatus))

# Pipeline steps (or None)
_pipeline_steps = st.one_of(st.none(), st.sampled_from(list(PipelineStep)))

# Datetime strategy
_datetimes = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
    timezones=st.just(timezone.utc),
)

# Progress percentage
_progress_percents = st.integers(min_value=0, max_value=100)

# Work directory paths
_work_dirs = st.builds(
    lambda job_id: f"storage/jobs/{job_id}",
    job_id=st.from_regex(r"[a-f0-9\-]{8,36}", fullmatch=True),
)

# Download URL strategy (non-empty strings)
_download_urls = st.builds(
    lambda path: f"http://localhost:8000/download/{path}",
    path=st.from_regex(r"[a-f0-9\-]{8,36}\.mp4", fullmatch=True),
)

# Video info strategy (optional)
_video_infos = st.one_of(
    st.none(),
    st.builds(
        VideoInfo,
        duration_seconds=st.floats(min_value=1.0, max_value=3600.0, allow_nan=False),
        file_size_bytes=st.integers(min_value=1000, max_value=500_000_000),
        resolution=st.sampled_from(["1080x1920", "720x1280", "480x854"]),
        title=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
    ),
)

# Error detail strategy (optional)
_error_details = st.one_of(
    st.none(),
    st.builds(
        ErrorDetail,
        step=st.sampled_from(list(PipelineStep)),
        message=st.text(min_size=1, max_size=100),
        retryable=st.booleans(),
        retry_count=st.integers(min_value=0, max_value=3),
    ),
)


def _build_job_state(
    job_id: str,
    url: str,
    status: JobStatus,
    current_step,
    progress_percent: int,
    video_info,
    download_url,
    error,
    created_at: datetime,
    updated_at: datetime,
    expires_at,
    work_dir: str,
) -> JobState:
    """Build a JobState instance."""
    return JobState(
        job_id=job_id,
        url=url,
        status=status,
        current_step=current_step,
        progress_percent=progress_percent,
        video_info=video_info,
        download_url=download_url,
        error=error,
        created_at=created_at,
        updated_at=updated_at,
        expires_at=expires_at,
        work_dir=work_dir,
        artifacts={},
    )


# Strategy for any JobState (various statuses)
_any_job_state = st.builds(
    _build_job_state,
    job_id=_non_empty_job_ids,
    url=_valid_urls,
    status=_job_statuses,
    current_step=_pipeline_steps,
    progress_percent=_progress_percents,
    video_info=_video_infos,
    download_url=st.one_of(st.none(), _download_urls),
    error=_error_details,
    created_at=_datetimes,
    updated_at=_datetimes,
    expires_at=st.one_of(st.none(), _datetimes),
    work_dir=_work_dirs,
)

# Strategy for completed JobState (must have download_url and expires_at)
_completed_job_state = st.builds(
    _build_job_state,
    job_id=_non_empty_job_ids,
    url=_valid_urls,
    status=st.just(JobStatus.COMPLETED),
    current_step=st.none(),
    progress_percent=st.just(100),
    video_info=_video_infos,
    download_url=_download_urls,
    error=st.none(),
    created_at=_datetimes,
    updated_at=_datetimes,
    expires_at=_datetimes,
    work_dir=_work_dirs,
)


def _serialize_job_to_response(job: JobState) -> JobStatusResponse:
    """Serialize JobState to JobStatusResponse (mirrors API route logic)."""
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        current_step=job.current_step.value if job.current_step else None,
        progress_percent=job.progress_percent,
        video_info=job.video_info,
        download_url=job.download_url,
        error=job.error,
        created_at=job.created_at,
        expires_at=job.expires_at,
    )


# Valid status enum values
_VALID_STATUS_VALUES = {s.value for s in JobStatus}


@pytest.mark.property
class TestAPIResponseFieldsProperty:
    """Property 11: API Response Contains Required Fields.

    **Validates: Requirements 8.2**
    """

    @given(job=_any_job_state)
    @settings(max_examples=100)
    def test_response_always_contains_required_fields(self, job: JobState):
        """Feature: douyin-video-translator, Property 11: API Response Contains Required Fields

        For any generated JobState, serializing to JobStatusResponse always has
        job_id, status, created_at.
        """
        response = _serialize_job_to_response(job)
        response_dict = response.model_dump()

        # job_id must be a non-empty string
        assert "job_id" in response_dict
        assert isinstance(response_dict["job_id"], str)
        assert len(response_dict["job_id"]) > 0

        # status must be present and a valid enum value
        assert "status" in response_dict
        assert isinstance(response_dict["status"], str)
        assert response_dict["status"] in _VALID_STATUS_VALUES

        # created_at must be a valid datetime
        assert "created_at" in response_dict
        assert isinstance(response_dict["created_at"], datetime)

    @given(job=_completed_job_state)
    @settings(max_examples=100)
    def test_completed_response_contains_download_url_and_expires_at(self, job: JobState):
        """Feature: douyin-video-translator, Property 11: API Response Contains Required Fields

        For any completed JobState, response additionally has download_url and expires_at.
        """
        response = _serialize_job_to_response(job)
        response_dict = response.model_dump()

        # status must be "completed"
        assert response_dict["status"] == "completed"

        # download_url must be a non-empty string
        assert "download_url" in response_dict
        assert isinstance(response_dict["download_url"], str)
        assert len(response_dict["download_url"]) > 0

        # expires_at must be a valid datetime
        assert "expires_at" in response_dict
        assert isinstance(response_dict["expires_at"], datetime)

    @given(job=_any_job_state)
    @settings(max_examples=100)
    def test_status_is_always_valid_enum_value(self, job: JobState):
        """Feature: douyin-video-translator, Property 11: API Response Contains Required Fields

        Status values are always valid enum values.
        """
        response = _serialize_job_to_response(job)
        response_dict = response.model_dump()

        assert response_dict["status"] in _VALID_STATUS_VALUES
