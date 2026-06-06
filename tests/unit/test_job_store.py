"""Unit tests for the Redis-backed JobStore service."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import redis

from app.models.job import JobState, JobStatus
from app.services.job_store import JobStore, JobStoreError


@pytest.fixture
def mock_redis():
    """Provide a mocked Redis client."""
    with patch("app.services.job_store.redis.Redis.from_url") as mock_from_url:
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client
        yield mock_client


@pytest.fixture
def job_store(mock_redis):
    """Provide a JobStore backed by a mock Redis."""
    store = JobStore(redis_url="redis://localhost:6379/0")
    return store


@pytest.fixture
def sample_job() -> JobState:
    """Provide a sample JobState for testing."""
    now = datetime.now(timezone.utc)
    return JobState(
        job_id="test-job-123",
        url="https://www.douyin.com/video/123",
        status=JobStatus.QUEUED,
        created_at=now,
        updated_at=now,
        work_dir="/tmp/storage/jobs/test-job-123",
    )


class TestCreateJob:
    """Tests for JobStore.create_job."""

    def test_creates_job_with_correct_fields(self, job_store, mock_redis):
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe

        job = job_store.create_job(
            job_id="job-001",
            url="https://www.douyin.com/video/999",
            client_ip="192.168.1.1",
            work_dir="/tmp/storage/jobs/job-001",
        )

        assert job.job_id == "job-001"
        assert job.url == "https://www.douyin.com/video/999"
        assert job.status == JobStatus.QUEUED
        assert job.work_dir == "/tmp/storage/jobs/job-001"
        assert job.progress_percent == 0
        assert job.current_step is None

    def test_stores_job_in_redis(self, job_store, mock_redis):
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe

        job_store.create_job(
            job_id="job-001",
            url="https://www.douyin.com/video/999",
            client_ip="192.168.1.1",
            work_dir="/tmp/storage/jobs/job-001",
        )

        # Pipeline should set the job key and add to active set
        mock_pipe.set.assert_called_once()
        key_arg = mock_pipe.set.call_args[0][0]
        assert key_arg == "job:job-001"

        mock_pipe.sadd.assert_called_once_with("jobs:active:192.168.1.1", "job-001")
        mock_pipe.execute.assert_called_once()

    def test_raises_job_store_error_on_redis_failure(self, job_store, mock_redis):
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        mock_pipe.execute.side_effect = redis.RedisError("Connection refused")

        with pytest.raises(JobStoreError) as exc_info:
            job_store.create_job(
                job_id="job-001",
                url="https://www.douyin.com/video/999",
                client_ip="192.168.1.1",
                work_dir="/tmp/storage/jobs/job-001",
            )

        assert "Failed to create job" in str(exc_info.value)
        assert exc_info.value.original is not None


class TestGetJob:
    """Tests for JobStore.get_job."""

    def test_returns_job_state(self, job_store, mock_redis, sample_job):
        mock_redis.get.return_value = sample_job.model_dump_json()

        result = job_store.get_job("test-job-123")

        assert result.job_id == "test-job-123"
        assert result.status == JobStatus.QUEUED
        assert result.url == "https://www.douyin.com/video/123"
        mock_redis.get.assert_called_once_with("job:test-job-123")

    def test_raises_key_error_when_not_found(self, job_store, mock_redis):
        mock_redis.get.return_value = None

        with pytest.raises(KeyError, match="Job not found: nonexistent"):
            job_store.get_job("nonexistent")

    def test_raises_job_store_error_on_redis_failure(self, job_store, mock_redis):
        mock_redis.get.side_effect = redis.RedisError("Timeout")

        with pytest.raises(JobStoreError) as exc_info:
            job_store.get_job("job-001")

        assert "Failed to read job" in str(exc_info.value)


class TestUpdateJob:
    """Tests for JobStore.update_job."""

    def test_updates_single_field(self, job_store, mock_redis, sample_job):
        mock_redis.get.return_value = sample_job.model_dump_json()

        job_store.update_job("test-job-123", status=JobStatus.PROCESSING)

        # Should write back to Redis
        mock_redis.set.assert_called_once()
        key_arg = mock_redis.set.call_args[0][0]
        assert key_arg == "job:test-job-123"

        # Verify the written data has updated status
        written_json = mock_redis.set.call_args[0][1]
        written_job = JobState.model_validate_json(written_json)
        assert written_job.status == JobStatus.PROCESSING

    def test_updates_multiple_fields(self, job_store, mock_redis, sample_job):
        mock_redis.get.return_value = sample_job.model_dump_json()

        from app.models.job import PipelineStep

        job_store.update_job(
            "test-job-123",
            status=JobStatus.PROCESSING,
            current_step=PipelineStep.DOWNLOADING,
            progress_percent=15,
        )

        written_json = mock_redis.set.call_args[0][1]
        written_job = JobState.model_validate_json(written_json)
        assert written_job.status == JobStatus.PROCESSING
        assert written_job.current_step == PipelineStep.DOWNLOADING
        assert written_job.progress_percent == 15

    def test_bumps_updated_at(self, job_store, mock_redis, sample_job):
        original_updated_at = sample_job.updated_at
        mock_redis.get.return_value = sample_job.model_dump_json()

        job_store.update_job("test-job-123", progress_percent=50)

        written_json = mock_redis.set.call_args[0][1]
        written_job = JobState.model_validate_json(written_json)
        assert written_job.updated_at >= original_updated_at

    def test_raises_key_error_when_not_found(self, job_store, mock_redis):
        mock_redis.get.return_value = None

        with pytest.raises(KeyError):
            job_store.update_job("nonexistent", status=JobStatus.FAILED)

    def test_raises_job_store_error_on_write_failure(self, job_store, mock_redis, sample_job):
        mock_redis.get.return_value = sample_job.model_dump_json()
        mock_redis.set.side_effect = redis.RedisError("Disk full")

        with pytest.raises(JobStoreError) as exc_info:
            job_store.update_job("test-job-123", status=JobStatus.FAILED)

        assert "Failed to update job" in str(exc_info.value)


class TestCancelJob:
    """Tests for JobStore.cancel_job."""

    def test_sets_status_to_cancelled(self, job_store, mock_redis, sample_job):
        mock_redis.get.return_value = sample_job.model_dump_json()

        job_store.cancel_job("test-job-123")

        written_json = mock_redis.set.call_args[0][1]
        written_job = JobState.model_validate_json(written_json)
        assert written_job.status == JobStatus.CANCELLED

    def test_raises_key_error_when_not_found(self, job_store, mock_redis):
        mock_redis.get.return_value = None

        with pytest.raises(KeyError):
            job_store.cancel_job("nonexistent")


class TestCountActiveJobs:
    """Tests for JobStore.count_active_jobs."""

    def test_returns_zero_when_no_jobs(self, job_store, mock_redis):
        mock_redis.smembers.return_value = set()

        count = job_store.count_active_jobs("192.168.1.1")

        assert count == 0

    def test_counts_queued_and_processing_jobs(self, job_store, mock_redis):
        mock_redis.smembers.return_value = {"job-1", "job-2", "job-3"}

        now = datetime.now(timezone.utc)
        jobs = {
            "job-1": JobState(
                job_id="job-1", url="u", status=JobStatus.QUEUED,
                created_at=now, updated_at=now, work_dir="/tmp/1",
            ),
            "job-2": JobState(
                job_id="job-2", url="u", status=JobStatus.PROCESSING,
                created_at=now, updated_at=now, work_dir="/tmp/2",
            ),
            "job-3": JobState(
                job_id="job-3", url="u", status=JobStatus.COMPLETED,
                created_at=now, updated_at=now, work_dir="/tmp/3",
            ),
        }

        def get_side_effect(key):
            job_id = key.replace("job:", "")
            if job_id in jobs:
                return jobs[job_id].model_dump_json()
            return None

        mock_redis.get.side_effect = get_side_effect

        count = job_store.count_active_jobs("192.168.1.1")

        assert count == 2

    def test_removes_stale_jobs_from_active_set(self, job_store, mock_redis):
        mock_redis.smembers.return_value = {"job-1", "job-2"}

        now = datetime.now(timezone.utc)
        # job-1 is completed, job-2 doesn't exist
        job_1 = JobState(
            job_id="job-1", url="u", status=JobStatus.COMPLETED,
            created_at=now, updated_at=now, work_dir="/tmp/1",
        )

        def get_side_effect(key):
            if key == "job:job-1":
                return job_1.model_dump_json()
            return None  # job-2 not found

        mock_redis.get.side_effect = get_side_effect

        count = job_store.count_active_jobs("192.168.1.1")

        assert count == 0
        # Should remove both stale entries
        mock_redis.srem.assert_called_once()
        removed_ids = set(mock_redis.srem.call_args[0][1:])
        assert removed_ids == {"job-1", "job-2"}

    def test_raises_job_store_error_on_redis_failure(self, job_store, mock_redis):
        mock_redis.smembers.side_effect = redis.RedisError("Connection lost")

        with pytest.raises(JobStoreError) as exc_info:
            job_store.count_active_jobs("192.168.1.1")

        assert "Failed to read active jobs" in str(exc_info.value)


class TestDeleteJob:
    """Tests for JobStore.delete_job."""

    def test_deletes_job_key(self, job_store, mock_redis):
        job_store.delete_job("test-job-123")

        mock_redis.delete.assert_called_once_with("job:test-job-123")

    def test_raises_job_store_error_on_redis_failure(self, job_store, mock_redis):
        mock_redis.delete.side_effect = redis.RedisError("Connection refused")

        with pytest.raises(JobStoreError):
            job_store.delete_job("test-job-123")


class TestJobSerialization:
    """Tests for Pydantic serialization/deserialization round-trip."""

    def test_round_trip_preserves_all_fields(self, job_store, mock_redis):
        from app.models.job import PipelineStep

        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe

        # Create a job
        job = job_store.create_job(
            job_id="serial-test",
            url="https://v.douyin.com/abc123",
            client_ip="10.0.0.1",
            work_dir="/tmp/storage/jobs/serial-test",
        )

        # Simulate reading back the stored JSON
        stored_json = mock_pipe.set.call_args[0][1]
        mock_redis.get.return_value = stored_json

        # Read it back
        retrieved = job_store.get_job("serial-test")

        assert retrieved.job_id == job.job_id
        assert retrieved.url == job.url
        assert retrieved.status == job.status
        assert retrieved.work_dir == job.work_dir
        assert retrieved.created_at == job.created_at
        assert retrieved.updated_at == job.updated_at
