"""Unit tests for the cleanup_expired_jobs periodic task."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.tasks.cleanup_task import cleanup_expired_jobs


@pytest.fixture
def tmp_storage(tmp_path):
    """Create a temporary storage directory structure."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    return jobs_dir


@pytest.fixture
def mock_job_store():
    """Create a mock JobStore."""
    with patch("app.services.job_store.JobStore") as mock_cls:
        store = MagicMock()
        mock_cls.return_value = store
        yield store


class TestCleanupExpiredJobs:
    """Tests for cleanup_expired_jobs task."""

    def test_removes_expired_directories(self, tmp_storage, mock_job_store):
        """Expired job directories are removed."""
        # Create an expired job directory (modify time > 24h ago)
        expired_dir = tmp_storage / "job-expired-1"
        expired_dir.mkdir()
        (expired_dir / "video.mp4").write_text("data")

        # Set modification time to 25 hours ago
        old_time = time.time() - (25 * 3600)
        import os

        os.utime(expired_dir, (old_time, old_time))

        with patch("app.tasks.cleanup_task.settings") as mock_settings:
            mock_settings.STORAGE_PATH = str(tmp_storage)
            mock_settings.FILE_EXPIRY_HOURS = 24

            result = cleanup_expired_jobs()

        assert result["cleaned_dirs"] == 1
        assert result["cleaned_records"] == 1
        assert not expired_dir.exists()

    def test_keeps_fresh_directories(self, tmp_storage, mock_job_store):
        """Non-expired job directories are preserved."""
        # Create a fresh job directory
        fresh_dir = tmp_storage / "job-fresh-1"
        fresh_dir.mkdir()
        (fresh_dir / "video.mp4").write_text("data")

        with patch("app.tasks.cleanup_task.settings") as mock_settings:
            mock_settings.STORAGE_PATH = str(tmp_storage)
            mock_settings.FILE_EXPIRY_HOURS = 24

            result = cleanup_expired_jobs()

        assert result["cleaned_dirs"] == 0
        assert result["cleaned_records"] == 0
        assert fresh_dir.exists()

    def test_skips_dotfiles(self, tmp_storage, mock_job_store):
        """.gitkeep and other dotfiles are skipped."""
        gitkeep = tmp_storage / ".gitkeep"
        gitkeep.write_text("")

        # Set modification time to long ago
        old_time = time.time() - (100 * 3600)
        import os

        os.utime(gitkeep, (old_time, old_time))

        with patch("app.tasks.cleanup_task.settings") as mock_settings:
            mock_settings.STORAGE_PATH = str(tmp_storage)
            mock_settings.FILE_EXPIRY_HOURS = 24

            result = cleanup_expired_jobs()

        assert result["cleaned_dirs"] == 0
        assert gitkeep.exists()

    def test_nonexistent_storage_path(self, tmp_path, mock_job_store):
        """Returns zeros when storage path doesn't exist."""
        with patch("app.tasks.cleanup_task.settings") as mock_settings:
            mock_settings.STORAGE_PATH = str(tmp_path / "nonexistent")
            mock_settings.FILE_EXPIRY_HOURS = 24

            result = cleanup_expired_jobs()

        assert result == {"cleaned_dirs": 0, "cleaned_records": 0}

    def test_handles_redis_delete_failure(self, tmp_storage, mock_job_store):
        """Continues cleanup even if Redis delete fails."""
        expired_dir = tmp_storage / "job-redis-fail"
        expired_dir.mkdir()

        old_time = time.time() - (25 * 3600)
        import os

        os.utime(expired_dir, (old_time, old_time))

        mock_job_store.delete_job.side_effect = Exception("Redis connection refused")

        with patch("app.tasks.cleanup_task.settings") as mock_settings:
            mock_settings.STORAGE_PATH = str(tmp_storage)
            mock_settings.FILE_EXPIRY_HOURS = 24

            result = cleanup_expired_jobs()

        # Directory was removed but Redis record wasn't
        assert result["cleaned_dirs"] == 1
        assert result["cleaned_records"] == 0
        assert not expired_dir.exists()
