"""Unit tests for the VideoDownloader service."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yt_dlp

from app.services.downloader import (
    DownloadError,
    InvalidURLError,
    NetworkError,
    VideoDownloader,
    VideoNotFoundError,
)


class TestValidateUrl:
    """Tests for URL validation logic."""

    def setup_method(self):
        self.downloader = VideoDownloader()

    def test_valid_douyin_url(self):
        assert self.downloader.validate_url("https://www.douyin.com/video/123456") is True

    def test_valid_v_subdomain(self):
        assert self.downloader.validate_url("https://v.douyin.com/abc123") is True

    def test_valid_bare_domain(self):
        assert self.downloader.validate_url("https://douyin.com/video/789") is True

    def test_valid_http_scheme(self):
        assert self.downloader.validate_url("http://douyin.com/video/789") is True

    def test_invalid_tiktok_url(self):
        assert self.downloader.validate_url("https://www.tiktok.com/video/123") is False

    def test_invalid_random_url(self):
        assert self.downloader.validate_url("https://example.com/video/123") is False

    def test_invalid_not_a_url(self):
        assert self.downloader.validate_url("not a url at all") is False

    def test_invalid_empty_string(self):
        assert self.downloader.validate_url("") is False

    def test_invalid_douyin_in_path(self):
        """URL with douyin.com in path but not in hostname should be rejected."""
        assert self.downloader.validate_url("https://evil.com/douyin.com/video") is False

    def test_invalid_subdomain_of_fake_domain(self):
        """douyin.com.evil.com should NOT pass validation."""
        assert self.downloader.validate_url("https://douyin.com.evil.com/test") is False

    def test_invalid_notdouyin(self):
        """notdouyin.com should NOT pass validation."""
        assert self.downloader.validate_url("https://notdouyin.com/video") is False


class TestGetVideoInfo:
    """Tests for video info extraction."""

    def setup_method(self):
        self.downloader = VideoDownloader(max_retries=1, backoff_base=0)

    def test_invalid_url_raises_error(self):
        with pytest.raises(InvalidURLError) as exc_info:
            self.downloader.get_video_info("https://invalid.com/video/123")
        assert exc_info.value.message == "URL không hợp lệ, vui lòng cung cấp link Douyin"
        assert exc_info.value.retryable is False

    @patch("app.services.downloader.yt_dlp.YoutubeDL")
    def test_successful_info_extraction(self, mock_ydl_class):
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "duration": 60.5,
            "filesize": 5000000,
            "width": 1080,
            "height": 1920,
            "title": "Test Video",
        }

        info = self.downloader.get_video_info("https://www.douyin.com/video/123")

        assert info.duration_seconds == 60.5
        assert info.file_size_bytes == 5000000
        assert info.resolution == "1080x1920"
        assert info.title == "Test Video"

    @patch("app.services.downloader.yt_dlp.YoutubeDL")
    def test_video_not_found_404(self, mock_ydl_class):
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.side_effect = yt_dlp.utils.DownloadError(
            "HTTP Error 404: Not Found"
        )

        with pytest.raises(VideoNotFoundError) as exc_info:
            self.downloader.get_video_info("https://www.douyin.com/video/deleted")
        assert exc_info.value.message == "Video không tồn tại hoặc đã bị xóa"
        assert exc_info.value.retryable is False

    @patch("app.services.downloader.yt_dlp.YoutubeDL")
    def test_video_removed_error(self, mock_ydl_class):
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.side_effect = yt_dlp.utils.DownloadError(
            "Video has been removed by the author"
        )

        with pytest.raises(VideoNotFoundError):
            self.downloader.get_video_info("https://www.douyin.com/video/removed")


class TestDownload:
    """Tests for video download logic."""

    def setup_method(self):
        self.downloader = VideoDownloader(max_retries=1, backoff_base=0)

    def test_invalid_url_raises_error(self, tmp_path):
        with pytest.raises(InvalidURLError):
            self.downloader.download("https://invalid.com/video", tmp_path)

    @patch("app.services.downloader.yt_dlp.YoutubeDL")
    def test_successful_download(self, mock_ydl_class, tmp_path):
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {
            "duration": 30.0,
            "filesize": 2000000,
            "width": 720,
            "height": 1280,
            "title": "Downloaded Video",
        }

        # Create the expected output file
        output_file = tmp_path / "original.mp4"
        output_file.write_bytes(b"fake video content")

        result = self.downloader.download("https://www.douyin.com/video/123", tmp_path)

        assert result.video_path == output_file
        assert result.video_info.duration_seconds == 30.0
        assert result.video_info.resolution == "720x1280"
        assert result.video_info.title == "Downloaded Video"

    @patch("app.services.downloader.yt_dlp.YoutubeDL")
    def test_download_video_not_found(self, mock_ydl_class, tmp_path):
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.side_effect = yt_dlp.utils.DownloadError(
            "HTTP Error 404: Not Found"
        )

        with pytest.raises(VideoNotFoundError):
            self.downloader.download("https://www.douyin.com/video/deleted", tmp_path)


class TestRetryLogic:
    """Tests for retry and exponential backoff logic."""

    @patch("app.services.downloader.time.sleep")
    @patch("app.services.downloader.yt_dlp.YoutubeDL")
    def test_retry_on_network_error(self, mock_ydl_class, mock_sleep):
        downloader = VideoDownloader(max_retries=3, backoff_base=2)
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.side_effect = Exception("Connection timeout")

        with pytest.raises(NetworkError) as exc_info:
            downloader.get_video_info("https://www.douyin.com/video/123")

        assert exc_info.value.message == "Lỗi kết nối mạng sau 3 lần thử"
        assert exc_info.value.retryable is True
        # Should have slept with exponential backoff: 2^1=2, 2^2=4
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(2)
        mock_sleep.assert_any_call(4)

    @patch("app.services.downloader.time.sleep")
    @patch("app.services.downloader.yt_dlp.YoutubeDL")
    def test_retry_succeeds_on_second_attempt(self, mock_ydl_class, mock_sleep):
        downloader = VideoDownloader(max_retries=3, backoff_base=2)
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.side_effect = [
            Exception("Connection timeout"),
            {
                "duration": 45.0,
                "filesize": 3000000,
                "width": 1080,
                "height": 1920,
                "title": "Retry Success",
            },
        ]

        info = downloader.get_video_info("https://www.douyin.com/video/123")

        assert info.title == "Retry Success"
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(2)

    @patch("app.services.downloader.time.sleep")
    @patch("app.services.downloader.yt_dlp.YoutubeDL")
    def test_no_retry_on_404(self, mock_ydl_class, mock_sleep):
        downloader = VideoDownloader(max_retries=3, backoff_base=2)
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.side_effect = yt_dlp.utils.DownloadError(
            "HTTP Error 404: Not Found"
        )

        with pytest.raises(VideoNotFoundError):
            downloader.get_video_info("https://www.douyin.com/video/deleted")

        # Should NOT retry for 404 errors - only one call made
        mock_sleep.assert_not_called()

    @patch("app.services.downloader.time.sleep")
    @patch("app.services.downloader.yt_dlp.YoutubeDL")
    def test_exponential_backoff_timing(self, mock_ydl_class, mock_sleep):
        """Verify backoff follows 2^1, 2^2, pattern (2s, 4s)."""
        downloader = VideoDownloader(max_retries=3, backoff_base=2)
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.side_effect = Exception("Network error")

        with pytest.raises(NetworkError):
            downloader.get_video_info("https://www.douyin.com/video/123")

        # backoff_base=2: sleep(2^1)=2, sleep(2^2)=4
        # No sleep after last attempt
        calls = mock_sleep.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == 2  # 2^1
        assert calls[1][0][0] == 4  # 2^2
