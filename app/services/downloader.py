"""Video downloader module for Douyin videos using yt-dlp."""

import time
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp

from app.core.config import settings
from app.models.job import VideoInfo
from app.models.pipeline import DownloadResult


class DownloadError(Exception):
    """Base exception for download errors."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.retryable = retryable


class InvalidURLError(DownloadError):
    """Raised when the provided URL is not a valid Douyin URL."""

    def __init__(self):
        super().__init__(
            message="URL không hợp lệ, vui lòng cung cấp link Douyin",
            retryable=False,
        )


class VideoNotFoundError(DownloadError):
    """Raised when the video does not exist or has been removed."""

    def __init__(self):
        super().__init__(
            message="Video không tồn tại hoặc đã bị xóa",
            retryable=False,
        )


class NetworkError(DownloadError):
    """Raised when a network error occurs after all retries are exhausted."""

    def __init__(self, max_retries: int):
        super().__init__(
            message=f"Lỗi kết nối mạng sau {max_retries} lần thử",
            retryable=True,
        )


class VideoDownloader:
    """Downloads videos from Douyin using yt-dlp with retry logic."""

    def __init__(
        self,
        max_retries: int | None = None,
        backoff_base: int | None = None,
    ):
        self.max_retries = max_retries if max_retries is not None else settings.MAX_RETRY_ATTEMPTS
        self.backoff_base = backoff_base if backoff_base is not None else settings.RETRY_BACKOFF_BASE

    def validate_url(self, url: str) -> bool:
        """Check if URL belongs to the douyin.com domain (including subdomains).

        Returns True if the URL hostname ends with 'douyin.com'.
        Supports subdomains like www.douyin.com, v.douyin.com.
        """
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.hostname:
                return False
            hostname = parsed.hostname.lower()
            return hostname == "douyin.com" or hostname.endswith(".douyin.com")
        except (ValueError, AttributeError):
            return False

    def get_video_info(self, url: str) -> VideoInfo:
        """Extract video metadata without downloading.

        Args:
            url: A valid Douyin URL.

        Returns:
            VideoInfo with duration, file size, resolution, and title.

        Raises:
            InvalidURLError: If URL is not a valid Douyin URL.
            VideoNotFoundError: If the video doesn't exist or was removed.
            NetworkError: If network fails after max retries.
        """
        if not self.validate_url(url):
            raise InvalidURLError()

        info = self._extract_info_with_retry(url)

        duration = info.get("duration") or 0.0
        filesize = info.get("filesize") or info.get("filesize_approx") or 0
        width = info.get("width") or 0
        height = info.get("height") or 0
        resolution = f"{width}x{height}" if width and height else "unknown"
        title = info.get("title")

        return VideoInfo(
            duration_seconds=float(duration),
            file_size_bytes=int(filesize),
            resolution=resolution,
            title=title,
        )

    def download(self, url: str, output_dir: Path) -> DownloadResult:
        """Download video from Douyin URL using yt-dlp.

        Args:
            url: A valid Douyin URL.
            output_dir: Directory to save the downloaded video.

        Returns:
            DownloadResult with video_path and video_info.

        Raises:
            InvalidURLError: If URL is not a valid Douyin URL.
            VideoNotFoundError: If the video doesn't exist or was removed.
            NetworkError: If network fails after max retries.
        """
        if not self.validate_url(url):
            raise InvalidURLError()

        output_dir.mkdir(parents=True, exist_ok=True)
        output_template = str(output_dir / "original.%(ext)s")

        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": output_template,
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
        }

        info = self._download_with_retry(url, ydl_opts)

        # Determine the actual output file path
        video_path = output_dir / "original.mp4"
        if not video_path.exists():
            # Fallback: look for any mp4 file in output_dir
            mp4_files = list(output_dir.glob("original.*"))
            if mp4_files:
                video_path = mp4_files[0]
            else:
                raise DownloadError("Download completed but output file not found", retryable=True)

        duration = info.get("duration") or 0.0
        filesize = video_path.stat().st_size if video_path.exists() else 0
        width = info.get("width") or 0
        height = info.get("height") or 0
        resolution = f"{width}x{height}" if width and height else "unknown"
        title = info.get("title")

        video_info = VideoInfo(
            duration_seconds=float(duration),
            file_size_bytes=int(filesize),
            resolution=resolution,
            title=title,
        )

        return DownloadResult(video_path=video_path, video_info=video_info)

    def _extract_info_with_retry(self, url: str) -> dict:
        """Extract video info with retry logic for network failures."""
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }
        return self._execute_with_retry(
            lambda: self._extract_info(url, ydl_opts)
        )

    def _download_with_retry(self, url: str, ydl_opts: dict) -> dict:
        """Download video with retry logic for network failures."""
        return self._execute_with_retry(
            lambda: self._perform_download(url, ydl_opts)
        )

    def _execute_with_retry(self, operation) -> dict:
        """Execute an operation with exponential backoff retry logic.

        Retries up to max_retries times with exponential backoff (base^attempt seconds).
        Only retries on network-related errors. Non-retryable errors (404, etc.)
        are raised immediately.
        """
        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return operation()
            except VideoNotFoundError:
                raise
            except InvalidURLError:
                raise
            except DownloadError as e:
                if not e.retryable:
                    raise
                last_exception = e
            except yt_dlp.utils.DownloadError as e:
                error_msg = str(e).lower()
                if self._is_not_found_error(error_msg):
                    raise VideoNotFoundError()
                last_exception = e
            except Exception as e:
                last_exception = e

            # Apply exponential backoff before next retry
            if attempt < self.max_retries - 1:
                backoff_time = self.backoff_base ** (attempt + 1)
                time.sleep(backoff_time)

        raise NetworkError(max_retries=self.max_retries)

    def _extract_info(self, url: str, ydl_opts: dict) -> dict:
        """Extract video information using yt-dlp."""
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info is None:
                    raise VideoNotFoundError()
                return info
        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e).lower()
            if self._is_not_found_error(error_msg):
                raise VideoNotFoundError()
            raise

    def _perform_download(self, url: str, ydl_opts: dict) -> dict:
        """Perform the actual download using yt-dlp."""
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    raise VideoNotFoundError()
                return info
        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e).lower()
            if self._is_not_found_error(error_msg):
                raise VideoNotFoundError()
            raise

    @staticmethod
    def _is_not_found_error(error_msg: str) -> bool:
        """Check if an error message indicates a 404/removed video."""
        not_found_indicators = [
            "404",
            "not found",
            "does not exist",
            "has been removed",
            "has been deleted",
            "is not available",
            "unavailable",
            "removed",
        ]
        return any(indicator in error_msg for indicator in not_found_indicators)
