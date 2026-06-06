"""Unit tests for AudioExtractor service."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app.services.audio_extractor import AudioExtractor, AudioExtractorError


@pytest.fixture
def extractor() -> AudioExtractor:
    """Provide an AudioExtractor instance."""
    return AudioExtractor()


@pytest.fixture
def fake_video(tmp_path: Path) -> Path:
    """Create a fake video file for testing."""
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b"fake video content")
    return video_file


class TestHasAudioTrack:
    """Tests for AudioExtractor.has_audio_track method."""

    def test_returns_true_when_audio_stream_exists(
        self, extractor: AudioExtractor, fake_video: Path
    ) -> None:
        probe_output = json.dumps({
            "streams": [
                {"index": 0, "codec_type": "audio", "codec_name": "aac"}
            ]
        })
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=probe_output, stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
            assert extractor.has_audio_track(fake_video) is True

    def test_returns_false_when_no_audio_stream(
        self, extractor: AudioExtractor, fake_video: Path
    ) -> None:
        probe_output = json.dumps({"streams": []})
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=probe_output, stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
            assert extractor.has_audio_track(fake_video) is False

    def test_raises_error_when_file_not_found(
        self, extractor: AudioExtractor, tmp_path: Path
    ) -> None:
        nonexistent = tmp_path / "nonexistent.mp4"
        with pytest.raises(AudioExtractorError, match="Video file not found"):
            extractor.has_audio_track(nonexistent)

    def test_raises_error_when_ffprobe_not_installed(
        self, extractor: AudioExtractor, fake_video: Path
    ) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(AudioExtractorError, match="ffprobe not found"):
                extractor.has_audio_track(fake_video)

    def test_raises_error_when_ffprobe_times_out(
        self, extractor: AudioExtractor, fake_video: Path
    ) -> None:
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffprobe", 30)):
            with pytest.raises(AudioExtractorError, match="timed out"):
                extractor.has_audio_track(fake_video)

    def test_raises_error_when_ffprobe_returns_nonzero(
        self, extractor: AudioExtractor, fake_video: Path
    ) -> None:
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="Invalid data"
        )
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(AudioExtractorError, match="ffprobe failed"):
                extractor.has_audio_track(fake_video)

    def test_raises_error_when_ffprobe_returns_invalid_json(
        self, extractor: AudioExtractor, fake_video: Path
    ) -> None:
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not json", stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(AudioExtractorError, match="invalid JSON"):
                extractor.has_audio_track(fake_video)


class TestExtract:
    """Tests for AudioExtractor.extract method."""

    def test_extracts_audio_successfully(
        self, extractor: AudioExtractor, fake_video: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        probe_output = json.dumps({
            "streams": [{"index": 0, "codec_type": "audio"}]
        })

        def mock_run(cmd, **kwargs):
            # First call is ffprobe (has_audio_track), second is ffmpeg (extract)
            if "ffprobe" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=probe_output, stderr=""
                )
            # ffmpeg call - create the output file to simulate extraction
            output_path = output_dir / "audio_full.wav"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"RIFF fake wav data")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        with patch("subprocess.run", side_effect=mock_run):
            result = extractor.extract(fake_video, output_dir)

        assert result == output_dir / "audio_full.wav"
        assert result.exists()

    def test_raises_error_when_video_has_no_audio(
        self, extractor: AudioExtractor, fake_video: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        probe_output = json.dumps({"streams": []})
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=probe_output, stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(AudioExtractorError, match="Video không có âm thanh"):
                extractor.extract(fake_video, output_dir)

    def test_raises_error_when_video_file_not_found(
        self, extractor: AudioExtractor, tmp_path: Path
    ) -> None:
        nonexistent = tmp_path / "nonexistent.mp4"
        output_dir = tmp_path / "output"
        with pytest.raises(AudioExtractorError, match="Video file not found"):
            extractor.extract(nonexistent, output_dir)

    def test_raises_error_when_ffmpeg_not_installed(
        self, extractor: AudioExtractor, fake_video: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        probe_output = json.dumps({
            "streams": [{"index": 0, "codec_type": "audio"}]
        })

        call_count = 0

        def mock_run(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # ffprobe call
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=probe_output, stderr=""
                )
            # ffmpeg call
            raise FileNotFoundError

        with patch("subprocess.run", side_effect=mock_run):
            with pytest.raises(AudioExtractorError, match="ffmpeg not found"):
                extractor.extract(fake_video, output_dir)

    def test_raises_error_when_ffmpeg_times_out(
        self, extractor: AudioExtractor, fake_video: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        probe_output = json.dumps({
            "streams": [{"index": 0, "codec_type": "audio"}]
        })

        call_count = 0

        def mock_run(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # ffprobe call
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=probe_output, stderr=""
                )
            # ffmpeg call
            raise subprocess.TimeoutExpired("ffmpeg", 300)

        with patch("subprocess.run", side_effect=mock_run):
            with pytest.raises(AudioExtractorError, match="timed out"):
                extractor.extract(fake_video, output_dir)

    def test_raises_error_when_ffmpeg_fails(
        self, extractor: AudioExtractor, fake_video: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "output"
        probe_output = json.dumps({
            "streams": [{"index": 0, "codec_type": "audio"}]
        })

        call_count = 0

        def mock_run(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # ffprobe call
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=probe_output, stderr=""
                )
            # ffmpeg call fails
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="Conversion failed"
            )

        with patch("subprocess.run", side_effect=mock_run):
            with pytest.raises(AudioExtractorError, match="extraction failed"):
                extractor.extract(fake_video, output_dir)

    def test_creates_output_directory_if_not_exists(
        self, extractor: AudioExtractor, fake_video: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "nested" / "deep" / "output"
        probe_output = json.dumps({
            "streams": [{"index": 0, "codec_type": "audio"}]
        })

        def mock_run(cmd, **kwargs):
            if "ffprobe" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=probe_output, stderr=""
                )
            # ffmpeg - create output file
            output_path = output_dir / "audio_full.wav"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"RIFF fake wav data")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        with patch("subprocess.run", side_effect=mock_run):
            result = extractor.extract(fake_video, output_dir)

        assert output_dir.exists()
        assert result.exists()
