"""Unit tests for VideoComposer service."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.video_composer import VideoComposer, VideoComposerError


@pytest.fixture
def composer() -> VideoComposer:
    """Provide a VideoComposer instance."""
    return VideoComposer()


@pytest.fixture
def fake_video(tmp_path: Path) -> Path:
    """Create a fake video file for testing."""
    video_file = tmp_path / "original.mp4"
    video_file.write_bytes(b"fake video content")
    return video_file


@pytest.fixture
def fake_vietnamese_audio(tmp_path: Path) -> Path:
    """Create a fake Vietnamese voiceover audio file."""
    audio_file = tmp_path / "vietnamese_audio.wav"
    audio_file.write_bytes(b"fake vietnamese audio")
    return audio_file


@pytest.fixture
def fake_background_audio(tmp_path: Path) -> Path:
    """Create a fake background music audio file."""
    audio_file = tmp_path / "background.wav"
    audio_file.write_bytes(b"fake background audio")
    return audio_file


class TestCompose:
    """Tests for VideoComposer.compose method."""

    def test_composes_video_successfully(
        self,
        composer: VideoComposer,
        fake_video: Path,
        fake_vietnamese_audio: Path,
        fake_background_audio: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "output"

        def mock_run(cmd, **kwargs):
            # Simulate successful ffmpeg composition
            output_path = output_dir / "output.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"composed video content")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        with patch("subprocess.run", side_effect=mock_run):
            result = composer.compose(
                fake_video, fake_vietnamese_audio, fake_background_audio, output_dir
            )

        assert result == output_dir / "output.mp4"
        assert result.exists()

    def test_uses_correct_ffmpeg_command_structure(
        self,
        composer: VideoComposer,
        fake_video: Path,
        fake_vietnamese_audio: Path,
        fake_background_audio: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "output"
        captured_cmd = None

        def mock_run(cmd, **kwargs):
            nonlocal captured_cmd
            captured_cmd = cmd
            output_path = output_dir / "output.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"composed video")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        with patch("subprocess.run", side_effect=mock_run):
            composer.compose(
                fake_video, fake_vietnamese_audio, fake_background_audio, output_dir
            )

        assert captured_cmd is not None
        assert "ffmpeg" in captured_cmd[0]
        assert "-c:v" in captured_cmd
        assert "copy" in captured_cmd
        assert "-c:a" in captured_cmd
        assert "aac" in captured_cmd
        assert "-filter_complex" in captured_cmd

    def test_applies_custom_background_volume(
        self,
        composer: VideoComposer,
        fake_video: Path,
        fake_vietnamese_audio: Path,
        fake_background_audio: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "output"
        captured_cmd = None

        def mock_run(cmd, **kwargs):
            nonlocal captured_cmd
            captured_cmd = cmd
            output_path = output_dir / "output.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"composed video")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        with patch("subprocess.run", side_effect=mock_run):
            composer.compose(
                fake_video,
                fake_vietnamese_audio,
                fake_background_audio,
                output_dir,
                background_volume=0.3,
            )

        # Check that the filter_complex contains the custom volume
        filter_idx = captured_cmd.index("-filter_complex")
        filter_value = captured_cmd[filter_idx + 1]
        assert "volume=0.3" in filter_value

    def test_raises_error_when_video_not_found(
        self,
        composer: VideoComposer,
        fake_vietnamese_audio: Path,
        fake_background_audio: Path,
        tmp_path: Path,
    ) -> None:
        nonexistent = tmp_path / "nonexistent.mp4"
        output_dir = tmp_path / "output"
        with pytest.raises(VideoComposerError, match="Video file not found"):
            composer.compose(
                nonexistent, fake_vietnamese_audio, fake_background_audio, output_dir
            )

    def test_raises_error_when_vietnamese_audio_not_found(
        self,
        composer: VideoComposer,
        fake_video: Path,
        fake_background_audio: Path,
        tmp_path: Path,
    ) -> None:
        nonexistent_audio = tmp_path / "nonexistent_audio.wav"
        output_dir = tmp_path / "output"
        with pytest.raises(VideoComposerError, match="Vietnamese audio file not found"):
            composer.compose(
                fake_video, nonexistent_audio, fake_background_audio, output_dir
            )

    def test_raises_error_when_background_audio_not_found(
        self,
        composer: VideoComposer,
        fake_video: Path,
        fake_vietnamese_audio: Path,
        tmp_path: Path,
    ) -> None:
        nonexistent_bg = tmp_path / "nonexistent_bg.wav"
        output_dir = tmp_path / "output"
        with pytest.raises(VideoComposerError, match="Background audio file not found"):
            composer.compose(
                fake_video, fake_vietnamese_audio, nonexistent_bg, output_dir
            )

    def test_raises_error_when_ffmpeg_not_installed(
        self,
        composer: VideoComposer,
        fake_video: Path,
        fake_vietnamese_audio: Path,
        fake_background_audio: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "output"
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(VideoComposerError, match="ffmpeg not found"):
                composer.compose(
                    fake_video,
                    fake_vietnamese_audio,
                    fake_background_audio,
                    output_dir,
                )

    def test_raises_error_when_ffmpeg_times_out(
        self,
        composer: VideoComposer,
        fake_video: Path,
        fake_vietnamese_audio: Path,
        fake_background_audio: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "output"
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired("ffmpeg", 600),
        ):
            with pytest.raises(VideoComposerError, match="timed out"):
                composer.compose(
                    fake_video,
                    fake_vietnamese_audio,
                    fake_background_audio,
                    output_dir,
                )

    def test_raises_error_when_ffmpeg_fails(
        self,
        composer: VideoComposer,
        fake_video: Path,
        fake_vietnamese_audio: Path,
        fake_background_audio: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "output"
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="Conversion failed"
        )
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(VideoComposerError, match="composition failed"):
                composer.compose(
                    fake_video,
                    fake_vietnamese_audio,
                    fake_background_audio,
                    output_dir,
                )

    def test_raises_error_on_disk_space_failure(
        self,
        composer: VideoComposer,
        fake_video: Path,
        fake_vietnamese_audio: Path,
        fake_background_audio: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "output"
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="Error writing output: No space left on device",
        )
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(VideoComposerError, match="Disk space error"):
                composer.compose(
                    fake_video,
                    fake_vietnamese_audio,
                    fake_background_audio,
                    output_dir,
                )

    def test_creates_output_directory_if_not_exists(
        self,
        composer: VideoComposer,
        fake_video: Path,
        fake_vietnamese_audio: Path,
        fake_background_audio: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "nested" / "deep" / "output"

        def mock_run(cmd, **kwargs):
            output_path = output_dir / "output.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"composed video")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        with patch("subprocess.run", side_effect=mock_run):
            result = composer.compose(
                fake_video, fake_vietnamese_audio, fake_background_audio, output_dir
            )

        assert output_dir.exists()
        assert result.exists()

    def test_preserves_video_quality_with_copy_codec(
        self,
        composer: VideoComposer,
        fake_video: Path,
        fake_vietnamese_audio: Path,
        fake_background_audio: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "output"
        captured_cmd = None

        def mock_run(cmd, **kwargs):
            nonlocal captured_cmd
            captured_cmd = cmd
            output_path = output_dir / "output.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"composed video")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        with patch("subprocess.run", side_effect=mock_run):
            composer.compose(
                fake_video, fake_vietnamese_audio, fake_background_audio, output_dir
            )

        # Verify -c:v copy is used to preserve original video quality
        cv_idx = captured_cmd.index("-c:v")
        assert captured_cmd[cv_idx + 1] == "copy"

    def test_default_background_volume_is_0_2(
        self,
        composer: VideoComposer,
        fake_video: Path,
        fake_vietnamese_audio: Path,
        fake_background_audio: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "output"
        captured_cmd = None

        def mock_run(cmd, **kwargs):
            nonlocal captured_cmd
            captured_cmd = cmd
            output_path = output_dir / "output.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"composed video")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="", stderr=""
            )

        with patch("subprocess.run", side_effect=mock_run):
            composer.compose(
                fake_video, fake_vietnamese_audio, fake_background_audio, output_dir
            )

        filter_idx = captured_cmd.index("-filter_complex")
        filter_value = captured_cmd[filter_idx + 1]
        assert "volume=0.2" in filter_value
