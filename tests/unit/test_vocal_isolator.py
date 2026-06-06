"""Unit tests for the VocalIsolator service."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.models.pipeline import VocalIsolationResult
from app.services.vocal_isolator import VocalIsolator


@pytest.fixture
def isolator() -> VocalIsolator:
    """Create a VocalIsolator instance."""
    return VocalIsolator()


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    """Create a fake audio file for testing."""
    audio = tmp_path / "audio_full.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 100)  # Minimal WAV-like content
    return audio


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create output directory for isolation results."""
    out = tmp_path / "output"
    out.mkdir()
    return out


class TestVocalIsolatorIsolate:
    """Tests for VocalIsolator.isolate method."""

    def test_successful_isolation(
        self, isolator: VocalIsolator, audio_file: Path, output_dir: Path
    ) -> None:
        """Test successful vocal isolation produces vocals and background paths."""
        # Simulate demucs creating output files
        def mock_run(*args, **kwargs):
            # Create the expected demucs output structure
            track_name = audio_file.stem
            stems_dir = output_dir / "htdemucs" / track_name
            stems_dir.mkdir(parents=True)
            (stems_dir / "vocals.wav").write_bytes(b"vocals_data")
            (stems_dir / "no_vocals.wav").write_bytes(b"background_data")

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            return mock_result

        with patch("app.services.vocal_isolator.subprocess.run", side_effect=mock_run):
            result = isolator.isolate(audio_file, output_dir)

        assert isinstance(result, VocalIsolationResult)
        assert result.vocals_path == output_dir / "vocals.wav"
        assert result.background_path == output_dir / "background.wav"
        assert result.vocals_path.exists()
        assert result.background_path.exists()
        assert result.vocals_path.read_bytes() == b"vocals_data"
        assert result.background_path.read_bytes() == b"background_data"

    def test_demucs_failure_triggers_fallback(
        self, isolator: VocalIsolator, audio_file: Path, output_dir: Path
    ) -> None:
        """Test that demucs failure results in graceful degradation."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "CUDA out of memory"

        with patch(
            "app.services.vocal_isolator.subprocess.run", return_value=mock_result
        ):
            result = isolator.isolate(audio_file, output_dir)

        assert isinstance(result, VocalIsolationResult)
        assert result.vocals_path == output_dir / "vocals.wav"
        assert result.background_path == output_dir / "background.wav"
        # Both should be copies of the original audio
        assert result.vocals_path.read_bytes() == audio_file.read_bytes()
        assert result.background_path.read_bytes() == audio_file.read_bytes()

    def test_timeout_triggers_fallback(
        self, isolator: VocalIsolator, audio_file: Path, output_dir: Path
    ) -> None:
        """Test that subprocess timeout triggers graceful degradation."""
        import subprocess

        with patch(
            "app.services.vocal_isolator.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="demucs", timeout=600),
        ):
            result = isolator.isolate(audio_file, output_dir)

        assert isinstance(result, VocalIsolationResult)
        assert result.vocals_path.exists()
        assert result.background_path.exists()
        assert result.vocals_path.read_bytes() == audio_file.read_bytes()

    def test_os_error_triggers_fallback(
        self, isolator: VocalIsolator, audio_file: Path, output_dir: Path
    ) -> None:
        """Test that OSError (e.g. demucs not found) triggers graceful degradation."""
        with patch(
            "app.services.vocal_isolator.subprocess.run",
            side_effect=OSError("No such file or directory: 'python'"),
        ):
            result = isolator.isolate(audio_file, output_dir)

        assert isinstance(result, VocalIsolationResult)
        assert result.vocals_path.exists()
        assert result.background_path.exists()
        assert result.vocals_path.read_bytes() == audio_file.read_bytes()

    def test_missing_output_files_triggers_fallback(
        self, isolator: VocalIsolator, audio_file: Path, output_dir: Path
    ) -> None:
        """Test fallback when demucs succeeds but output files are missing."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch(
            "app.services.vocal_isolator.subprocess.run", return_value=mock_result
        ):
            result = isolator.isolate(audio_file, output_dir)

        # Should fallback since no output files were created
        assert isinstance(result, VocalIsolationResult)
        assert result.vocals_path.read_bytes() == audio_file.read_bytes()
        assert result.background_path.read_bytes() == audio_file.read_bytes()

    def test_creates_output_dir_if_not_exists(
        self, isolator: VocalIsolator, audio_file: Path, tmp_path: Path
    ) -> None:
        """Test that output_dir is created if it doesn't exist."""
        new_output = tmp_path / "nested" / "output" / "dir"
        assert not new_output.exists()

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error"

        with patch(
            "app.services.vocal_isolator.subprocess.run", return_value=mock_result
        ):
            result = isolator.isolate(audio_file, new_output)

        assert new_output.exists()
        assert result.vocals_path.exists()

    def test_cleans_up_htdemucs_directory(
        self, isolator: VocalIsolator, audio_file: Path, output_dir: Path
    ) -> None:
        """Test that intermediate htdemucs directory is cleaned up after success."""
        def mock_run(*args, **kwargs):
            track_name = audio_file.stem
            stems_dir = output_dir / "htdemucs" / track_name
            stems_dir.mkdir(parents=True)
            (stems_dir / "vocals.wav").write_bytes(b"vocals")
            (stems_dir / "no_vocals.wav").write_bytes(b"no_vocals")

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            return mock_result

        with patch("app.services.vocal_isolator.subprocess.run", side_effect=mock_run):
            isolator.isolate(audio_file, output_dir)

        # htdemucs directory should be cleaned up
        assert not (output_dir / "htdemucs").exists()


class TestVocalIsolatorFallback:
    """Tests for the _fallback method."""

    def test_fallback_copies_original_to_both_paths(
        self, isolator: VocalIsolator, audio_file: Path, output_dir: Path
    ) -> None:
        """Test that fallback creates copies of original audio for both outputs."""
        result = isolator._fallback(audio_file, output_dir)

        assert result.vocals_path == output_dir / "vocals.wav"
        assert result.background_path == output_dir / "background.wav"
        assert result.vocals_path.read_bytes() == audio_file.read_bytes()
        assert result.background_path.read_bytes() == audio_file.read_bytes()

    def test_fallback_returns_valid_result_type(
        self, isolator: VocalIsolator, audio_file: Path, output_dir: Path
    ) -> None:
        """Test that fallback returns a proper VocalIsolationResult."""
        result = isolator._fallback(audio_file, output_dir)

        assert isinstance(result, VocalIsolationResult)
        assert isinstance(result.vocals_path, Path)
        assert isinstance(result.background_path, Path)
