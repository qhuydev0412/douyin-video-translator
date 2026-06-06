"""Audio extraction service using FFmpeg."""

import json
import subprocess
from pathlib import Path


class AudioExtractorError(Exception):
    """Raised when audio extraction fails."""


class AudioExtractor:
    """Extracts audio tracks from video files using FFmpeg.

    This class is stateless — methods use subprocess to invoke ffprobe and ffmpeg.
    """

    def has_audio_track(self, video_path: Path) -> bool:
        """Check if the video file contains an audio stream.

        Args:
            video_path: Path to the video file.

        Returns:
            True if the video has at least one audio stream, False otherwise.

        Raises:
            AudioExtractorError: If ffprobe fails to analyze the file.
        """
        if not video_path.exists():
            raise AudioExtractorError(f"Video file not found: {video_path}")

        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "a",
            str(video_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            raise AudioExtractorError(
                "ffprobe not found. Please install FFmpeg."
            )
        except subprocess.TimeoutExpired:
            raise AudioExtractorError(
                "ffprobe timed out while analyzing the video file."
            )

        if result.returncode != 0:
            raise AudioExtractorError(
                f"ffprobe failed: {result.stderr.strip()}"
            )

        try:
            probe_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            raise AudioExtractorError(
                "ffprobe returned invalid JSON output."
            )

        streams = probe_data.get("streams", [])
        return len(streams) > 0

    def extract(self, video_path: Path, output_dir: Path) -> Path:
        """Extract the audio track from a video file as WAV.

        Uses FFmpeg to extract audio, preserving quality with PCM signed 16-bit
        little-endian encoding for maximum compatibility.

        Args:
            video_path: Path to the input video file.
            output_dir: Directory where the WAV file will be saved.

        Returns:
            Path to the extracted WAV audio file.

        Raises:
            AudioExtractorError: If extraction fails or video has no audio.
        """
        if not video_path.exists():
            raise AudioExtractorError(f"Video file not found: {video_path}")

        if not self.has_audio_track(video_path):
            raise AudioExtractorError("Video không có âm thanh")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "audio_full.wav"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except FileNotFoundError:
            raise AudioExtractorError(
                "ffmpeg not found. Please install FFmpeg."
            )
        except subprocess.TimeoutExpired:
            raise AudioExtractorError(
                "ffmpeg timed out while extracting audio."
            )

        if result.returncode != 0:
            raise AudioExtractorError(
                f"ffmpeg extraction failed: {result.stderr.strip()}"
            )

        if not output_path.exists():
            raise AudioExtractorError(
                "Audio extraction completed but output file was not created."
            )

        return output_path
