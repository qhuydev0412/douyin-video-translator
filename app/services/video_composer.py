"""Video composition service using FFmpeg."""

import subprocess
from pathlib import Path


class VideoComposerError(Exception):
    """Raised when video composition fails."""


class VideoComposer:
    """Composes final video by mixing Vietnamese voiceover with background music.

    Uses FFmpeg to merge the original video stream with a new audio track
    consisting of Vietnamese voiceover mixed with background music at reduced volume.

    This class is stateless — methods use subprocess to invoke ffmpeg.
    """

    def compose(
        self,
        video_path: Path,
        vietnamese_audio: Path,
        background_audio: Path,
        output_dir: Path,
        background_volume: float = 0.2,
    ) -> Path:
        """Merge Vietnamese voiceover with background music into original video.

        Uses FFmpeg to:
        1. Copy the original video stream (no re-encoding for quality preservation)
        2. Mix Vietnamese voiceover with background audio at reduced volume
        3. Encode mixed audio as AAC for MP4 container compatibility

        Args:
            video_path: Path to the original video file.
            vietnamese_audio: Path to the synthesized Vietnamese voiceover audio.
            background_audio: Path to the isolated background music.
            output_dir: Directory where the output MP4 will be saved.
            background_volume: Volume level for background music (0.0–1.0).
                Defaults to 0.2 (20%).

        Returns:
            Path to the composed output MP4 file.

        Raises:
            VideoComposerError: If composition fails for any reason.
        """
        if not video_path.exists():
            raise VideoComposerError(f"Video file not found: {video_path}")

        if not vietnamese_audio.exists():
            raise VideoComposerError(
                f"Vietnamese audio file not found: {vietnamese_audio}"
            )

        if not background_audio.exists():
            raise VideoComposerError(
                f"Background audio file not found: {background_audio}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "output.mp4"

        # Build the FFmpeg filter to mix voiceover with background at reduced volume
        # [1:a] is the Vietnamese voiceover, [2:a] is the background music
        filter_complex = (
            f"[2:a]volume={background_volume}[bg];"
            f"[1:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-i", str(vietnamese_audio),
            "-i", str(background_audio),
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except FileNotFoundError:
            raise VideoComposerError(
                "ffmpeg not found. Please install FFmpeg."
            )
        except subprocess.TimeoutExpired:
            raise VideoComposerError(
                "ffmpeg timed out while composing video (exceeded 10 minutes)."
            )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "No space left on device" in stderr:
                raise VideoComposerError(
                    "Disk space error: No space left on device."
                )
            raise VideoComposerError(
                f"ffmpeg composition failed: {stderr}"
            )

        if not output_path.exists():
            raise VideoComposerError(
                "Video composition completed but output file was not created."
            )

        return output_path
