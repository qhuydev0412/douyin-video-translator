"""Video composition service using FFmpeg."""

import subprocess
from pathlib import Path

from app.models.pipeline import TranslationResult


class VideoComposerError(Exception):
    """Raised when video composition fails."""


class VideoComposer:
    """Composes final video by mixing Vietnamese voiceover with background music.

    Uses FFmpeg to merge the original video stream with a new audio track
    consisting of Vietnamese voiceover mixed with background music at reduced volume.
    Optionally burns Vietnamese subtitles into the video.

    This class is stateless — methods use subprocess to invoke ffmpeg.
    """

    def compose(
        self,
        video_path: Path,
        vietnamese_audio: Path,
        background_audio: Path,
        output_dir: Path,
        background_volume: float = 0.2,
        translation: TranslationResult | None = None,
    ) -> Path:
        """Merge Vietnamese voiceover with background music into original video.

        Uses FFmpeg to:
        1. Mix Vietnamese voiceover with background audio at reduced volume
        2. Burn Vietnamese subtitles into the video (if translation provided)
        3. Encode as H.264 + AAC MP4

        Args:
            video_path: Path to the original video file.
            vietnamese_audio: Path to the synthesized Vietnamese voiceover audio.
            background_audio: Path to the isolated background music.
            output_dir: Directory where the output MP4 will be saved.
            background_volume: Volume level for background music (0.0–1.0).
            translation: TranslationResult for subtitle generation (optional).

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

        # Generate subtitle file if translation provided
        subtitle_path = None
        if translation and translation.segments:
            subtitle_path = output_dir / "subtitles.ass"
            self._generate_ass_subtitles(translation, subtitle_path)

        # Build FFmpeg command
        # Audio filter: mix voiceover with background
        audio_filter = (
            f"[2:a]volume={background_volume}[bg];"
            f"[1:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )

        if subtitle_path and subtitle_path.exists():
            # With subtitles: need to re-encode video (can't use -c:v copy with subtitles filter)
            # Use ASS subtitles for styled text
            sub_path_escaped = str(subtitle_path).replace("\\", "/").replace(":", "\\:")
            video_filter = f"ass='{sub_path_escaped}'"

            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(video_path),
                "-i", str(vietnamese_audio),
                "-i", str(background_audio),
                "-filter_complex", audio_filter,
                "-vf", video_filter,
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "192k",
                str(output_path),
            ]
        else:
            # No subtitles: copy video stream (fast, no quality loss)
            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(video_path),
                "-i", str(vietnamese_audio),
                "-i", str(background_audio),
                "-filter_complex", audio_filter,
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

    def _generate_ass_subtitles(
        self, translation: TranslationResult, output_path: Path
    ) -> None:
        """Generate ASS subtitle file with styled Vietnamese text.

        ASS format allows custom fonts, colors, and positioning
        for better readability over video content.
        """
        # ASS header with style definition
        header = """[Script Info]
Title: Vietnamese Subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,30,30,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        lines = []
        for seg in translation.segments:
            text = seg.translated_text.strip()
            if not text or text == "...":
                continue

            start_ts = self._seconds_to_ass_time(seg.start)
            end_ts = self._seconds_to_ass_time(seg.end)

            # Escape special ASS characters
            text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
            # Wrap long lines
            if len(text) > 40:
                mid = len(text) // 2
                space_idx = text.find(" ", mid)
                if space_idx > 0:
                    text = text[:space_idx] + "\\N" + text[space_idx + 1:]

            lines.append(f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{text}")

        content = header + "\n".join(lines) + "\n"
        output_path.write_text(content, encoding="utf-8")

    @staticmethod
    def _seconds_to_ass_time(seconds: float) -> str:
        """Convert seconds to ASS time format (H:MM:SS.CC)."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds % 1) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
