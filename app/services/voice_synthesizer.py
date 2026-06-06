"""Voice synthesis service using edge-tts for Vietnamese TTS generation."""

import asyncio
import json
import logging
import subprocess
from pathlib import Path

import edge_tts

from app.models.pipeline import SegmentAudio, SynthesisResult, TranslationResult

logger = logging.getLogger(__name__)

# Available Vietnamese voices in edge-tts
VIETNAMESE_VOICES = [
    "vi-VN-HoaiMyNeural",   # Female
    "vi-VN-NamMinhNeural",  # Male
]

DEFAULT_VOICE = "vi-VN-HoaiMyNeural"

# Maximum speed multiplier allowed
MAX_SPEED_MULTIPLIER = 2.0


class VoiceSynthesizerError(Exception):
    """Raised when voice synthesis fails."""


class VoiceSynthesizer:
    """Generates Vietnamese TTS audio using edge-tts.

    Synthesizes translated text segments into Vietnamese speech audio files,
    handles multi-speaker voice assignment, speed adjustment for timing sync,
    and combines segments into a final audio file.
    """

    def __init__(self) -> None:
        self._speaker_voice_map: dict[str, str] = {}
        self._voice_index: int = 0

    async def synthesize(
        self, translation: TranslationResult, output_dir: Path
    ) -> SynthesisResult:
        """Generate Vietnamese TTS audio for each segment using edge-tts.

        Args:
            translation: Translation result containing segments to synthesize.
            output_dir: Directory where audio files will be saved.

        Returns:
            SynthesisResult with combined audio path and individual segment audios.

        Raises:
            VoiceSynthesizerError: If synthesis fails completely.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Reset speaker-voice mapping for each synthesis call
        self._speaker_voice_map = {}
        self._voice_index = 0

        segment_audios: list[SegmentAudio] = []

        for i, segment in enumerate(translation.segments):
            if not segment.translated_text.strip():
                logger.warning("Segment %d has empty translated text, skipping", i)
                continue

            # Clean text for TTS (remove special chars that cause "No audio received")
            clean_text = self._clean_text_for_tts(segment.translated_text)
            if not clean_text:
                logger.warning("Segment %d has no speakable text after cleaning, skipping", i)
                continue

            target_duration = segment.end - segment.start
            segment_path = output_dir / f"segment_{i:04d}.mp3"

            try:
                voice = self.select_voice(segment.speaker)
                segment_audio = await self._synthesize_segment(
                    text=clean_text,
                    voice=voice,
                    target_duration=target_duration,
                    output_path=segment_path,
                    start=segment.start,
                    end=segment.end,
                )
                segment_audios.append(segment_audio)
            except Exception as e:
                logger.warning(
                    "Failed to synthesize segment %d with voice %s: %s. "
                    "Trying default voice.",
                    i,
                    self._speaker_voice_map.get(segment.speaker, "unknown"),
                    str(e),
                )
                # Graceful degradation: try default voice
                try:
                    segment_audio = await self._synthesize_segment(
                        text=clean_text,
                        voice=DEFAULT_VOICE,
                        target_duration=target_duration,
                        output_path=segment_path,
                        start=segment.start,
                        end=segment.end,
                    )
                    segment_audios.append(segment_audio)
                except Exception as fallback_err:
                    # Skip this segment instead of crashing
                    logger.warning(
                        "Skipping segment %d — TTS failed for both voices: %s",
                        i,
                        str(fallback_err),
                    )

        if not segment_audios:
            raise VoiceSynthesizerError("No segments were synthesized successfully")

        # Combine segment audios into final audio file with correct timing
        combined_path = output_dir / "vietnamese_audio.wav"
        self._combine_segments(segment_audios, combined_path)

        logger.info(
            "Voice synthesis completed: %d segments synthesized", len(segment_audios)
        )
        return SynthesisResult(audio_path=combined_path, segment_audios=segment_audios)

    @staticmethod
    def _clean_text_for_tts(text: str) -> str:
        """Clean text to avoid edge-tts 'No audio received' errors.

        Removes characters that TTS cannot pronounce (emojis, special symbols,
        lone punctuation). Returns empty string if nothing speakable remains.
        """
        import re
        # Remove emojis and special Unicode symbols
        cleaned = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        # Remove standalone special characters that TTS can't read
        cleaned = re.sub(r'[#@*~`|\\<>{}[\]^]', '', cleaned)
        # Collapse multiple spaces/newlines
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        # If only punctuation remains, return empty
        if cleaned and not re.search(r'[\w\u00C0-\u024F\u1E00-\u1EFF\u4e00-\u9fff]', cleaned):
            return ""
        return cleaned

    def select_voice(self, speaker: str | None) -> str:
        """Select appropriate Vietnamese voice for speaker.

        Assigns distinct voices to distinct speakers. If speaker is None,
        returns the default voice.

        Args:
            speaker: Speaker label or None for default.

        Returns:
            Vietnamese voice ID string for edge-tts.
        """
        if speaker is None:
            return DEFAULT_VOICE

        if speaker in self._speaker_voice_map:
            return self._speaker_voice_map[speaker]

        # Assign next available voice (cycling through available voices)
        voice = VIETNAMESE_VOICES[self._voice_index % len(VIETNAMESE_VOICES)]
        self._speaker_voice_map[speaker] = voice
        self._voice_index += 1

        return voice

    async def _synthesize_segment(
        self,
        text: str,
        voice: str,
        target_duration: float,
        output_path: Path,
        start: float,
        end: float,
    ) -> SegmentAudio:
        """Synthesize a single segment with optional speed adjustment.

        First generates audio at normal speed, then checks duration.
        If TTS duration exceeds target, adjusts speed up to MAX_SPEED_MULTIPLIER.

        Args:
            text: Vietnamese text to synthesize.
            voice: edge-tts voice ID.
            target_duration: Target duration in seconds.
            output_path: Path to save the audio file.
            start: Segment start time in seconds.
            end: Segment end time in seconds.

        Returns:
            SegmentAudio with actual duration and speed adjustment info.
        """
        # Generate initial TTS audio at normal speed
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))

        # Get actual duration of generated audio
        actual_duration = self._get_audio_duration(output_path)
        speed_adjusted = False

        # If TTS duration exceeds target, adjust speed
        if actual_duration > target_duration and target_duration > 0:
            speed_multiplier = actual_duration / target_duration

            if speed_multiplier <= MAX_SPEED_MULTIPLIER:
                # Adjust speed using edge-tts rate parameter
                rate_percent = int((speed_multiplier - 1.0) * 100)
                rate_str = f"+{rate_percent}%"

                communicate = edge_tts.Communicate(text, voice, rate=rate_str)
                await communicate.save(str(output_path))

                actual_duration = self._get_audio_duration(output_path)
                speed_adjusted = True
            else:
                # Speed exceeds max, apply maximum speed adjustment
                rate_str = "+100%"
                communicate = edge_tts.Communicate(text, voice, rate=rate_str)
                await communicate.save(str(output_path))

                actual_duration = self._get_audio_duration(output_path)
                speed_adjusted = True

        return SegmentAudio(
            path=output_path,
            start=start,
            end=end,
            duration=actual_duration,
            target_duration=target_duration,
            speed_adjusted=speed_adjusted,
        )

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of an audio file using FFprobe.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Duration in seconds.

        Raises:
            VoiceSynthesizerError: If FFprobe fails.
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(audio_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            raise VoiceSynthesizerError(
                "ffprobe not found. Please install FFmpeg."
            )
        except subprocess.TimeoutExpired:
            raise VoiceSynthesizerError(
                "ffprobe timed out while analyzing audio file."
            )

        if result.returncode != 0:
            raise VoiceSynthesizerError(
                f"ffprobe failed: {result.stderr.strip()}"
            )

        try:
            probe_data = json.loads(result.stdout)
            duration = float(probe_data["format"]["duration"])
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise VoiceSynthesizerError(
                f"Failed to parse ffprobe output: {e}"
            )

        return duration

    def _combine_segments(
        self, segment_audios: list[SegmentAudio], output_path: Path
    ) -> None:
        """Combine individual segment audios into a final audio file with correct timing.

        Uses FFmpeg to create a combined audio file where each segment is placed
        at its correct start time, with silence filling gaps between segments.

        Args:
            segment_audios: List of synthesized segment audio files with timing info.
            output_path: Path for the combined output WAV file.

        Raises:
            VoiceSynthesizerError: If FFmpeg combination fails.
        """
        if not segment_audios:
            raise VoiceSynthesizerError("No segments to combine")

        # Determine total duration from the last segment's end time
        total_duration = max(seg.end for seg in segment_audios)

        # Build FFmpeg filter complex to place each segment at correct time
        inputs = []
        filter_parts = []

        for i, seg in enumerate(segment_audios):
            inputs.extend(["-i", str(seg.path)])
            # Delay each segment by its start time (in milliseconds)
            delay_ms = int(seg.start * 1000)
            filter_parts.append(
                f"[{i}:a]adelay={delay_ms}|{delay_ms}[delayed{i}]"
            )

        # Mix all delayed segments together
        mix_inputs = "".join(f"[delayed{i}]" for i in range(len(segment_audios)))
        filter_parts.append(
            f"{mix_inputs}amix=inputs={len(segment_audios)}:duration=longest:dropout_transition=0[mixed]"
        )

        # Normalize volume after mixing (amix reduces volume by number of inputs)
        filter_parts.append(
            f"[mixed]volume={len(segment_audios)}[out]"
        )

        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[out]",
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
            raise VoiceSynthesizerError(
                "ffmpeg not found. Please install FFmpeg."
            )
        except subprocess.TimeoutExpired:
            raise VoiceSynthesizerError(
                "ffmpeg timed out while combining audio segments."
            )

        if result.returncode != 0:
            raise VoiceSynthesizerError(
                f"ffmpeg combination failed: {result.stderr.strip()}"
            )

        if not output_path.exists():
            raise VoiceSynthesizerError(
                "Audio combination completed but output file was not created."
            )
