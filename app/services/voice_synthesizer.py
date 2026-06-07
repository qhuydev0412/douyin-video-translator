"""Voice synthesis service using OpenAI TTS for natural Vietnamese voices."""

import json
import logging
import os
import subprocess
from pathlib import Path

from openai import OpenAI

from app.models.pipeline import SegmentAudio, SynthesisResult, TranslationResult

logger = logging.getLogger(__name__)

# OpenAI TTS voices — assign by gender/tone
# alloy: neutral, echo: male deep, fable: expressive, onyx: male authoritative
# nova: female warm, shimmer: female soft
MALE_VOICES = ["echo", "onyx", "alloy"]
FEMALE_VOICES = ["nova", "shimmer", "fable"]
DEFAULT_VOICE = "nova"

# All available Vietnamese TTS voices in assignment order
VIETNAMESE_VOICES = ["nova", "onyx", "shimmer", "echo", "alloy", "fable"]

# Maximum speed multiplier
MAX_SPEED_MULTIPLIER = 2.0


class VoiceSynthesizerError(Exception):
    """Raised when voice synthesis fails."""


class VoiceSynthesizer:
    """Generates Vietnamese TTS audio using OpenAI TTS API.

    Provides higher quality, more natural voices compared to edge-tts.
    Supports distinct male/female voices for multi-speaker content.
    """

    def __init__(self, model: str = "tts-1") -> None:
        """Initialize voice synthesizer.

        Args:
            model: OpenAI TTS model. "tts-1" (fast) or "tts-1-hd" (higher quality).
        """
        self._model = model
        self._client: OpenAI | None = None
        self._speaker_voice_map: dict[str, str] = {}
        self._male_index: int = 0
        self._female_index: int = 0
        self._speaker_count: int = 0

    @property
    def client(self) -> OpenAI:
        """Lazy-initialize OpenAI client."""
        if self._client is None:
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client

    async def synthesize(
        self, translation: TranslationResult, output_dir: Path, voice: str | None = None
    ) -> SynthesisResult:
        """Generate Vietnamese TTS audio for each segment.

        Synthesizes all segments in parallel for speed.

        Args:
            translation: The translation result containing segments to synthesize.
            output_dir: Directory to store generated audio files.
            voice: Optional voice ID to use for all segments. If None, auto-selects
                based on speaker labels.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        self._speaker_voice_map = {}
        self._speaker_count = 0
        self._male_index = 0
        self._female_index = 0

        # Prepare tasks for parallel execution
        import concurrent.futures

        tasks = []
        for i, segment in enumerate(translation.segments):
            text = segment.translated_text.strip()
            if not text:
                continue
            clean_text = self._clean_text_for_tts(text)
            if not clean_text:
                continue

            target_duration = segment.end - segment.start
            segment_path = output_dir / f"segment_{i:04d}.mp3"
            # Use explicit voice if provided, otherwise auto-select by speaker
            segment_voice = voice if voice else self.select_voice(segment.speaker)

            tasks.append({
                "index": i,
                "text": clean_text,
                "voice": segment_voice,
                "target_duration": target_duration,
                "output_path": segment_path,
                "start": segment.start,
                "end": segment.end,
            })

        if not tasks:
            raise VoiceSynthesizerError("No segments to synthesize")

        # Execute TTS calls in parallel (max 5 concurrent)
        segment_audios: list[SegmentAudio] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self._synthesize_segment_sync, t): t
                for t in tasks
            }
            for future in concurrent.futures.as_completed(futures):
                task_info = futures[future]
                try:
                    result = future.result()
                    segment_audios.append(result)
                except Exception as e:
                    logger.warning("Skipping segment %d — TTS failed: %s", task_info["index"], str(e))

        if not segment_audios:
            raise VoiceSynthesizerError("No segments were synthesized")

        # Sort by start time
        segment_audios.sort(key=lambda s: s.start)

        # Combine segments into final audio
        combined_path = output_dir / "vietnamese_audio.wav"
        self._combine_segments(segment_audios, combined_path)

        logger.info("Voice synthesis completed: %d segments", len(segment_audios))
        return SynthesisResult(audio_path=combined_path, segment_audios=segment_audios)

    def _synthesize_segment_sync(self, task: dict) -> SegmentAudio:
        """Synchronous wrapper for parallel TTS execution."""
        return self._synthesize_segment(
            text=task["text"],
            voice=task["voice"],
            target_duration=task["target_duration"],
            output_path=task["output_path"],
            start=task["start"],
            end=task["end"],
        )

    def select_voice(self, speaker: str | None) -> str:
        """Assign voice based on speaker label.

        Uses a single consistent voice for the primary speaker.
        Only assigns different voice if there are clearly multiple speakers.
        Cycles through VIETNAMESE_VOICES in round-robin order.
        """
        if speaker is None:
            return DEFAULT_VOICE

        if speaker in self._speaker_voice_map:
            return self._speaker_voice_map[speaker]

        voice = VIETNAMESE_VOICES[self._speaker_count % len(VIETNAMESE_VOICES)]

        self._speaker_count += 1
        self._speaker_voice_map[speaker] = voice
        return voice

    def _synthesize_segment(
        self,
        text: str,
        voice: str,
        target_duration: float,
        output_path: Path,
        start: float,
        end: float,
    ) -> SegmentAudio:
        """Synthesize a single segment with OpenAI TTS."""
        # Estimate speed based on text length vs target duration
        # Vietnamese ~3-4 syllables/sec at normal speed
        speed = 1.0
        if target_duration > 0.5 and len(text) > 20:
            # If text is long relative to duration, speed up slightly
            estimated_duration = len(text) * 0.15  # rough: 0.15s per char
            if estimated_duration > target_duration:
                speed = min(estimated_duration / target_duration, 2.0)

        # Generate TTS (single API call)
        response = self.client.audio.speech.create(
            model=self._model,
            voice=voice,
            input=text,
            response_format="mp3",
            speed=speed,
        )
        response.stream_to_file(str(output_path))

        # Get duration (fast, local ffprobe)
        actual_duration = self._get_audio_duration(output_path)

        return SegmentAudio(
            path=output_path,
            start=start,
            end=end,
            duration=actual_duration,
            target_duration=target_duration,
            speed_adjusted=speed != 1.0,
        )

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration using ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", str(audio_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return 0.0
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
        except Exception:
            return 0.0

    def _combine_segments(self, segment_audios: list[SegmentAudio], output_path: Path) -> None:
        """Combine segment audios with correct timing using FFmpeg."""
        if not segment_audios:
            raise VoiceSynthesizerError("No segments to combine")

        inputs = []
        filter_parts = []

        for i, seg in enumerate(segment_audios):
            inputs.extend(["-i", str(seg.path)])
            delay_ms = int(seg.start * 1000)
            filter_parts.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[delayed{i}]")

        mix_inputs = "".join(f"[delayed{i}]" for i in range(len(segment_audios)))
        filter_parts.append(
            f"{mix_inputs}amix=inputs={len(segment_audios)}:duration=longest:dropout_transition=0[mixed]"
        )
        filter_parts.append(f"[mixed]volume={len(segment_audios)}[out]")

        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg", "-y", *inputs,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
            str(output_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise VoiceSynthesizerError(f"FFmpeg failed: {result.stderr[:200]}")
        except FileNotFoundError:
            raise VoiceSynthesizerError("ffmpeg not found")

        if not output_path.exists():
            raise VoiceSynthesizerError("Combined audio not created")

    @staticmethod
    def _clean_text_for_tts(text: str) -> str:
        """Clean text for TTS — remove unspeakable characters."""
        import re
        cleaned = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        cleaned = re.sub(r'[#@*~`|\\<>{}[\]^]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if cleaned and not re.search(r'[\w\u00C0-\u024F\u1E00-\u1EFF\u4e00-\u9fff]', cleaned):
            return ""
        return cleaned
