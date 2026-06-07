"""Voice preview generation service for the pipeline checkpoint workflow.

Generates short audio previews of multiple voice options so users can
listen and select their preferred voice before full synthesis.
"""

import json
import logging
from pathlib import Path

from app.models.job import VoiceOption
from app.models.pipeline import TranslatedSegment, TranslationResult
from app.services.voice_synthesizer import VoiceSynthesizer

logger = logging.getLogger(__name__)

# Voice options available for preview
PREVIEW_VOICES: list[dict[str, str]] = [
    {"voice_id": "nova", "voice_name": "Nova (female, warm)"},
    {"voice_id": "onyx", "voice_name": "Onyx (male, deep)"},
    {"voice_id": "shimmer", "voice_name": "Shimmer (female, soft)"},
    {"voice_id": "echo", "voice_name": "Echo (male, neutral)"},
    {"voice_id": "alloy", "voice_name": "Alloy (neutral)"},
]

# Estimated characters per second for Vietnamese speech duration estimation
CHARS_PER_SECOND = 4.0

# Maximum estimated speech duration for preview segment (seconds)
MAX_PREVIEW_DURATION_SECONDS = 15.0

# Minimum number of successful previews required to proceed
MIN_SUCCESSFUL_PREVIEWS = 2

# Minimum number of voices to attempt
MIN_VOICE_OPTIONS = 3


class VoicePreviewError(Exception):
    """Raised when voice preview generation fails critically."""


class VoicePreviewGenerator:
    """Generates voice preview samples for user selection.

    Selects the longest translated segment within 15 seconds estimated
    speech duration, then generates audio previews using multiple voice options.
    """

    def __init__(self, synthesizer: VoiceSynthesizer) -> None:
        """Initialize with a VoiceSynthesizer instance for TTS calls.

        Args:
            synthesizer: The VoiceSynthesizer used to generate preview audio.
        """
        self._synthesizer = synthesizer

    def select_preview_segment(
        self, segments: list[TranslatedSegment]
    ) -> TranslatedSegment:
        """Select the longest translated segment not exceeding 15s estimated duration.

        Duration is estimated at ~4 characters per second for Vietnamese.
        A segment exceeding 15s estimated speech duration is excluded.

        Args:
            segments: List of translated segments to choose from.

        Returns:
            The best segment for preview (longest text within duration limit).

        Raises:
            VoicePreviewError: If no suitable segment is found.
        """
        max_chars = int(MAX_PREVIEW_DURATION_SECONDS * CHARS_PER_SECOND)
        best_segment: TranslatedSegment | None = None
        best_length = 0

        for segment in segments:
            text = segment.translated_text.strip()
            if not text:
                continue
            text_length = len(text)
            if text_length <= max_chars and text_length > best_length:
                best_segment = segment
                best_length = text_length

        if best_segment is None:
            # If all segments exceed the limit, pick the shortest one
            # (still provides a reasonable preview)
            non_empty = [s for s in segments if s.translated_text.strip()]
            if non_empty:
                best_segment = min(
                    non_empty, key=lambda s: len(s.translated_text.strip())
                )
            else:
                raise VoicePreviewError("No non-empty segments available for preview")

        return best_segment

    async def generate_previews(
        self,
        translation: TranslationResult,
        work_dir: Path,
        job_id: str,
    ) -> list[VoiceOption]:
        """Generate voice previews for multiple voice options.

        Selects the best preview segment, then generates audio for each
        available voice. Handles partial failures: proceeds if >= 2 voice
        previews succeed, fails otherwise.

        Args:
            translation: The full translation result with segments.
            work_dir: Working directory for this job.
            job_id: Job ID for constructing preview URLs.

        Returns:
            List of VoiceOption objects for successfully generated previews.

        Raises:
            VoicePreviewError: If fewer than 2 previews generate successfully.
        """
        # Select the best segment for preview
        segment = self.select_preview_segment(translation.segments)
        preview_text = segment.translated_text.strip()

        # Create voice previews directory
        previews_dir = work_dir / "voice_previews"
        previews_dir.mkdir(parents=True, exist_ok=True)

        # Generate previews for each voice option
        voices_to_try = PREVIEW_VOICES[:max(MIN_VOICE_OPTIONS, len(PREVIEW_VOICES))]
        successful_options: list[VoiceOption] = []

        for voice_info in voices_to_try:
            voice_id = voice_info["voice_id"]
            voice_name = voice_info["voice_name"]
            output_path = previews_dir / f"{voice_id}_preview.mp3"

            try:
                self._generate_single_preview(
                    text=preview_text,
                    voice=voice_id,
                    output_path=output_path,
                )
                preview_url = f"/api/v1/jobs/{job_id}/preview/voice/{voice_id}"
                successful_options.append(
                    VoiceOption(
                        voice_id=voice_id,
                        voice_name=voice_name,
                        preview_url=preview_url,
                    )
                )
                logger.info(
                    "Generated voice preview for '%s' (job %s)",
                    voice_id,
                    job_id,
                )
            except Exception as e:
                logger.warning(
                    "Failed to generate preview for voice '%s' (job %s): %s",
                    voice_id,
                    job_id,
                    str(e),
                )

        # Check minimum threshold
        if len(successful_options) < MIN_SUCCESSFUL_PREVIEWS:
            raise VoicePreviewError(
                f"Only {len(successful_options)} voice previews generated successfully "
                f"(minimum {MIN_SUCCESSFUL_PREVIEWS} required)"
            )

        # Save voice_options.json metadata
        self._save_voice_options_metadata(
            previews_dir, successful_options, preview_text
        )

        logger.info(
            "Voice preview generation complete: %d/%d options available (job %s)",
            len(successful_options),
            len(voices_to_try),
            job_id,
        )

        return successful_options

    def _generate_single_preview(
        self, text: str, voice: str, output_path: Path
    ) -> None:
        """Generate a single voice preview audio file.

        Uses the VoiceSynthesizer's OpenAI TTS client to generate audio.

        Args:
            text: The Vietnamese text to synthesize.
            voice: The OpenAI voice ID to use.
            output_path: Path where the MP3 file will be written.
        """
        # Use the synthesizer's OpenAI client for TTS
        response = self._synthesizer.client.audio.speech.create(
            model=self._synthesizer._model,
            voice=voice,
            input=text,
            response_format="mp3",
            speed=1.0,
        )
        response.stream_to_file(str(output_path))

    def _save_voice_options_metadata(
        self,
        previews_dir: Path,
        options: list[VoiceOption],
        preview_text: str,
    ) -> None:
        """Save voice_options.json metadata file in the previews directory.

        Args:
            previews_dir: Directory where previews are stored.
            options: List of successful VoiceOption objects.
            preview_text: The text used for preview generation.
        """
        metadata = {
            "preview_text": preview_text,
            "voice_options": [
                {
                    "voice_id": opt.voice_id,
                    "voice_name": opt.voice_name,
                    "preview_url": opt.preview_url,
                }
                for opt in options
            ],
        }
        metadata_path = previews_dir / "voice_options.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
