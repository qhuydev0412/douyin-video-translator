"""Speech recognition service using OpenAI Whisper API."""

import os
import logging
from pathlib import Path

from openai import OpenAI

from app.models.pipeline import TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)


def validate_segment_ordering(result: TranscriptionResult) -> bool:
    """Validate that transcription segments maintain temporal ordering."""
    for segment in result.segments:
        if segment.start >= segment.end:
            return False
    for i in range(len(result.segments) - 1):
        if result.segments[i].end > result.segments[i + 1].start:
            return False
    return True


class SpeechRecognitionError(Exception):
    """Raised when speech recognition fails."""


class SpeechRecognizer:
    """Speech recognition using OpenAI Whisper API.

    Much more accurate than local Whisper, especially for Chinese.
    No model loading needed — runs on OpenAI's GPU servers.
    Cost: ~$0.006/minute of audio.
    """

    def __init__(self, model_name: str = "whisper-1") -> None:
        self._model_name = model_name
        self._client: OpenAI | None = None
        # Keep _model attribute for compatibility with web_ui preloading
        self._model = "api"

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._client

    def recognize(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe Chinese speech using OpenAI Whisper API.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)

        Returns:
            TranscriptionResult with timestamped segments.

        Raises:
            SpeechRecognitionError: If transcription fails.
        """
        if not audio_path.exists():
            raise SpeechRecognitionError(f"Audio file not found: {audio_path}")

        try:
            with open(audio_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=self._model_name,
                    file=audio_file,
                    language="zh",
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
        except Exception as e:
            raise SpeechRecognitionError(f"Transcription failed: {e}")

        # Parse response
        segments_data = response.segments or []

        if not segments_data:
            raise SpeechRecognitionError(
                "Không nhận dạng được giọng nói trong file âm thanh"
            )

        full_text = response.text.strip() if response.text else ""
        if not full_text:
            raise SpeechRecognitionError(
                "Không nhận dạng được giọng nói trong file âm thanh"
            )

        # Build segments
        segments = []
        for seg in segments_data:
            text = seg.get("text", "").strip() if isinstance(seg, dict) else getattr(seg, "text", "").strip()
            start = seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0)
            end = seg.get("end", 0.0) if isinstance(seg, dict) else getattr(seg, "end", 0.0)

            if text:
                segments.append(
                    TranscriptionSegment(
                        start=float(start),
                        end=float(end),
                        text=text,
                        speaker="speaker_1",
                    )
                )

        if not segments:
            raise SpeechRecognitionError(
                "Không nhận dạng được giọng nói trong file âm thanh"
            )

        # Calculate confidence (API doesn't return this, use 0.95 as default)
        confidence = 0.95

        logger.info(
            "Transcribed %d segments, total text: %d chars",
            len(segments),
            len(full_text),
        )

        return TranscriptionResult(
            segments=segments,
            full_text=full_text,
            language="zh",
            confidence=confidence,
        )
