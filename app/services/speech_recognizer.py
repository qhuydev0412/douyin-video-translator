"""Speech recognition service using OpenAI Whisper."""

from pathlib import Path

import whisper
import numpy as np

from app.models.pipeline import TranscriptionResult, TranscriptionSegment


class SpeechRecognitionError(Exception):
    """Raised when speech recognition fails."""


class SpeechRecognizer:
    """Transcribes Chinese speech to text with timestamps using Whisper.

    Uses OpenAI Whisper large-v3 model for high-accuracy Chinese speech
    recognition with segment-level timestamps.
    """

    def __init__(self, model_name: str = "large-v3") -> None:
        """Initialize the speech recognizer.

        Args:
            model_name: Whisper model to load. Defaults to large-v3.
        """
        self._model_name = model_name
        self._model: whisper.Whisper | None = None

    def _load_model(self) -> whisper.Whisper:
        """Load the Whisper model (lazy loading).

        Returns:
            Loaded Whisper model instance.

        Raises:
            SpeechRecognitionError: If model loading fails.
        """
        if self._model is None:
            try:
                self._model = whisper.load_model(self._model_name)
            except Exception as e:
                raise SpeechRecognitionError(
                    f"Failed to load Whisper model '{self._model_name}': {e}"
                )
        return self._model

    def recognize(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe Chinese speech to text with timestamps.

        Args:
            audio_path: Path to a WAV audio file containing speech.

        Returns:
            TranscriptionResult with segments, full text, language, and confidence.

        Raises:
            SpeechRecognitionError: If audio file not found, no speech detected,
                or transcription fails.
        """
        if not audio_path.exists():
            raise SpeechRecognitionError(f"Audio file not found: {audio_path}")

        model = self._load_model()

        try:
            result = model.transcribe(
                str(audio_path),
                language="zh",
                task="transcribe",
                fp16=False,
                verbose=False,
            )
        except Exception as e:
            raise SpeechRecognitionError(f"Transcription failed: {e}")

        segments_data = result.get("segments", [])

        if not segments_data:
            raise SpeechRecognitionError(
                "Không nhận dạng được giọng nói trong file âm thanh"
            )

        # Build transcription segments
        segments = self._build_segments(segments_data)

        # Validate we have meaningful content
        full_text = result.get("text", "").strip()
        if not full_text:
            raise SpeechRecognitionError(
                "Không nhận dạng được giọng nói trong file âm thanh"
            )

        # Calculate average confidence from segment no_speech_prob
        confidence = self._calculate_confidence(segments_data)

        # Detect language from result
        language = result.get("language", "zh")

        return TranscriptionResult(
            segments=segments,
            full_text=full_text,
            language=language,
            confidence=confidence,
        )

    def _build_segments(
        self, segments_data: list[dict]
    ) -> list[TranscriptionSegment]:
        """Build TranscriptionSegment list from Whisper output.

        Assigns speaker labels based on basic speaker change detection
        using pause gaps between segments.

        Args:
            segments_data: Raw segments from Whisper transcribe result.

        Returns:
            List of TranscriptionSegment with timestamps and speaker labels.
        """
        segments: list[TranscriptionSegment] = []
        current_speaker = "speaker_1"
        speaker_count = 1

        for i, seg in enumerate(segments_data):
            # Basic speaker change detection: if there's a significant gap
            # (> 2 seconds) between segments, assume a new speaker
            if i > 0:
                prev_end = segments_data[i - 1]["end"]
                current_start = seg["start"]
                gap = current_start - prev_end
                if gap > 2.0:
                    speaker_count += 1
                    current_speaker = f"speaker_{speaker_count}"

            segments.append(
                TranscriptionSegment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=seg["text"].strip(),
                    speaker=current_speaker,
                )
            )

        return segments

    def _calculate_confidence(self, segments_data: list[dict]) -> float:
        """Calculate overall confidence from segment probabilities.

        Uses (1 - no_speech_prob) as a proxy for segment confidence.

        Args:
            segments_data: Raw segments from Whisper transcribe result.

        Returns:
            Average confidence score between 0.0 and 1.0.
        """
        if not segments_data:
            return 0.0

        confidences = [
            1.0 - seg.get("no_speech_prob", 0.0) for seg in segments_data
        ]
        return float(np.mean(confidences))
