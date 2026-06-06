"""Translation service using deep-translator (free, no API key needed)."""

import re
import time

from deep_translator import GoogleTranslator

from app.core.config import settings
from app.models.pipeline import (
    TranscriptionResult,
    TranslatedSegment,
    TranslationResult,
)


class TranslationError(Exception):
    """Base exception for translation errors."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.retryable = retryable


class EmptyTextError(TranslationError):
    """Raised when input text is empty or contains no Chinese content."""

    def __init__(self):
        super().__init__(
            message="Không có nội dung tiếng Trung để dịch",
            retryable=False,
        )


class Translator:
    """Translates Chinese text segments to Vietnamese using deep-translator (Google Translate free)."""

    def __init__(
        self,
        max_retries: int | None = None,
        backoff_base: int | None = None,
    ):
        self.max_retries = max_retries if max_retries is not None else settings.MAX_RETRY_ATTEMPTS
        self.backoff_base = backoff_base if backoff_base is not None else settings.RETRY_BACKOFF_BASE
        self._client: GoogleTranslator | None = None

    @property
    def client(self) -> GoogleTranslator:
        """Lazy-initialize the translator client."""
        if self._client is None:
            self._client = GoogleTranslator(source="zh-CN", target="vi")
        return self._client

    def translate(self, transcription: TranscriptionResult) -> TranslationResult:
        """Translate Chinese text segments to Vietnamese.

        Args:
            transcription: TranscriptionResult with Chinese text segments.

        Returns:
            TranslationResult with translated Vietnamese segments preserving timestamps.

        Raises:
            EmptyTextError: If text is empty or contains no Chinese characters.
            TranslationError: If the translation fails after retries.
        """
        self._validate_chinese_content(transcription.full_text)

        translated_segments: list[TranslatedSegment] = []

        for segment in transcription.segments:
            translated_text = self._translate_text_with_retry(segment.text)
            translated_segments.append(
                TranslatedSegment(
                    start=segment.start,
                    end=segment.end,
                    original_text=segment.text,
                    translated_text=translated_text,
                    speaker=segment.speaker,
                )
            )

        full_text_translated = self._translate_text_with_retry(transcription.full_text)

        return TranslationResult(
            segments=translated_segments,
            full_text_original=transcription.full_text,
            full_text_translated=full_text_translated,
        )

    def _validate_chinese_content(self, text: str) -> None:
        """Validate that text contains Chinese characters.

        Args:
            text: Input text to validate.

        Raises:
            EmptyTextError: If text is empty, whitespace-only, or has no Chinese chars.
        """
        if not text or not text.strip():
            raise EmptyTextError()

        if not self._contains_chinese(text):
            raise EmptyTextError()

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        """Check if text contains at least one Chinese character (CJK Unified Ideographs)."""
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def _translate_text_with_retry(self, text: str) -> str:
        """Translate a single text string with exponential backoff retry.

        Args:
            text: Chinese text to translate.

        Returns:
            Translated Vietnamese text.

        Raises:
            TranslationError: If all retry attempts fail.
        """
        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                result = self.client.translate(text)
                return result
            except Exception as e:
                last_exception = e

                # Apply exponential backoff before next retry
                if attempt < self.max_retries - 1:
                    backoff_time = self.backoff_base ** (attempt + 1)
                    time.sleep(backoff_time)

        raise TranslationError(
            message=f"Lỗi dịch thuật sau {self.max_retries} lần thử: {last_exception}",
            retryable=True,
        )
