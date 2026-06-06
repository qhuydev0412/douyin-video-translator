"""Unit tests for the Translator service."""

from unittest.mock import MagicMock, patch

import pytest

from app.models.pipeline import (
    TranscriptionResult,
    TranscriptionSegment,
    TranslatedSegment,
    TranslationResult,
)
from app.services.translator import EmptyTextError, TranslationError, Translator


@pytest.fixture
def translator():
    """Create a Translator instance with mocked client."""
    t = Translator(max_retries=3, backoff_base=2)
    return t


@pytest.fixture
def sample_transcription():
    """Create a sample TranscriptionResult for testing."""
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(start=0.0, end=2.5, text="你好世界", speaker="speaker_1"),
            TranscriptionSegment(start=3.0, end=5.0, text="今天天气很好", speaker="speaker_1"),
        ],
        full_text="你好世界。今天天气很好。",
        language="zh",
        confidence=0.95,
    )


class TestChineseTextValidation:
    """Tests for Chinese text detection and validation."""

    def test_contains_chinese_with_chinese_text(self, translator):
        assert translator._contains_chinese("你好世界") is True

    def test_contains_chinese_with_mixed_text(self, translator):
        assert translator._contains_chinese("Hello 你好") is True

    def test_contains_chinese_with_english_only(self, translator):
        assert translator._contains_chinese("Hello World") is False

    def test_contains_chinese_with_empty_string(self, translator):
        assert translator._contains_chinese("") is False

    def test_contains_chinese_with_numbers(self, translator):
        assert translator._contains_chinese("12345") is False

    def test_contains_chinese_with_single_char(self, translator):
        assert translator._contains_chinese("中") is True

    def test_validate_raises_on_empty_text(self, translator):
        with pytest.raises(EmptyTextError) as exc_info:
            translator._validate_chinese_content("")
        assert exc_info.value.message == "Không có nội dung tiếng Trung để dịch"

    def test_validate_raises_on_whitespace_only(self, translator):
        with pytest.raises(EmptyTextError):
            translator._validate_chinese_content("   \n\t  ")

    def test_validate_raises_on_non_chinese_text(self, translator):
        with pytest.raises(EmptyTextError):
            translator._validate_chinese_content("Hello World, no Chinese here!")

    def test_validate_passes_with_chinese_text(self, translator):
        # Should not raise
        translator._validate_chinese_content("这是中文")


class TestTranslatePreservesTimestamps:
    """Tests that translation preserves segment structure and timestamps."""

    @patch("app.services.translator.translate.Client")
    def test_preserves_segment_count(self, mock_client_cls, sample_transcription):
        mock_client = MagicMock()
        mock_client.translate.return_value = {"translatedText": "Xin chào thế giới"}
        mock_client_cls.return_value = mock_client

        translator = Translator(max_retries=3, backoff_base=2)
        translator._client = mock_client

        result = translator.translate(sample_transcription)

        assert len(result.segments) == len(sample_transcription.segments)

    @patch("app.services.translator.translate.Client")
    def test_preserves_timestamps(self, mock_client_cls, sample_transcription):
        mock_client = MagicMock()
        mock_client.translate.return_value = {"translatedText": "Xin chào"}
        mock_client_cls.return_value = mock_client

        translator = Translator(max_retries=3, backoff_base=2)
        translator._client = mock_client

        result = translator.translate(sample_transcription)

        for i, segment in enumerate(result.segments):
            assert segment.start == sample_transcription.segments[i].start
            assert segment.end == sample_transcription.segments[i].end

    @patch("app.services.translator.translate.Client")
    def test_preserves_speaker_info(self, mock_client_cls, sample_transcription):
        mock_client = MagicMock()
        mock_client.translate.return_value = {"translatedText": "Xin chào"}
        mock_client_cls.return_value = mock_client

        translator = Translator(max_retries=3, backoff_base=2)
        translator._client = mock_client

        result = translator.translate(sample_transcription)

        for i, segment in enumerate(result.segments):
            assert segment.speaker == sample_transcription.segments[i].speaker

    @patch("app.services.translator.translate.Client")
    def test_stores_original_text(self, mock_client_cls, sample_transcription):
        mock_client = MagicMock()
        mock_client.translate.return_value = {"translatedText": "Xin chào"}
        mock_client_cls.return_value = mock_client

        translator = Translator(max_retries=3, backoff_base=2)
        translator._client = mock_client

        result = translator.translate(sample_transcription)

        for i, segment in enumerate(result.segments):
            assert segment.original_text == sample_transcription.segments[i].text


class TestTranslationErrorHandling:
    """Tests for error handling in translation."""

    def test_empty_text_raises_empty_text_error(self, translator):
        transcription = TranscriptionResult(
            segments=[],
            full_text="",
            language="zh",
            confidence=0.0,
        )
        with pytest.raises(EmptyTextError) as exc_info:
            translator.translate(transcription)
        assert exc_info.value.message == "Không có nội dung tiếng Trung để dịch"
        assert exc_info.value.retryable is False

    def test_non_chinese_text_raises_empty_text_error(self, translator):
        transcription = TranscriptionResult(
            segments=[
                TranscriptionSegment(start=0.0, end=1.0, text="Hello"),
            ],
            full_text="Hello World",
            language="en",
            confidence=0.9,
        )
        with pytest.raises(EmptyTextError):
            translator.translate(transcription)

    @patch("app.services.translator.time.sleep")
    def test_api_failure_retries_then_raises(self, mock_sleep):
        mock_client = MagicMock()
        mock_client.translate.side_effect = Exception("API Error")

        translator = Translator(max_retries=3, backoff_base=2)
        translator._client = mock_client

        transcription = TranscriptionResult(
            segments=[
                TranscriptionSegment(start=0.0, end=1.0, text="你好"),
            ],
            full_text="你好",
            language="zh",
            confidence=0.9,
        )

        with pytest.raises(TranslationError) as exc_info:
            translator.translate(transcription)
        assert exc_info.value.retryable is True
        assert "3 lần thử" in exc_info.value.message

    @patch("app.services.translator.time.sleep")
    def test_retry_uses_exponential_backoff(self, mock_sleep):
        mock_client = MagicMock()
        mock_client.translate.side_effect = Exception("API Error")

        translator = Translator(max_retries=3, backoff_base=2)
        translator._client = mock_client

        transcription = TranscriptionResult(
            segments=[
                TranscriptionSegment(start=0.0, end=1.0, text="你好"),
            ],
            full_text="你好",
            language="zh",
            confidence=0.9,
        )

        with pytest.raises(TranslationError):
            translator.translate(transcription)

        # Backoff: 2^1=2, 2^2=4 (only 2 sleeps for 3 attempts)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(2)
        mock_sleep.assert_any_call(4)

    @patch("app.services.translator.time.sleep")
    def test_succeeds_on_retry(self, mock_sleep):
        mock_client = MagicMock()
        # Fail first, succeed second
        mock_client.translate.side_effect = [
            Exception("Temporary failure"),
            {"translatedText": "Xin chào"},
            {"translatedText": "Xin chào thế giới"},
        ]

        translator = Translator(max_retries=3, backoff_base=2)
        translator._client = mock_client

        transcription = TranscriptionResult(
            segments=[
                TranscriptionSegment(start=0.0, end=1.0, text="你好"),
            ],
            full_text="你好世界",
            language="zh",
            confidence=0.9,
        )

        result = translator.translate(transcription)
        assert result.segments[0].translated_text == "Xin chào"


class TestTranslationResult:
    """Tests for the overall TranslationResult structure."""

    @patch("app.services.translator.translate.Client")
    def test_full_text_translated(self, mock_client_cls, sample_transcription):
        mock_client = MagicMock()
        mock_client.translate.side_effect = [
            {"translatedText": "Xin chào thế giới"},
            {"translatedText": "Hôm nay thời tiết đẹp"},
            {"translatedText": "Xin chào thế giới. Hôm nay thời tiết đẹp."},
        ]
        mock_client_cls.return_value = mock_client

        translator = Translator(max_retries=3, backoff_base=2)
        translator._client = mock_client

        result = translator.translate(sample_transcription)

        assert result.full_text_original == sample_transcription.full_text
        assert result.full_text_translated == "Xin chào thế giới. Hôm nay thời tiết đẹp."

    @patch("app.services.translator.translate.Client")
    def test_result_type(self, mock_client_cls, sample_transcription):
        mock_client = MagicMock()
        mock_client.translate.return_value = {"translatedText": "Xin chào"}
        mock_client_cls.return_value = mock_client

        translator = Translator(max_retries=3, backoff_base=2)
        translator._client = mock_client

        result = translator.translate(sample_transcription)

        assert isinstance(result, TranslationResult)
        for seg in result.segments:
            assert isinstance(seg, TranslatedSegment)
