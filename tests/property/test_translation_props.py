"""Property-based tests for Chinese text validation.

Feature: douyin-video-translator, Property 5: Non-Chinese or Empty Text Validation

Validates: Requirements 4.4

For any input string that is empty, contains only whitespace, or contains no Chinese characters
(Unicode range \u4e00-\u9fff), the translator SHALL reject it with the error
"Không có nội dung tiếng Trung để dịch". Conversely, any string containing at least one
Chinese character SHALL pass this validation.
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.services.translator import Translator, EmptyTextError


# --- Strategies ---

# Chinese character strategy: generates a single char from CJK Unified Ideographs block
chinese_char = st.integers(min_value=0x4E00, max_value=0x9FFF).map(chr)

# Non-Chinese text: characters that are NOT in the CJK Unified Ideographs range
non_chinese_char = st.characters(
    blacklist_categories=("Cs",),  # exclude surrogates
    blacklist_characters="".join(chr(c) for c in range(0x4E00, 0x9FFF + 1)),
)

# Non-Chinese text string (at least 1 char, no Chinese characters)
non_chinese_text = st.text(
    alphabet=non_chinese_char,
    min_size=1,
).filter(lambda s: s.strip() != "")  # ensure not whitespace-only (tested separately)

# Text containing at least one Chinese character mixed with other text
text_with_chinese = st.builds(
    lambda prefix, ch, suffix: prefix + ch + suffix,
    prefix=st.text(alphabet=non_chinese_char, min_size=0, max_size=20),
    ch=chinese_char,
    suffix=st.text(alphabet=non_chinese_char, min_size=0, max_size=20),
)

# Whitespace-only strings
whitespace_only = st.text(
    alphabet=st.sampled_from([" ", "\t", "\n", "\r", "\f", "\v"]),
    min_size=1,
    max_size=20,
)


# --- Tests ---

@pytest.mark.property
class TestChineseTextValidation:
    """Property 5: Non-Chinese or Empty Text Validation.

    **Validates: Requirements 4.4**
    """

    def setup_method(self):
        """Create a Translator instance for validation testing."""
        self.translator = Translator()

    @given(text=non_chinese_text)
    @settings(max_examples=100)
    def test_non_chinese_text_raises_empty_text_error(self, text: str):
        """Feature: douyin-video-translator, Property 5: Non-Chinese or Empty Text Validation

        Strings with no Chinese characters are rejected with EmptyTextError.
        """
        with pytest.raises(EmptyTextError) as exc_info:
            self.translator._validate_chinese_content(text)
        assert exc_info.value.message == "Không có nội dung tiếng Trung để dịch"

    @given(text=text_with_chinese)
    @settings(max_examples=100)
    def test_text_with_chinese_passes_validation(self, text: str):
        """Feature: douyin-video-translator, Property 5: Non-Chinese or Empty Text Validation

        Strings containing at least one Chinese character pass validation without error.
        """
        # Should NOT raise any exception
        self.translator._validate_chinese_content(text)

    @given(text=st.just(""))
    @settings(max_examples=1)
    def test_empty_string_raises_empty_text_error(self, text: str):
        """Feature: douyin-video-translator, Property 5: Non-Chinese or Empty Text Validation

        Empty strings are rejected with EmptyTextError.
        """
        with pytest.raises(EmptyTextError) as exc_info:
            self.translator._validate_chinese_content(text)
        assert exc_info.value.message == "Không có nội dung tiếng Trung để dịch"

    @given(text=whitespace_only)
    @settings(max_examples=100)
    def test_whitespace_only_raises_empty_text_error(self, text: str):
        """Feature: douyin-video-translator, Property 5: Non-Chinese or Empty Text Validation

        Whitespace-only strings are rejected with EmptyTextError.
        """
        with pytest.raises(EmptyTextError) as exc_info:
            self.translator._validate_chinese_content(text)
        assert exc_info.value.message == "Không có nội dung tiếng Trung để dịch"


# =============================================================================
# Property 4: Translation Preserves Segment Structure and Timestamps
# =============================================================================

from unittest.mock import MagicMock, patch

from app.models.pipeline import (
    TranscriptionResult,
    TranscriptionSegment,
)


# --- Strategies for Property 4 ---

# Generate Chinese text using only CJK Unified Ideographs range (\u4e00-\u9fff)
chinese_segment_text = st.text(
    alphabet=st.sampled_from(
        [chr(c) for c in range(0x4E00, 0x4E00 + 100)]  # subset of CJK chars for efficiency
    ),
    min_size=1,
    max_size=50,
)

# Optional speaker labels for Property 4
prop4_speaker_labels = st.one_of(
    st.none(), st.sampled_from(["speaker_0", "speaker_1", "speaker_2"])
)

# Confidence scores between 0 and 1
prop4_confidence = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


@st.composite
def valid_transcription_with_segments(draw):
    """Strategy that generates a valid TranscriptionResult with Chinese text segments."""
    num_segments = draw(st.integers(min_value=1, max_value=20))
    segments = []
    current_time = draw(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    )

    for _ in range(num_segments):
        duration = draw(
            st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
        )
        start = current_time
        end = start + duration

        text = draw(chinese_segment_text)
        speaker = draw(prop4_speaker_labels)

        segments.append(
            TranscriptionSegment(
                start=start,
                end=end,
                text=text,
                speaker=speaker,
            )
        )

        gap = draw(
            st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
        )
        current_time = end + gap

    full_text = "".join(seg.text for seg in segments)

    return TranscriptionResult(
        segments=segments,
        full_text=full_text,
        language="zh",
        confidence=draw(prop4_confidence),
    )


# --- Tests for Property 4 ---

@pytest.mark.property
class TestTranslationSegmentPreservation:
    """Property 4: Translation Preserves Segment Structure and Timestamps.

    **Validates: Requirements 4.2**
    """

    @given(transcription=valid_transcription_with_segments())
    @settings(max_examples=100)
    def test_translation_preserves_segment_count(self, transcription: TranscriptionResult):
        """Feature: douyin-video-translator, Property 4: Translation Preserves Segment Structure and Timestamps

        The number of output segments must equal the number of input segments.
        """
        with patch("app.services.translator.translate.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.translate.return_value = {"translatedText": "bản dịch tiếng Việt"}
            mock_client_cls.return_value = mock_client

            translator = Translator(max_retries=1, backoff_base=1)
            translator._client = mock_client

            result = translator.translate(transcription)

            assert len(result.segments) == len(transcription.segments)

    @given(transcription=valid_transcription_with_segments())
    @settings(max_examples=100)
    def test_translation_preserves_timestamps(self, transcription: TranscriptionResult):
        """Feature: douyin-video-translator, Property 4: Translation Preserves Segment Structure and Timestamps

        For each segment i, start and end timestamps must be identical to the original.
        """
        with patch("app.services.translator.translate.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.translate.return_value = {"translatedText": "bản dịch tiếng Việt"}
            mock_client_cls.return_value = mock_client

            translator = Translator(max_retries=1, backoff_base=1)
            translator._client = mock_client

            result = translator.translate(transcription)

            for i, (original, translated) in enumerate(
                zip(transcription.segments, result.segments)
            ):
                assert translated.start == original.start, (
                    f"Segment {i}: start timestamp mismatch. "
                    f"Expected {original.start}, got {translated.start}"
                )
                assert translated.end == original.end, (
                    f"Segment {i}: end timestamp mismatch. "
                    f"Expected {original.end}, got {translated.end}"
                )
