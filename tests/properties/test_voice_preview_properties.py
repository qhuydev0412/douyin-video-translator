"""Property-based tests for voice preview segment selection.

Feature: pipeline-preview-confirm

Tests cover:
- Property 15: Preview segment selection uses longest under 15 seconds
"""

from unittest.mock import MagicMock

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from app.models.pipeline import TranslatedSegment
from app.services.voice_preview import (
    CHARS_PER_SECOND,
    MAX_PREVIEW_DURATION_SECONDS,
    VoicePreviewGenerator,
)

# Maximum character count for a segment to be within the 15s duration limit
MAX_CHARS = int(MAX_PREVIEW_DURATION_SECONDS * CHARS_PER_SECOND)  # 60


# --- Strategies ---

segment_strategy = st.builds(
    TranslatedSegment,
    start=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    end=st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    original_text=st.text(min_size=1, max_size=100),
    translated_text=st.text(min_size=1, max_size=200).filter(lambda t: t.strip()),
)


# --- Property Tests ---


class TestVoicePreviewSegmentSelection:
    """Property 15: Preview segment selection uses longest under 15 seconds.

    **Validates: Requirements 4.4**

    For any set of translated segments, the voice preview SHALL be generated
    using the segment with the longest text that does not exceed 15 seconds
    of estimated speech duration.
    """

    @given(segments=st.lists(segment_strategy, min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_selects_longest_segment_within_duration_limit(
        self, segments: list[TranslatedSegment]
    ) -> None:
        """The selected segment should be the longest non-empty text within the char limit.

        **Validates: Requirements 4.4**
        """
        generator = VoicePreviewGenerator(synthesizer=MagicMock())

        selected = generator.select_preview_segment(segments)

        # Determine which segments are within the duration limit
        within_limit = [
            s for s in segments
            if s.translated_text.strip() and len(s.translated_text.strip()) <= MAX_CHARS
        ]

        if within_limit:
            # The selected segment should be the one with the longest stripped text
            expected_length = max(len(s.translated_text.strip()) for s in within_limit)
            assert len(selected.translated_text.strip()) == expected_length
            assert len(selected.translated_text.strip()) <= MAX_CHARS
        else:
            # Fallback: all segments exceed the limit, pick the shortest non-empty one
            non_empty = [s for s in segments if s.translated_text.strip()]
            expected_length = min(len(s.translated_text.strip()) for s in non_empty)
            assert len(selected.translated_text.strip()) == expected_length

    @given(segments=st.lists(segment_strategy, min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_selected_segment_is_from_input(
        self, segments: list[TranslatedSegment]
    ) -> None:
        """The selected segment must be one of the input segments.

        **Validates: Requirements 4.4**
        """
        generator = VoicePreviewGenerator(synthesizer=MagicMock())

        selected = generator.select_preview_segment(segments)

        # The selected segment must exist in the original list
        assert selected in segments

    @given(segments=st.lists(segment_strategy, min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_selected_segment_has_non_empty_text(
        self, segments: list[TranslatedSegment]
    ) -> None:
        """The selected segment must have non-empty stripped translated text.

        **Validates: Requirements 4.4**
        """
        generator = VoicePreviewGenerator(synthesizer=MagicMock())

        selected = generator.select_preview_segment(segments)

        assert selected.translated_text.strip() != ""

    @given(
        segments=st.lists(
            st.builds(
                TranslatedSegment,
                start=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                end=st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
                original_text=st.text(min_size=1, max_size=100),
                translated_text=st.text(min_size=MAX_CHARS + 1, max_size=200).filter(lambda t: t.strip()),
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=100)
    def test_fallback_selects_shortest_when_all_exceed_limit(
        self, segments: list[TranslatedSegment]
    ) -> None:
        """When all segments exceed the duration limit, the shortest is selected as fallback.

        **Validates: Requirements 4.4**
        """
        # Ensure all segments actually exceed the limit after stripping
        assume(all(len(s.translated_text.strip()) > MAX_CHARS for s in segments))

        generator = VoicePreviewGenerator(synthesizer=MagicMock())

        selected = generator.select_preview_segment(segments)

        # Should be the shortest non-empty segment
        non_empty = [s for s in segments if s.translated_text.strip()]
        expected_length = min(len(s.translated_text.strip()) for s in non_empty)
        assert len(selected.translated_text.strip()) == expected_length
