"""Property-based tests for transcription segment ordering.

Feature: douyin-video-translator, Property 3: Transcription Segments Temporal Ordering

Validates: Requirements 3.2

For any transcription result produced by the speech recognizer, all segments SHALL have
start < end timestamps, segments SHALL be in chronological order
(segment[i].end <= segment[i+1].start), and no two segments SHALL overlap in time.
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.models.pipeline import TranscriptionResult, TranscriptionSegment
from app.services.speech_recognizer import validate_segment_ordering


# --- Strategies ---

# Generate Chinese-like text for segments
segment_text = st.text(
    alphabet=st.characters(whitelist_categories=("Lo",), whitelist_characters="，。！？"),
    min_size=1,
    max_size=50,
)

# Optional speaker labels
speaker_labels = st.one_of(st.none(), st.sampled_from(["speaker_0", "speaker_1", "speaker_2"]))

# Language codes
languages = st.sampled_from(["zh", "zh-cn", "zh-tw"])

# Confidence scores between 0 and 1
confidence_scores = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


def well_ordered_segments(min_segments: int = 0, max_segments: int = 20):
    """Strategy that generates well-ordered transcription segments.

    Segments have start < end and are in chronological non-overlapping order.
    """
    @st.composite
    def strategy(draw):
        num_segments = draw(st.integers(min_value=min_segments, max_value=max_segments))
        segments = []
        current_time = draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))

        for _ in range(num_segments):
            # Duration of this segment (positive)
            duration = draw(st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False))
            start = current_time
            end = start + duration

            text = draw(segment_text)
            speaker = draw(speaker_labels)

            segments.append(TranscriptionSegment(
                start=start,
                end=end,
                text=text,
                speaker=speaker,
            ))

            # Gap between segments (>= 0 to ensure non-overlapping)
            gap = draw(st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False))
            current_time = end + gap

        return segments

    return strategy()


def valid_transcription_result():
    """Strategy that generates a valid TranscriptionResult with well-ordered segments."""
    @st.composite
    def strategy(draw):
        segments = draw(well_ordered_segments(min_segments=0, max_segments=20))
        full_text = " ".join(seg.text for seg in segments) if segments else ""
        language = draw(languages)
        confidence = draw(confidence_scores)

        return TranscriptionResult(
            segments=segments,
            full_text=full_text,
            language=language,
            confidence=confidence,
        )

    return strategy()


def segments_with_invalid_duration():
    """Strategy that generates segments where at least one has start >= end."""
    @st.composite
    def strategy(draw):
        # Generate some well-ordered segments first
        good_segments = draw(well_ordered_segments(min_segments=1, max_segments=10))

        # Pick a random segment to corrupt
        idx = draw(st.integers(min_value=0, max_value=len(good_segments) - 1))
        bad_segment = good_segments[idx]

        # Make start >= end (either equal or reversed)
        if draw(st.booleans()):
            # start == end
            good_segments[idx] = TranscriptionSegment(
                start=bad_segment.start,
                end=bad_segment.start,
                text=bad_segment.text,
                speaker=bad_segment.speaker,
            )
        else:
            # start > end (reversed)
            good_segments[idx] = TranscriptionSegment(
                start=bad_segment.end,
                end=bad_segment.start,
                text=bad_segment.text,
                speaker=bad_segment.speaker,
            )

        full_text = " ".join(seg.text for seg in good_segments)
        language = draw(languages)
        confidence = draw(confidence_scores)

        return TranscriptionResult(
            segments=good_segments,
            full_text=full_text,
            language=language,
            confidence=confidence,
        )

    return strategy()


def segments_with_overlap():
    """Strategy that generates segments where at least two overlap in time."""
    @st.composite
    def strategy(draw):
        # Generate at least 2 well-ordered segments
        good_segments = draw(well_ordered_segments(min_segments=2, max_segments=10))

        # Pick a pair of adjacent segments to create overlap
        idx = draw(st.integers(min_value=0, max_value=len(good_segments) - 2))
        current = good_segments[idx]
        next_seg = good_segments[idx + 1]

        # Make current segment end after next segment starts (create overlap)
        overlap_amount = draw(st.floats(
            min_value=0.01,
            max_value=max(0.02, (next_seg.end - next_seg.start) / 2),
            allow_nan=False,
            allow_infinity=False,
        ))
        new_end = next_seg.start + overlap_amount

        good_segments[idx] = TranscriptionSegment(
            start=current.start,
            end=new_end,
            text=current.text,
            speaker=current.speaker,
        )

        full_text = " ".join(seg.text for seg in good_segments)
        language = draw(languages)
        confidence = draw(confidence_scores)

        return TranscriptionResult(
            segments=good_segments,
            full_text=full_text,
            language=language,
            confidence=confidence,
        )

    return strategy()


def segments_out_of_order():
    """Strategy that generates segments that are not in chronological order."""
    @st.composite
    def strategy(draw):
        # Generate at least 2 well-ordered segments
        good_segments = draw(well_ordered_segments(min_segments=3, max_segments=10))

        # Swap two non-adjacent segments to break ordering
        idx1 = draw(st.integers(min_value=0, max_value=len(good_segments) - 2))
        idx2 = draw(st.integers(min_value=idx1 + 1, max_value=len(good_segments) - 1))

        good_segments[idx1], good_segments[idx2] = good_segments[idx2], good_segments[idx1]

        full_text = " ".join(seg.text for seg in good_segments)
        language = draw(languages)
        confidence = draw(confidence_scores)

        return TranscriptionResult(
            segments=good_segments,
            full_text=full_text,
            language=language,
            confidence=confidence,
        )

    return strategy()


# --- Tests ---

@pytest.mark.property
class TestTranscriptionSegmentOrdering:
    """Property 3: Transcription Segments Temporal Ordering.

    **Validates: Requirements 3.2**
    """

    @given(result=valid_transcription_result())
    @settings(max_examples=100)
    def test_well_ordered_segments_pass_validation(self, result: TranscriptionResult):
        """Well-ordered segments (start < end, chronological, non-overlapping) pass validation."""
        assert validate_segment_ordering(result) is True

    @given(result=segments_with_invalid_duration())
    @settings(max_examples=100)
    def test_segments_with_invalid_duration_fail_validation(self, result: TranscriptionResult):
        """Segments where start >= end are rejected by the validator."""
        assert validate_segment_ordering(result) is False

    @given(result=segments_with_overlap())
    @settings(max_examples=100)
    def test_overlapping_segments_fail_validation(self, result: TranscriptionResult):
        """Segments that overlap in time are rejected by the validator."""
        assert validate_segment_ordering(result) is False

    @given(result=segments_out_of_order())
    @settings(max_examples=100)
    def test_out_of_order_segments_fail_validation(self, result: TranscriptionResult):
        """Segments not in chronological order are rejected by the validator."""
        assert validate_segment_ordering(result) is False

    @given(
        language=languages,
        confidence=confidence_scores,
    )
    @settings(max_examples=100)
    def test_empty_segments_pass_validation(self, language: str, confidence: float):
        """An empty segment list is trivially valid (no ordering to violate)."""
        result = TranscriptionResult(
            segments=[],
            full_text="",
            language=language,
            confidence=confidence,
        )
        assert validate_segment_ordering(result) is True
