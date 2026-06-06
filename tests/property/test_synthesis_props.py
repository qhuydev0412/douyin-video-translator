"""Property-based tests for voice synthesis.

Feature: douyin-video-translator, Property 6: TTS Duration Matches Target Segment Duration
Feature: douyin-video-translator, Property 7: Distinct Speakers Receive Distinct Voices

Validates: Requirements 5.2, 5.3, 5.4
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.services.voice_synthesizer import (
    DEFAULT_VOICE,
    MAX_SPEED_MULTIPLIER,
    VIETNAMESE_VOICES,
    VoiceSynthesizer,
)


# --- Strategies ---

# Generate unique speaker labels (non-empty strings)
speaker_label = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=20,
)


def distinct_speaker_sets(min_size: int = 1, max_size: int | None = None):
    """Strategy that generates sets of distinct speaker labels.

    Constrains max_size to the number of available voices by default,
    since the distinctness property only holds when N <= len(VIETNAMESE_VOICES).
    """
    if max_size is None:
        max_size = len(VIETNAMESE_VOICES)

    return st.lists(
        speaker_label,
        min_size=min_size,
        max_size=max_size,
        unique=True,
    )


# --- Tests ---


@pytest.mark.property
class TestDistinctSpeakersReceiveDistinctVoices:
    """Property 7: Distinct Speakers Receive Distinct Voices.

    **Validates: Requirements 5.4**
    """

    @given(speakers=distinct_speaker_sets(min_size=2, max_size=len(VIETNAMESE_VOICES)))
    @settings(max_examples=100)
    def test_distinct_speakers_get_distinct_voices(self, speakers: list[str]):
        """Feature: douyin-video-translator, Property 7: Distinct Speakers Receive Distinct Voices

        For any set of N distinct speaker labels (where N <= number of available voices),
        all N speakers receive distinct voice IDs.
        """
        synthesizer = VoiceSynthesizer()

        assigned_voices = [synthesizer.select_voice(speaker) for speaker in speakers]

        # All assigned voices must be unique
        assert len(set(assigned_voices)) == len(speakers), (
            f"Expected {len(speakers)} distinct voices for {len(speakers)} distinct speakers, "
            f"but got voices: {assigned_voices}"
        )

    @given(speaker=speaker_label, num_calls=st.integers(min_value=2, max_value=10))
    @settings(max_examples=100)
    def test_same_speaker_always_gets_same_voice(self, speaker: str, num_calls: int):
        """Feature: douyin-video-translator, Property 7: Distinct Speakers Receive Distinct Voices

        The same speaker always receives the same voice (idempotent mapping).
        """
        synthesizer = VoiceSynthesizer()

        voices = [synthesizer.select_voice(speaker) for _ in range(num_calls)]

        # All calls for same speaker must return the same voice
        assert all(v == voices[0] for v in voices), (
            f"Expected same voice for speaker '{speaker}' across {num_calls} calls, "
            f"but got: {voices}"
        )

    @given(data=st.data())
    @settings(max_examples=100)
    def test_none_speaker_always_returns_default_voice(self, data):
        """Feature: douyin-video-translator, Property 7: Distinct Speakers Receive Distinct Voices

        None speaker always returns DEFAULT_VOICE regardless of prior state.
        """
        synthesizer = VoiceSynthesizer()

        # Optionally assign some speakers first to alter internal state
        num_prior_speakers = data.draw(st.integers(min_value=0, max_value=5))
        for _ in range(num_prior_speakers):
            prior_speaker = data.draw(speaker_label)
            synthesizer.select_voice(prior_speaker)

        # None should always return DEFAULT_VOICE
        voice = synthesizer.select_voice(None)
        assert voice == DEFAULT_VOICE, (
            f"Expected DEFAULT_VOICE '{DEFAULT_VOICE}' for None speaker, but got '{voice}'"
        )

    @given(
        speakers=st.lists(
            speaker_label,
            min_size=len(VIETNAMESE_VOICES) + 1,
            max_size=len(VIETNAMESE_VOICES) + 5,
            unique=True,
        )
    )
    @settings(max_examples=100)
    def test_voices_cycle_when_speakers_exceed_available_voices(self, speakers: list[str]):
        """Feature: douyin-video-translator, Property 7: Distinct Speakers Receive Distinct Voices

        When N > len(VIETNAMESE_VOICES), voices cycle through available options.
        This verifies the cycling behavior is consistent and predictable.
        """
        synthesizer = VoiceSynthesizer()

        assigned_voices = [synthesizer.select_voice(speaker) for speaker in speakers]

        # Each voice should be from VIETNAMESE_VOICES
        for voice in assigned_voices:
            assert voice in VIETNAMESE_VOICES, (
                f"Voice '{voice}' not in available voices: {VIETNAMESE_VOICES}"
            )

        # Voice assignment follows round-robin pattern
        for i, speaker in enumerate(speakers):
            expected_voice = VIETNAMESE_VOICES[i % len(VIETNAMESE_VOICES)]
            assert assigned_voices[i] == expected_voice, (
                f"Speaker '{speaker}' at index {i} expected voice '{expected_voice}' "
                f"but got '{assigned_voices[i]}'"
            )



# --- Property 6 Strategies ---

# Target durations: realistic segment durations (0.5s to 30s)
target_durations = st.floats(min_value=0.5, max_value=30.0, allow_nan=False, allow_infinity=False)

# TTS durations: realistic audio output durations (0.3s to 60s)
tts_durations = st.floats(min_value=0.3, max_value=60.0, allow_nan=False, allow_infinity=False)

# Start times for segments
start_times = st.floats(min_value=0.0, max_value=300.0, allow_nan=False, allow_infinity=False)

# Vietnamese text samples
vietnamese_texts = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"), whitelist_characters="àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"),
    min_size=1,
    max_size=100,
)

# Voice IDs for Property 6
voices_p6 = st.sampled_from(["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"])


@st.composite
def tts_within_target(draw):
    """Strategy: TTS duration is less than or equal to target duration (no speed adjustment needed)."""
    target = draw(target_durations)
    # TTS duration fits within target
    actual_tts = draw(st.floats(min_value=0.1, max_value=target, allow_nan=False, allow_infinity=False))
    assume(actual_tts > 0)
    return target, actual_tts


@st.composite
def tts_exceeds_target_within_2x(draw):
    """Strategy: TTS duration exceeds target but multiplier is <= 2x."""
    target = draw(st.floats(min_value=0.5, max_value=30.0, allow_nan=False, allow_infinity=False))
    # Multiplier between 1.0 (exclusive) and 2.0 (inclusive)
    multiplier = draw(st.floats(min_value=1.01, max_value=2.0, allow_nan=False, allow_infinity=False))
    actual_tts = target * multiplier
    return target, actual_tts


@st.composite
def tts_exceeds_target_beyond_2x(draw):
    """Strategy: TTS duration exceeds target and multiplier is > 2x."""
    target = draw(st.floats(min_value=0.5, max_value=15.0, allow_nan=False, allow_infinity=False))
    # Multiplier above 2.0
    multiplier = draw(st.floats(min_value=2.01, max_value=4.0, allow_nan=False, allow_infinity=False))
    actual_tts = target * multiplier
    return target, actual_tts


# --- Property 6 Tests ---


@pytest.mark.property
class TestTTSDurationMatching:
    """Property 6: TTS Duration Matches Target Segment Duration.

    **Validates: Requirements 5.2, 5.3**
    """

    @given(data=tts_within_target(), text=vietnamese_texts, voice=voices_p6, start=start_times)
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_no_speed_adjustment_when_tts_fits_target(self, data, text, voice, start):
        """When TTS duration <= target duration, speed_adjusted should be False and output duration equals initial TTS duration.

        Feature: douyin-video-translator, Property 6: TTS Duration Matches Target Segment Duration
        """
        target_duration, actual_tts_duration = data
        end = start + target_duration
        output_path = Path("/tmp/test_segment.mp3")

        synthesizer = VoiceSynthesizer()

        mock_communicate_instance = AsyncMock()
        mock_communicate_instance.save = AsyncMock()

        with patch("app.services.voice_synthesizer.edge_tts.Communicate", return_value=mock_communicate_instance) as mock_communicate_cls, \
             patch.object(synthesizer, "_get_audio_duration", return_value=actual_tts_duration):

            result = await synthesizer._synthesize_segment(
                text=text,
                voice=voice,
                target_duration=target_duration,
                output_path=output_path,
                start=start,
                end=end,
            )

            # No speed adjustment needed
            assert result.speed_adjusted is False
            # Output duration equals the initial TTS duration
            assert result.duration == actual_tts_duration
            # edge_tts.Communicate called only once (no re-generation)
            assert mock_communicate_cls.call_count == 1

    @given(data=tts_exceeds_target_within_2x(), text=vietnamese_texts, voice=voices_p6, start=start_times)
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_speed_adjusted_when_exceeds_target_within_2x(self, data, text, voice, start):
        """When TTS duration > target and multiplier <= 2x, speed is adjusted with appropriate rate param.

        Feature: douyin-video-translator, Property 6: TTS Duration Matches Target Segment Duration
        """
        target_duration, actual_tts_duration = data
        end = start + target_duration
        output_path = Path("/tmp/test_segment.mp3")

        synthesizer = VoiceSynthesizer()

        # After speed adjustment, simulate the adjusted audio fitting target
        adjusted_duration = target_duration

        mock_communicate_instance = AsyncMock()
        mock_communicate_instance.save = AsyncMock()

        # _get_audio_duration returns actual_tts_duration first, then adjusted_duration after speed change
        duration_calls = [actual_tts_duration, adjusted_duration]
        call_idx = {"i": 0}

        def mock_get_duration(path):
            idx = call_idx["i"]
            call_idx["i"] += 1
            return duration_calls[idx]

        with patch("app.services.voice_synthesizer.edge_tts.Communicate", return_value=mock_communicate_instance) as mock_communicate_cls, \
             patch.object(synthesizer, "_get_audio_duration", side_effect=mock_get_duration):

            result = await synthesizer._synthesize_segment(
                text=text,
                voice=voice,
                target_duration=target_duration,
                output_path=output_path,
                start=start,
                end=end,
            )

            # Speed was adjusted
            assert result.speed_adjusted is True
            # edge_tts.Communicate called twice (initial + speed-adjusted)
            assert mock_communicate_cls.call_count == 2

            # Verify the rate parameter was set correctly
            second_call_kwargs = mock_communicate_cls.call_args_list[1]
            rate_arg = second_call_kwargs[1].get("rate") if second_call_kwargs[1] else None
            if rate_arg is None:
                rate_arg = second_call_kwargs[0][2] if len(second_call_kwargs[0]) > 2 else None

            # Calculate expected rate
            speed_multiplier = actual_tts_duration / target_duration
            expected_rate_percent = int((speed_multiplier - 1.0) * 100)
            expected_rate_str = f"+{expected_rate_percent}%"
            assert rate_arg == expected_rate_str

    @given(data=tts_exceeds_target_beyond_2x(), text=vietnamese_texts, voice=voices_p6, start=start_times)
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_speed_capped_at_2x_when_exceeds_max(self, data, text, voice, start):
        """When TTS duration > target and multiplier > 2x, rate is capped at "+100%" (max 2x speed).

        Feature: douyin-video-translator, Property 6: TTS Duration Matches Target Segment Duration
        """
        target_duration, actual_tts_duration = data
        end = start + target_duration
        output_path = Path("/tmp/test_segment.mp3")

        synthesizer = VoiceSynthesizer()

        # After max speed adjustment, audio is still longer than target but capped
        capped_duration = actual_tts_duration / MAX_SPEED_MULTIPLIER

        mock_communicate_instance = AsyncMock()
        mock_communicate_instance.save = AsyncMock()

        duration_calls = [actual_tts_duration, capped_duration]
        call_idx = {"i": 0}

        def mock_get_duration(path):
            idx = call_idx["i"]
            call_idx["i"] += 1
            return duration_calls[idx]

        with patch("app.services.voice_synthesizer.edge_tts.Communicate", return_value=mock_communicate_instance) as mock_communicate_cls, \
             patch.object(synthesizer, "_get_audio_duration", side_effect=mock_get_duration):

            result = await synthesizer._synthesize_segment(
                text=text,
                voice=voice,
                target_duration=target_duration,
                output_path=output_path,
                start=start,
                end=end,
            )

            # Speed was adjusted
            assert result.speed_adjusted is True
            # edge_tts.Communicate called twice (initial + capped speed)
            assert mock_communicate_cls.call_count == 2

            # Verify the rate is capped at "+100%" (2x max)
            second_call_kwargs = mock_communicate_cls.call_args_list[1]
            rate_arg = second_call_kwargs[1].get("rate") if second_call_kwargs[1] else None
            if rate_arg is None:
                rate_arg = second_call_kwargs[0][2] if len(second_call_kwargs[0]) > 2 else None

            assert rate_arg == "+100%"

    @given(
        target_duration=st.floats(min_value=0.5, max_value=30.0, allow_nan=False, allow_infinity=False),
        actual_tts_duration=tts_durations,
        text=vietnamese_texts,
        voice=voices_p6,
        start=start_times,
    )
    @settings(max_examples=100)
    @pytest.mark.asyncio
    async def test_speed_multiplier_never_exceeds_2x(self, target_duration, actual_tts_duration, text, voice, start):
        """The speed adjustment algorithm never applies a rate exceeding 2x ("+100%").

        Feature: douyin-video-translator, Property 6: TTS Duration Matches Target Segment Duration
        """
        assume(actual_tts_duration > 0)
        end = start + target_duration
        output_path = Path("/tmp/test_segment.mp3")

        synthesizer = VoiceSynthesizer()

        # Simulated post-adjustment duration
        if actual_tts_duration <= target_duration:
            post_duration = actual_tts_duration
        else:
            multiplier = actual_tts_duration / target_duration
            if multiplier <= MAX_SPEED_MULTIPLIER:
                post_duration = target_duration
            else:
                post_duration = actual_tts_duration / MAX_SPEED_MULTIPLIER

        mock_communicate_instance = AsyncMock()
        mock_communicate_instance.save = AsyncMock()

        duration_calls = [actual_tts_duration, post_duration]
        call_idx = {"i": 0}

        def mock_get_duration(path):
            idx = call_idx["i"]
            call_idx["i"] += 1
            if idx < len(duration_calls):
                return duration_calls[idx]
            return duration_calls[-1]

        with patch("app.services.voice_synthesizer.edge_tts.Communicate", return_value=mock_communicate_instance) as mock_communicate_cls, \
             patch.object(synthesizer, "_get_audio_duration", side_effect=mock_get_duration):

            result = await synthesizer._synthesize_segment(
                text=text,
                voice=voice,
                target_duration=target_duration,
                output_path=output_path,
                start=start,
                end=end,
            )

            # If speed was adjusted, verify the rate never exceeds "+100%"
            if mock_communicate_cls.call_count > 1:
                second_call_kwargs = mock_communicate_cls.call_args_list[1]
                rate_arg = second_call_kwargs[1].get("rate") if second_call_kwargs[1] else None
                if rate_arg is None:
                    rate_arg = second_call_kwargs[0][2] if len(second_call_kwargs[0]) > 2 else None

                # Parse the rate percentage
                assert rate_arg is not None
                assert rate_arg.startswith("+")
                assert rate_arg.endswith("%")
                rate_value = int(rate_arg[1:-1])
                # Rate should never exceed 100% (which corresponds to 2x speed)
                assert rate_value <= 100, f"Rate {rate_arg} exceeds maximum 2x speed (+100%)"
