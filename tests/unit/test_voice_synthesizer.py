"""Unit tests for the VoiceSynthesizer service."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.pipeline import (
    SegmentAudio,
    SynthesisResult,
    TranslatedSegment,
    TranslationResult,
)
from app.services.voice_synthesizer import (
    DEFAULT_VOICE,
    MAX_SPEED_MULTIPLIER,
    VIETNAMESE_VOICES,
    VoiceSynthesizer,
    VoiceSynthesizerError,
)


@pytest.fixture
def synthesizer() -> VoiceSynthesizer:
    """Create a VoiceSynthesizer instance."""
    return VoiceSynthesizer()


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create output directory for synthesis results."""
    out = tmp_path / "tts_output"
    out.mkdir()
    return out


@pytest.fixture
def sample_translation() -> TranslationResult:
    """Create a sample translation result for testing."""
    return TranslationResult(
        segments=[
            TranslatedSegment(
                start=0.0,
                end=3.0,
                original_text="你好世界",
                translated_text="Xin chào thế giới",
                speaker="speaker_1",
            ),
            TranslatedSegment(
                start=3.5,
                end=6.0,
                original_text="谢谢大家",
                translated_text="Cảm ơn mọi người",
                speaker="speaker_2",
            ),
        ],
        full_text_original="你好世界 谢谢大家",
        full_text_translated="Xin chào thế giới Cảm ơn mọi người",
    )


class TestSelectVoice:
    """Tests for VoiceSynthesizer.select_voice method."""

    def test_none_speaker_returns_default_voice(
        self, synthesizer: VoiceSynthesizer
    ) -> None:
        """Test that None speaker always returns the default voice."""
        voice = synthesizer.select_voice(None)
        assert voice == DEFAULT_VOICE

    def test_distinct_speakers_get_distinct_voices(
        self, synthesizer: VoiceSynthesizer
    ) -> None:
        """Test that different speakers are assigned different voices."""
        voice1 = synthesizer.select_voice("speaker_1")
        voice2 = synthesizer.select_voice("speaker_2")
        assert voice1 != voice2

    def test_same_speaker_gets_same_voice(
        self, synthesizer: VoiceSynthesizer
    ) -> None:
        """Test that the same speaker always returns the same voice."""
        voice1 = synthesizer.select_voice("speaker_A")
        voice2 = synthesizer.select_voice("speaker_A")
        assert voice1 == voice2

    def test_voices_cycle_when_more_speakers_than_voices(
        self, synthesizer: VoiceSynthesizer
    ) -> None:
        """Test that voices cycle when there are more speakers than available voices."""
        voices = []
        for i in range(len(VIETNAMESE_VOICES) + 1):
            voice = synthesizer.select_voice(f"speaker_{i}")
            voices.append(voice)

        # The voice should cycle back to the first one
        assert voices[-1] == voices[0]

    def test_assigned_voices_are_valid_vietnamese_voices(
        self, synthesizer: VoiceSynthesizer
    ) -> None:
        """Test that all assigned voices are from the VIETNAMESE_VOICES list."""
        voice = synthesizer.select_voice("some_speaker")
        assert voice in VIETNAMESE_VOICES


class TestGetAudioDuration:
    """Tests for VoiceSynthesizer._get_audio_duration method."""

    def test_successful_duration_retrieval(
        self, synthesizer: VoiceSynthesizer, tmp_path: Path
    ) -> None:
        """Test successful audio duration retrieval via ffprobe."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        ffprobe_output = json.dumps({"format": {"duration": "2.5"}})
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ffprobe_output
        mock_result.stderr = ""

        with patch(
            "app.services.voice_synthesizer.subprocess.run", return_value=mock_result
        ):
            duration = synthesizer._get_audio_duration(audio_file)

        assert duration == 2.5

    def test_ffprobe_not_found_raises_error(
        self, synthesizer: VoiceSynthesizer, tmp_path: Path
    ) -> None:
        """Test that missing ffprobe raises VoiceSynthesizerError."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        with patch(
            "app.services.voice_synthesizer.subprocess.run",
            side_effect=FileNotFoundError(),
        ):
            with pytest.raises(VoiceSynthesizerError, match="ffprobe not found"):
                synthesizer._get_audio_duration(audio_file)

    def test_ffprobe_timeout_raises_error(
        self, synthesizer: VoiceSynthesizer, tmp_path: Path
    ) -> None:
        """Test that ffprobe timeout raises VoiceSynthesizerError."""
        import subprocess

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        with patch(
            "app.services.voice_synthesizer.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=30),
        ):
            with pytest.raises(VoiceSynthesizerError, match="timed out"):
                synthesizer._get_audio_duration(audio_file)

    def test_ffprobe_failure_raises_error(
        self, synthesizer: VoiceSynthesizer, tmp_path: Path
    ) -> None:
        """Test that ffprobe non-zero exit raises VoiceSynthesizerError."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Invalid data found"

        with patch(
            "app.services.voice_synthesizer.subprocess.run", return_value=mock_result
        ):
            with pytest.raises(VoiceSynthesizerError, match="ffprobe failed"):
                synthesizer._get_audio_duration(audio_file)

    def test_invalid_json_output_raises_error(
        self, synthesizer: VoiceSynthesizer, tmp_path: Path
    ) -> None:
        """Test that invalid JSON from ffprobe raises VoiceSynthesizerError."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "not json"

        with patch(
            "app.services.voice_synthesizer.subprocess.run", return_value=mock_result
        ):
            with pytest.raises(VoiceSynthesizerError, match="Failed to parse"):
                synthesizer._get_audio_duration(audio_file)


class TestSynthesizeSegment:
    """Tests for VoiceSynthesizer._synthesize_segment method."""

    @pytest.mark.asyncio
    async def test_normal_speed_when_within_target(
        self, synthesizer: VoiceSynthesizer, tmp_path: Path
    ) -> None:
        """Test no speed adjustment when TTS duration is within target."""
        output_path = tmp_path / "segment_0000.mp3"

        with (
            patch("app.services.voice_synthesizer.edge_tts.Communicate") as mock_comm_cls,
            patch.object(
                synthesizer, "_get_audio_duration", return_value=2.5
            ),
        ):
            mock_comm = MagicMock()
            mock_comm.save = AsyncMock()
            mock_comm_cls.return_value = mock_comm

            # Create fake file
            output_path.write_bytes(b"audio")

            result = await synthesizer._synthesize_segment(
                text="Xin chào",
                voice=DEFAULT_VOICE,
                target_duration=3.0,
                output_path=output_path,
                start=0.0,
                end=3.0,
            )

        assert result.speed_adjusted is False
        assert result.duration == 2.5
        assert result.target_duration == 3.0
        # Should only call Communicate once (no speed adjustment needed)
        assert mock_comm_cls.call_count == 1

    @pytest.mark.asyncio
    async def test_speed_adjustment_when_exceeds_target(
        self, synthesizer: VoiceSynthesizer, tmp_path: Path
    ) -> None:
        """Test speed is adjusted when TTS duration exceeds target duration."""
        output_path = tmp_path / "segment_0000.mp3"

        # First call returns 4.0s, second call (with speed) returns 2.8s
        duration_calls = [4.0, 2.8]
        duration_iter = iter(duration_calls)

        with (
            patch("app.services.voice_synthesizer.edge_tts.Communicate") as mock_comm_cls,
            patch.object(
                synthesizer,
                "_get_audio_duration",
                side_effect=lambda _: next(duration_iter),
            ),
        ):
            mock_comm = MagicMock()
            mock_comm.save = AsyncMock()
            mock_comm_cls.return_value = mock_comm

            output_path.write_bytes(b"audio")

            result = await synthesizer._synthesize_segment(
                text="Xin chào thế giới",
                voice=DEFAULT_VOICE,
                target_duration=3.0,
                output_path=output_path,
                start=0.0,
                end=3.0,
            )

        assert result.speed_adjusted is True
        assert result.duration == 2.8
        # Communicate should be called twice (initial + speed-adjusted)
        assert mock_comm_cls.call_count == 2
        # Second call should have a rate parameter
        second_call_kwargs = mock_comm_cls.call_args_list[1]
        assert "rate" in second_call_kwargs.kwargs or len(second_call_kwargs.args) > 2

    @pytest.mark.asyncio
    async def test_max_speed_cap_at_2x(
        self, synthesizer: VoiceSynthesizer, tmp_path: Path
    ) -> None:
        """Test that speed adjustment is capped at 2x (rate=+100%)."""
        output_path = tmp_path / "segment_0000.mp3"

        # Duration is 3x target — exceeds max 2x multiplier
        duration_calls = [9.0, 5.0]
        duration_iter = iter(duration_calls)

        with (
            patch("app.services.voice_synthesizer.edge_tts.Communicate") as mock_comm_cls,
            patch.object(
                synthesizer,
                "_get_audio_duration",
                side_effect=lambda _: next(duration_iter),
            ),
        ):
            mock_comm = MagicMock()
            mock_comm.save = AsyncMock()
            mock_comm_cls.return_value = mock_comm

            output_path.write_bytes(b"audio")

            result = await synthesizer._synthesize_segment(
                text="Một đoạn văn bản rất dài",
                voice=DEFAULT_VOICE,
                target_duration=3.0,
                output_path=output_path,
                start=0.0,
                end=3.0,
            )

        assert result.speed_adjusted is True
        # Verify the rate was capped at +100% (2x)
        second_call = mock_comm_cls.call_args_list[1]
        assert second_call.kwargs.get("rate") == "+100%"


    @pytest.mark.asyncio
    async def test_segment_audio_preserves_timing(
        self, synthesizer: VoiceSynthesizer, tmp_path: Path
    ) -> None:
        """Test that synthesized segment retains correct start/end timestamps."""
        output_path = tmp_path / "segment_0000.mp3"

        with (
            patch("app.services.voice_synthesizer.edge_tts.Communicate") as mock_comm_cls,
            patch.object(synthesizer, "_get_audio_duration", return_value=1.5),
        ):
            mock_comm = MagicMock()
            mock_comm.save = AsyncMock()
            mock_comm_cls.return_value = mock_comm
            output_path.write_bytes(b"audio")

            result = await synthesizer._synthesize_segment(
                text="Xin chào",
                voice=DEFAULT_VOICE,
                target_duration=2.0,
                output_path=output_path,
                start=5.0,
                end=7.0,
            )

        assert result.start == 5.0
        assert result.end == 7.0
        assert result.target_duration == 2.0

    @pytest.mark.asyncio
    async def test_no_speed_adjustment_when_target_duration_zero(
        self, synthesizer: VoiceSynthesizer, tmp_path: Path
    ) -> None:
        """Test no speed adjustment when target_duration is 0 (avoids division by zero)."""
        output_path = tmp_path / "segment_0000.mp3"

        with (
            patch("app.services.voice_synthesizer.edge_tts.Communicate") as mock_comm_cls,
            patch.object(synthesizer, "_get_audio_duration", return_value=2.0),
        ):
            mock_comm = MagicMock()
            mock_comm.save = AsyncMock()
            mock_comm_cls.return_value = mock_comm
            output_path.write_bytes(b"audio")

            result = await synthesizer._synthesize_segment(
                text="Xin chào",
                voice=DEFAULT_VOICE,
                target_duration=0.0,
                output_path=output_path,
                start=0.0,
                end=0.0,
            )

        assert result.speed_adjusted is False
        # Should only call Communicate once (no speed adjustment for zero target)
        assert mock_comm_cls.call_count == 1


class TestCombineSegments:
    """Tests for VoiceSynthesizer._combine_segments method."""

    def test_successful_combination(
        self, synthesizer: VoiceSynthesizer, tmp_path: Path
    ) -> None:
        """Test successful audio combination via FFmpeg."""
        seg1_path = tmp_path / "seg1.mp3"
        seg2_path = tmp_path / "seg2.mp3"
        seg1_path.write_bytes(b"audio1")
        seg2_path.write_bytes(b"audio2")

        segments = [
            SegmentAudio(
                path=seg1_path, start=0.0, end=2.0,
                duration=1.8, target_duration=2.0, speed_adjusted=False
            ),
            SegmentAudio(
                path=seg2_path, start=3.0, end=5.0,
                duration=1.9, target_duration=2.0, speed_adjusted=False
            ),
        ]

        output_path = tmp_path / "combined.wav"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch(
            "app.services.voice_synthesizer.subprocess.run",
            side_effect=lambda *args, **kwargs: (
                output_path.write_bytes(b"combined_audio"),
                mock_result,
            )[-1],
        ):
            synthesizer._combine_segments(segments, output_path)

        assert output_path.exists()

    def test_empty_segments_raises_error(
        self, synthesizer: VoiceSynthesizer, tmp_path: Path
    ) -> None:
        """Test that empty segments list raises VoiceSynthesizerError."""
        output_path = tmp_path / "combined.wav"
        with pytest.raises(VoiceSynthesizerError, match="No segments to combine"):
            synthesizer._combine_segments([], output_path)

    def test_ffmpeg_not_found_raises_error(
        self, synthesizer: VoiceSynthesizer, tmp_path: Path
    ) -> None:
        """Test that missing ffmpeg raises VoiceSynthesizerError."""
        seg_path = tmp_path / "seg.mp3"
        seg_path.write_bytes(b"audio")

        segments = [
            SegmentAudio(
                path=seg_path, start=0.0, end=2.0,
                duration=1.8, target_duration=2.0, speed_adjusted=False
            ),
        ]

        output_path = tmp_path / "combined.wav"

        with patch(
            "app.services.voice_synthesizer.subprocess.run",
            side_effect=FileNotFoundError(),
        ):
            with pytest.raises(VoiceSynthesizerError, match="ffmpeg not found"):
                synthesizer._combine_segments(segments, output_path)

    def test_ffmpeg_failure_raises_error(
        self, synthesizer: VoiceSynthesizer, tmp_path: Path
    ) -> None:
        """Test that FFmpeg failure raises VoiceSynthesizerError."""
        seg_path = tmp_path / "seg.mp3"
        seg_path.write_bytes(b"audio")

        segments = [
            SegmentAudio(
                path=seg_path, start=0.0, end=2.0,
                duration=1.8, target_duration=2.0, speed_adjusted=False
            ),
        ]

        output_path = tmp_path / "combined.wav"
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "encoding error"

        with patch(
            "app.services.voice_synthesizer.subprocess.run",
            return_value=mock_result,
        ):
            with pytest.raises(VoiceSynthesizerError, match="ffmpeg combination failed"):
                synthesizer._combine_segments(segments, output_path)


class TestSynthesize:
    """Tests for VoiceSynthesizer.synthesize method."""

    @pytest.mark.asyncio
    async def test_successful_synthesis(
        self,
        synthesizer: VoiceSynthesizer,
        sample_translation: TranslationResult,
        output_dir: Path,
    ) -> None:
        """Test successful synthesis of all segments."""
        seg_audio = SegmentAudio(
            path=output_dir / "segment_0000.mp3",
            start=0.0, end=3.0,
            duration=2.5, target_duration=3.0, speed_adjusted=False,
        )

        mock_synth = AsyncMock(return_value=seg_audio)

        with (
            patch.object(synthesizer, "_synthesize_segment", mock_synth),
            patch.object(synthesizer, "_combine_segments") as mock_combine,
        ):
            result = await synthesizer.synthesize(sample_translation, output_dir)

        assert isinstance(result, SynthesisResult)
        assert result.audio_path == output_dir / "vietnamese_audio.wav"
        assert len(result.segment_audios) == 2

    @pytest.mark.asyncio
    async def test_empty_segment_text_skipped(
        self, synthesizer: VoiceSynthesizer, output_dir: Path
    ) -> None:
        """Test that segments with empty translated text are skipped."""
        translation = TranslationResult(
            segments=[
                TranslatedSegment(
                    start=0.0, end=3.0,
                    original_text="你好", translated_text="Xin chào",
                    speaker=None,
                ),
                TranslatedSegment(
                    start=3.5, end=5.0,
                    original_text="", translated_text="   ",
                    speaker=None,
                ),
            ],
            full_text_original="你好",
            full_text_translated="Xin chào",
        )

        seg_audio = SegmentAudio(
            path=output_dir / "segment_0000.mp3",
            start=0.0, end=3.0,
            duration=2.5, target_duration=3.0, speed_adjusted=False,
        )

        mock_synth = AsyncMock(return_value=seg_audio)

        with (
            patch.object(synthesizer, "_synthesize_segment", mock_synth),
            patch.object(synthesizer, "_combine_segments"),
        ):
            result = await synthesizer.synthesize(translation, output_dir)

        # Only 1 segment should be synthesized (second is empty)
        assert len(result.segment_audios) == 1

    @pytest.mark.asyncio
    async def test_graceful_degradation_to_default_voice(
        self, synthesizer: VoiceSynthesizer, output_dir: Path
    ) -> None:
        """Test graceful degradation: if multi-voice fails, use default voice."""
        translation = TranslationResult(
            segments=[
                TranslatedSegment(
                    start=0.0, end=3.0,
                    original_text="你好", translated_text="Xin chào",
                    speaker="special_speaker",
                ),
            ],
            full_text_original="你好",
            full_text_translated="Xin chào",
        )

        seg_audio = SegmentAudio(
            path=output_dir / "segment_0000.mp3",
            start=0.0, end=3.0,
            duration=2.5, target_duration=3.0, speed_adjusted=False,
        )

        # First call fails, second (fallback) succeeds
        call_count = [0]

        async def mock_synthesize_segment(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Voice connection failed")
            return seg_audio

        with (
            patch.object(
                synthesizer,
                "_synthesize_segment",
                side_effect=mock_synthesize_segment,
            ),
            patch.object(synthesizer, "_combine_segments"),
        ):
            result = await synthesizer.synthesize(translation, output_dir)

        assert len(result.segment_audios) == 1

    @pytest.mark.asyncio
    async def test_all_empty_segments_raises_error(
        self, synthesizer: VoiceSynthesizer, output_dir: Path
    ) -> None:
        """Test that synthesis fails if all segments have empty text."""
        translation = TranslationResult(
            segments=[
                TranslatedSegment(
                    start=0.0, end=3.0,
                    original_text="", translated_text="",
                    speaker=None,
                ),
            ],
            full_text_original="",
            full_text_translated="",
        )

        with pytest.raises(VoiceSynthesizerError, match="No segments were synthesized"):
            await synthesizer.synthesize(translation, output_dir)

    @pytest.mark.asyncio
    async def test_different_voices_assigned_per_speaker(
        self, synthesizer: VoiceSynthesizer, output_dir: Path
    ) -> None:
        """Test that synthesize assigns different voices for different speakers."""
        translation = TranslationResult(
            segments=[
                TranslatedSegment(
                    start=0.0, end=2.0,
                    original_text="你好", translated_text="Xin chào",
                    speaker="speaker_A",
                ),
                TranslatedSegment(
                    start=2.5, end=5.0,
                    original_text="再见", translated_text="Tạm biệt",
                    speaker="speaker_B",
                ),
                TranslatedSegment(
                    start=5.5, end=8.0,
                    original_text="好的", translated_text="Được rồi",
                    speaker="speaker_A",
                ),
            ],
            full_text_original="你好 再见 好的",
            full_text_translated="Xin chào Tạm biệt Được rồi",
        )

        seg_audio = SegmentAudio(
            path=output_dir / "segment_0000.mp3",
            start=0.0, end=2.0,
            duration=1.5, target_duration=2.0, speed_adjusted=False,
        )

        voices_used: list[str] = []

        async def mock_synthesize_segment(**kwargs):
            voices_used.append(kwargs["voice"])
            return seg_audio

        with (
            patch.object(
                synthesizer, "_synthesize_segment", side_effect=mock_synthesize_segment
            ),
            patch.object(synthesizer, "_combine_segments"),
        ):
            await synthesizer.synthesize(translation, output_dir)

        # speaker_A and speaker_B should get different voices
        assert voices_used[0] != voices_used[1]
        # speaker_A should get the same voice both times
        assert voices_used[0] == voices_used[2]

    @pytest.mark.asyncio
    async def test_creates_output_dir_if_not_exists(
        self, synthesizer: VoiceSynthesizer, tmp_path: Path
    ) -> None:
        """Test that output_dir is created if it doesn't exist."""
        new_output = tmp_path / "nested" / "tts" / "output"
        assert not new_output.exists()

        translation = TranslationResult(
            segments=[
                TranslatedSegment(
                    start=0.0, end=3.0,
                    original_text="你好", translated_text="Xin chào",
                    speaker=None,
                ),
            ],
            full_text_original="你好",
            full_text_translated="Xin chào",
        )

        seg_audio = SegmentAudio(
            path=new_output / "segment_0000.mp3",
            start=0.0, end=3.0,
            duration=2.5, target_duration=3.0, speed_adjusted=False,
        )

        mock_synth = AsyncMock(return_value=seg_audio)

        with (
            patch.object(synthesizer, "_synthesize_segment", mock_synth),
            patch.object(synthesizer, "_combine_segments"),
        ):
            await synthesizer.synthesize(translation, new_output)

        assert new_output.exists()
