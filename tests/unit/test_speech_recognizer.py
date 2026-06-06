"""Unit tests for SpeechRecognizer service."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app.models.pipeline import TranscriptionResult, TranscriptionSegment
from app.services.speech_recognizer import SpeechRecognizer, SpeechRecognitionError


@pytest.fixture
def recognizer() -> SpeechRecognizer:
    """Provide a SpeechRecognizer instance with default model name."""
    return SpeechRecognizer(model_name="large-v3")


@pytest.fixture
def fake_audio(tmp_path: Path) -> Path:
    """Create a fake audio file for testing."""
    audio_file = tmp_path / "vocals.wav"
    audio_file.write_bytes(b"RIFF fake wav data")
    return audio_file


@pytest.fixture
def mock_whisper_model() -> MagicMock:
    """Provide a mocked Whisper model."""
    return MagicMock()


class TestRecognizeSuccess:
    """Tests for successful speech recognition scenarios."""

    def test_transcription_output_has_correct_structure(
        self, recognizer: SpeechRecognizer, fake_audio: Path, mock_whisper_model: MagicMock
    ) -> None:
        """Test that transcription returns properly structured result with timestamps."""
        mock_whisper_model.transcribe.return_value = {
            "text": "你好世界，欢迎来到这里",
            "language": "zh",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "你好世界",
                    "no_speech_prob": 0.05,
                },
                {
                    "start": 2.8,
                    "end": 5.0,
                    "text": "欢迎来到这里",
                    "no_speech_prob": 0.1,
                },
            ],
        }

        with patch("whisper.load_model", return_value=mock_whisper_model):
            result = recognizer.recognize(fake_audio)

        assert isinstance(result, TranscriptionResult)
        assert len(result.segments) == 2
        assert result.full_text == "你好世界，欢迎来到这里"
        assert result.language == "zh"
        assert 0.0 <= result.confidence <= 1.0

    def test_segments_have_valid_timestamps(
        self, recognizer: SpeechRecognizer, fake_audio: Path, mock_whisper_model: MagicMock
    ) -> None:
        """Test that each segment has start < end and proper float types."""
        mock_whisper_model.transcribe.return_value = {
            "text": "第一段话。第二段话。第三段话。",
            "language": "zh",
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "第一段话。", "no_speech_prob": 0.02},
                {"start": 1.8, "end": 3.2, "text": "第二段话。", "no_speech_prob": 0.03},
                {"start": 3.5, "end": 5.0, "text": "第三段话。", "no_speech_prob": 0.04},
            ],
        }

        with patch("whisper.load_model", return_value=mock_whisper_model):
            result = recognizer.recognize(fake_audio)

        for segment in result.segments:
            assert isinstance(segment, TranscriptionSegment)
            assert isinstance(segment.start, float)
            assert isinstance(segment.end, float)
            assert segment.start < segment.end
            assert isinstance(segment.text, str)
            assert len(segment.text) > 0

    def test_segments_maintain_temporal_ordering(
        self, recognizer: SpeechRecognizer, fake_audio: Path, mock_whisper_model: MagicMock
    ) -> None:
        """Test that segments are in chronological order."""
        mock_whisper_model.transcribe.return_value = {
            "text": "一二三四",
            "language": "zh",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "一", "no_speech_prob": 0.01},
                {"start": 1.2, "end": 2.5, "text": "二", "no_speech_prob": 0.02},
                {"start": 3.0, "end": 4.0, "text": "三", "no_speech_prob": 0.01},
                {"start": 4.5, "end": 5.5, "text": "四", "no_speech_prob": 0.03},
            ],
        }

        with patch("whisper.load_model", return_value=mock_whisper_model):
            result = recognizer.recognize(fake_audio)

        for i in range(len(result.segments) - 1):
            assert result.segments[i].end <= result.segments[i + 1].start

    def test_confidence_calculated_from_no_speech_prob(
        self, recognizer: SpeechRecognizer, fake_audio: Path, mock_whisper_model: MagicMock
    ) -> None:
        """Test confidence is calculated as mean of (1 - no_speech_prob)."""
        mock_whisper_model.transcribe.return_value = {
            "text": "测试",
            "language": "zh",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "测试", "no_speech_prob": 0.1},
                {"start": 1.5, "end": 2.5, "text": "内容", "no_speech_prob": 0.3},
            ],
        }

        with patch("whisper.load_model", return_value=mock_whisper_model):
            result = recognizer.recognize(fake_audio)

        # Expected: mean([1 - 0.1, 1 - 0.3]) = mean([0.9, 0.7]) = 0.8
        assert abs(result.confidence - 0.8) < 1e-6


class TestRecognizeEmptyOrSilent:
    """Tests for empty or silent audio handling."""

    def test_raises_error_when_no_segments_returned(
        self, recognizer: SpeechRecognizer, fake_audio: Path, mock_whisper_model: MagicMock
    ) -> None:
        """Test that empty segments list raises SpeechRecognitionError."""
        mock_whisper_model.transcribe.return_value = {
            "text": "",
            "language": "zh",
            "segments": [],
        }

        with patch("whisper.load_model", return_value=mock_whisper_model):
            with pytest.raises(
                SpeechRecognitionError,
                match="Không nhận dạng được giọng nói trong file âm thanh",
            ):
                recognizer.recognize(fake_audio)

    def test_raises_error_when_text_is_empty(
        self, recognizer: SpeechRecognizer, fake_audio: Path, mock_whisper_model: MagicMock
    ) -> None:
        """Test that empty full text (whitespace only) raises error."""
        mock_whisper_model.transcribe.return_value = {
            "text": "   ",
            "language": "zh",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "   ", "no_speech_prob": 0.95},
            ],
        }

        with patch("whisper.load_model", return_value=mock_whisper_model):
            with pytest.raises(
                SpeechRecognitionError,
                match="Không nhận dạng được giọng nói trong file âm thanh",
            ):
                recognizer.recognize(fake_audio)

    def test_raises_error_when_audio_file_not_found(
        self, recognizer: SpeechRecognizer, tmp_path: Path
    ) -> None:
        """Test that a nonexistent audio file raises error."""
        nonexistent = tmp_path / "nonexistent.wav"
        with pytest.raises(SpeechRecognitionError, match="Audio file not found"):
            recognizer.recognize(nonexistent)

    def test_raises_error_when_transcription_fails(
        self, recognizer: SpeechRecognizer, fake_audio: Path, mock_whisper_model: MagicMock
    ) -> None:
        """Test that exceptions during transcription are wrapped."""
        mock_whisper_model.transcribe.side_effect = RuntimeError("CUDA out of memory")

        with patch("whisper.load_model", return_value=mock_whisper_model):
            with pytest.raises(SpeechRecognitionError, match="Transcription failed"):
                recognizer.recognize(fake_audio)


class TestMultiSpeakerLabeling:
    """Tests for multi-speaker detection and labeling."""

    def test_single_speaker_when_no_gaps(
        self, recognizer: SpeechRecognizer, fake_audio: Path, mock_whisper_model: MagicMock
    ) -> None:
        """Test that continuous speech is labeled as single speaker."""
        mock_whisper_model.transcribe.return_value = {
            "text": "连续说话没有停顿",
            "language": "zh",
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "连续", "no_speech_prob": 0.05},
                {"start": 1.6, "end": 3.0, "text": "说话", "no_speech_prob": 0.04},
                {"start": 3.1, "end": 4.5, "text": "没有停顿", "no_speech_prob": 0.06},
            ],
        }

        with patch("whisper.load_model", return_value=mock_whisper_model):
            result = recognizer.recognize(fake_audio)

        # All segments should have the same speaker
        speakers = {seg.speaker for seg in result.segments}
        assert len(speakers) == 1
        assert "speaker_1" in speakers

    def test_multiple_speakers_detected_with_large_gaps(
        self, recognizer: SpeechRecognizer, fake_audio: Path, mock_whisper_model: MagicMock
    ) -> None:
        """Test that gaps > 2s between segments trigger new speaker labels."""
        mock_whisper_model.transcribe.return_value = {
            "text": "第一个人说话第二个人回答第三个人补充",
            "language": "zh",
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "第一个人说话", "no_speech_prob": 0.05},
                # Gap of 3 seconds -> new speaker
                {"start": 5.0, "end": 7.0, "text": "第二个人回答", "no_speech_prob": 0.04},
                # Gap of 2.5 seconds -> new speaker
                {"start": 9.5, "end": 11.0, "text": "第三个人补充", "no_speech_prob": 0.06},
            ],
        }

        with patch("whisper.load_model", return_value=mock_whisper_model):
            result = recognizer.recognize(fake_audio)

        assert result.segments[0].speaker == "speaker_1"
        assert result.segments[1].speaker == "speaker_2"
        assert result.segments[2].speaker == "speaker_3"

    def test_speaker_label_does_not_change_within_small_gap(
        self, recognizer: SpeechRecognizer, fake_audio: Path, mock_whisper_model: MagicMock
    ) -> None:
        """Test that gaps <= 2s keep the same speaker label."""
        mock_whisper_model.transcribe.return_value = {
            "text": "短暂停顿后继续",
            "language": "zh",
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "短暂", "no_speech_prob": 0.05},
                # Gap of exactly 2.0 seconds -> same speaker (threshold is > 2.0)
                {"start": 4.0, "end": 5.5, "text": "停顿", "no_speech_prob": 0.04},
                # Gap of 1.5 seconds -> same speaker
                {"start": 7.0, "end": 8.5, "text": "后继续", "no_speech_prob": 0.06},
            ],
        }

        with patch("whisper.load_model", return_value=mock_whisper_model):
            result = recognizer.recognize(fake_audio)

        # All segments should have the same speaker (gaps <= 2.0)
        speakers = {seg.speaker for seg in result.segments}
        assert len(speakers) == 1

    def test_all_segments_have_speaker_label(
        self, recognizer: SpeechRecognizer, fake_audio: Path, mock_whisper_model: MagicMock
    ) -> None:
        """Test that every segment has a non-None speaker label."""
        mock_whisper_model.transcribe.return_value = {
            "text": "每段都有说话人标签",
            "language": "zh",
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "每段", "no_speech_prob": 0.02},
                {"start": 1.8, "end": 3.0, "text": "都有", "no_speech_prob": 0.03},
                {"start": 5.5, "end": 7.0, "text": "说话人标签", "no_speech_prob": 0.04},
            ],
        }

        with patch("whisper.load_model", return_value=mock_whisper_model):
            result = recognizer.recognize(fake_audio)

        for segment in result.segments:
            assert segment.speaker is not None
            assert segment.speaker.startswith("speaker_")


class TestModelLoading:
    """Tests for Whisper model loading behavior."""

    def test_raises_error_when_model_loading_fails(
        self, recognizer: SpeechRecognizer, fake_audio: Path
    ) -> None:
        """Test that model loading errors are wrapped in SpeechRecognitionError."""
        with patch("whisper.load_model", side_effect=RuntimeError("Model not found")):
            with pytest.raises(SpeechRecognitionError, match="Failed to load Whisper model"):
                recognizer.recognize(fake_audio)

    def test_model_loaded_lazily_on_first_call(
        self, recognizer: SpeechRecognizer, fake_audio: Path, mock_whisper_model: MagicMock
    ) -> None:
        """Test that model is not loaded until recognize() is called."""
        assert recognizer._model is None

        mock_whisper_model.transcribe.return_value = {
            "text": "测试",
            "language": "zh",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "测试", "no_speech_prob": 0.1},
            ],
        }

        with patch("whisper.load_model", return_value=mock_whisper_model) as mock_load:
            recognizer.recognize(fake_audio)
            assert mock_load.call_count == 1
            assert recognizer._model is not None

    def test_model_reused_on_subsequent_calls(
        self, recognizer: SpeechRecognizer, fake_audio: Path, mock_whisper_model: MagicMock
    ) -> None:
        """Test that model is loaded once and reused across calls."""
        mock_whisper_model.transcribe.return_value = {
            "text": "测试",
            "language": "zh",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "测试", "no_speech_prob": 0.1},
            ],
        }

        with patch("whisper.load_model", return_value=mock_whisper_model) as mock_load:
            recognizer.recognize(fake_audio)
            recognizer.recognize(fake_audio)
            # Model should only be loaded once
            assert mock_load.call_count == 1
