"""Unit tests for voice preview generation service."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.models.pipeline import TranslatedSegment, TranslationResult
from app.services.voice_preview import (
    CHARS_PER_SECOND,
    MAX_PREVIEW_DURATION_SECONDS,
    MIN_SUCCESSFUL_PREVIEWS,
    PREVIEW_VOICES,
    VoicePreviewError,
    VoicePreviewGenerator,
)
from app.services.voice_synthesizer import VoiceSynthesizer


@pytest.fixture
def mock_synthesizer() -> MagicMock:
    """Create a mock VoiceSynthesizer."""
    synth = MagicMock(spec=VoiceSynthesizer)
    synth._model = "tts-1"
    # Mock the client.audio.speech.create chain
    mock_response = MagicMock()
    mock_response.stream_to_file = MagicMock()
    synth.client.audio.speech.create.return_value = mock_response
    return synth


@pytest.fixture
def generator(mock_synthesizer: MagicMock) -> VoicePreviewGenerator:
    """Create a VoicePreviewGenerator with a mock synthesizer."""
    return VoicePreviewGenerator(mock_synthesizer)


@pytest.fixture
def sample_segments() -> list[TranslatedSegment]:
    """Create sample translated segments for testing."""
    return [
        TranslatedSegment(
            start=0.0,
            end=3.0,
            original_text="短句",
            translated_text="Câu ngắn",
            speaker="speaker_1",
        ),
        TranslatedSegment(
            start=3.0,
            end=10.0,
            original_text="这是一个中等长度的句子",
            translated_text="Đây là một câu có độ dài trung bình cho việc kiểm tra",
            speaker="speaker_1",
        ),
        TranslatedSegment(
            start=10.0,
            end=20.0,
            original_text="这是一个非常长的句子",
            translated_text="Đây là một câu rất dài dùng để kiểm tra xem hệ thống có chọn đúng đoạn văn bản hay không trong quá trình tạo bản xem trước",
            speaker="speaker_2",
        ),
    ]


@pytest.fixture
def sample_translation(
    sample_segments: list[TranslatedSegment],
) -> TranslationResult:
    """Create a sample TranslationResult."""
    return TranslationResult(
        segments=sample_segments,
        full_text_original="短句. 这是一个中等长度的句子. 这是一个非常长的句子.",
        full_text_translated="Câu ngắn. Đây là một câu có độ dài trung bình. Đây là một câu rất dài.",
    )


class TestSelectPreviewSegment:
    """Tests for VoicePreviewGenerator.select_preview_segment."""

    def test_selects_longest_within_limit(
        self, generator: VoicePreviewGenerator
    ) -> None:
        """Should select the longest segment that doesn't exceed 15s estimated."""
        max_chars = int(MAX_PREVIEW_DURATION_SECONDS * CHARS_PER_SECOND)
        # Create segments: short, medium (within limit), long (exceeds limit)
        segments = [
            TranslatedSegment(
                start=0.0, end=2.0, original_text="a", translated_text="Short",
            ),
            TranslatedSegment(
                start=2.0, end=8.0, original_text="b",
                translated_text="A" * (max_chars - 5),  # Within limit
            ),
            TranslatedSegment(
                start=8.0, end=20.0, original_text="c",
                translated_text="B" * (max_chars + 10),  # Exceeds limit
            ),
        ]

        result = generator.select_preview_segment(segments)
        # Should pick the medium segment (longest within limit)
        assert result.translated_text == "A" * (max_chars - 5)

    def test_selects_segment_exactly_at_limit(
        self, generator: VoicePreviewGenerator
    ) -> None:
        """A segment at exactly the char limit should be eligible."""
        max_chars = int(MAX_PREVIEW_DURATION_SECONDS * CHARS_PER_SECOND)
        segments = [
            TranslatedSegment(
                start=0.0, end=5.0, original_text="a",
                translated_text="X" * max_chars,
            ),
        ]
        result = generator.select_preview_segment(segments)
        assert len(result.translated_text) == max_chars

    def test_falls_back_to_shortest_if_all_exceed(
        self, generator: VoicePreviewGenerator
    ) -> None:
        """If all segments exceed limit, fallback to shortest one."""
        max_chars = int(MAX_PREVIEW_DURATION_SECONDS * CHARS_PER_SECOND)
        segments = [
            TranslatedSegment(
                start=0.0, end=20.0, original_text="a",
                translated_text="A" * (max_chars + 100),
            ),
            TranslatedSegment(
                start=20.0, end=40.0, original_text="b",
                translated_text="B" * (max_chars + 5),  # Shortest exceeding
            ),
        ]
        result = generator.select_preview_segment(segments)
        # Should pick the shortest one as fallback
        assert result.translated_text == "B" * (max_chars + 5)

    def test_skips_empty_segments(
        self, generator: VoicePreviewGenerator
    ) -> None:
        """Segments with empty/whitespace text should be skipped."""
        segments = [
            TranslatedSegment(
                start=0.0, end=1.0, original_text="", translated_text="",
            ),
            TranslatedSegment(
                start=1.0, end=2.0, original_text="a", translated_text="   ",
            ),
            TranslatedSegment(
                start=2.0, end=5.0, original_text="b",
                translated_text="Valid text here",
            ),
        ]
        result = generator.select_preview_segment(segments)
        assert result.translated_text == "Valid text here"

    def test_raises_error_when_all_empty(
        self, generator: VoicePreviewGenerator
    ) -> None:
        """Should raise VoicePreviewError when all segments are empty."""
        segments = [
            TranslatedSegment(
                start=0.0, end=1.0, original_text="", translated_text="",
            ),
            TranslatedSegment(
                start=1.0, end=2.0, original_text="a", translated_text="   ",
            ),
        ]
        with pytest.raises(VoicePreviewError, match="No non-empty segments"):
            generator.select_preview_segment(segments)


class TestGeneratePreviews:
    """Tests for VoicePreviewGenerator.generate_previews."""

    @pytest.mark.asyncio
    async def test_generates_previews_successfully(
        self,
        generator: VoicePreviewGenerator,
        sample_translation: TranslationResult,
        tmp_path: Path,
    ) -> None:
        """Should generate previews for all voice options."""
        # The mock will succeed for all voices (stream_to_file is mocked)
        result = await generator.generate_previews(
            translation=sample_translation,
            work_dir=tmp_path,
            job_id="test-job-123",
        )

        # Should have generated previews for all voices
        assert len(result) == len(PREVIEW_VOICES)
        for opt in result:
            assert opt.voice_id in [v["voice_id"] for v in PREVIEW_VOICES]
            assert opt.preview_url.startswith("/api/v1/jobs/test-job-123/preview/voice/")

    @pytest.mark.asyncio
    async def test_stores_voice_options_metadata(
        self,
        generator: VoicePreviewGenerator,
        sample_translation: TranslationResult,
        tmp_path: Path,
    ) -> None:
        """Should store voice_options.json in the previews directory."""
        await generator.generate_previews(
            translation=sample_translation,
            work_dir=tmp_path,
            job_id="test-job-123",
        )

        metadata_path = tmp_path / "voice_previews" / "voice_options.json"
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert "preview_text" in metadata
        assert "voice_options" in metadata
        assert len(metadata["voice_options"]) == len(PREVIEW_VOICES)

    @pytest.mark.asyncio
    async def test_handles_partial_failures(
        self,
        mock_synthesizer: MagicMock,
        sample_translation: TranslationResult,
        tmp_path: Path,
    ) -> None:
        """Should succeed if at least 2 previews generate, fail otherwise."""
        # Make all but 2 calls fail
        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                raise RuntimeError("TTS API failed")
            mock_resp = MagicMock()
            mock_resp.stream_to_file = MagicMock()
            return mock_resp

        mock_synthesizer.client.audio.speech.create.side_effect = side_effect
        gen = VoicePreviewGenerator(mock_synthesizer)

        result = await gen.generate_previews(
            translation=sample_translation,
            work_dir=tmp_path,
            job_id="test-job-partial",
        )
        # Should have exactly 2 successful previews
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_fails_if_fewer_than_2_succeed(
        self,
        mock_synthesizer: MagicMock,
        sample_translation: TranslationResult,
        tmp_path: Path,
    ) -> None:
        """Should raise VoicePreviewError if fewer than 2 previews succeed."""
        # Make all but 1 call fail
        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise RuntimeError("TTS API failed")
            mock_resp = MagicMock()
            mock_resp.stream_to_file = MagicMock()
            return mock_resp

        mock_synthesizer.client.audio.speech.create.side_effect = side_effect
        gen = VoicePreviewGenerator(mock_synthesizer)

        with pytest.raises(VoicePreviewError, match="minimum 2 required"):
            await gen.generate_previews(
                translation=sample_translation,
                work_dir=tmp_path,
                job_id="test-job-fail",
            )

    @pytest.mark.asyncio
    async def test_creates_previews_directory(
        self,
        generator: VoicePreviewGenerator,
        sample_translation: TranslationResult,
        tmp_path: Path,
    ) -> None:
        """Should create the voice_previews directory."""
        await generator.generate_previews(
            translation=sample_translation,
            work_dir=tmp_path,
            job_id="test-job-dir",
        )
        assert (tmp_path / "voice_previews").is_dir()
