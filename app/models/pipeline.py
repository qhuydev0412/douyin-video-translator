"""Pipeline data models using dataclasses for intermediate processing results."""

from dataclasses import dataclass, field
from pathlib import Path

from app.models.job import VideoInfo


@dataclass
class DownloadResult:
    """Result of downloading a video from Douyin."""

    video_path: Path
    video_info: VideoInfo


@dataclass
class TranscriptionSegment:
    """A single segment of transcribed speech."""

    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class TranscriptionResult:
    """Complete transcription result from speech recognition."""

    segments: list[TranscriptionSegment]
    full_text: str
    language: str
    confidence: float


@dataclass
class TranslatedSegment:
    """A single translated text segment with timing."""

    start: float
    end: float
    original_text: str
    translated_text: str
    speaker: str | None = None


@dataclass
class TranslationResult:
    """Complete translation result."""

    segments: list[TranslatedSegment]
    full_text_original: str
    full_text_translated: str


@dataclass
class VocalIsolationResult:
    """Result of separating vocals from background audio."""

    vocals_path: Path
    background_path: Path


@dataclass
class SegmentAudio:
    """Audio file for a single synthesized segment."""

    path: Path
    start: float
    end: float
    duration: float
    target_duration: float
    speed_adjusted: bool


@dataclass
class SynthesisResult:
    """Result of voice synthesis for all segments."""

    audio_path: Path
    segment_audios: list[SegmentAudio] = field(default_factory=list)
    segment_voices: list[str | None] = field(default_factory=list)
