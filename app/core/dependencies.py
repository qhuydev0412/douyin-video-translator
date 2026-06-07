"""Dependency injection for all application services.

Provides factory functions to create service instances with proper
configuration. Used by the pipeline, Celery tasks, and API layer.
"""

from app.services.audio_extractor import AudioExtractor
from app.services.checkpoint_manager import CheckpointManager
from app.services.downloader import VideoDownloader
from app.services.job_store import JobStore
from app.services.speech_recognizer import SpeechRecognizer
from app.services.translator import Translator
from app.services.video_composer import VideoComposer
from app.services.vocal_isolator import VocalIsolator
from app.services.voice_preview import VoicePreviewGenerator
from app.services.voice_synthesizer import VoiceSynthesizer


def get_job_store() -> JobStore:
    """Create a JobStore instance connected to Redis."""
    return JobStore()


def get_checkpoint_manager(job_store: JobStore | None = None) -> CheckpointManager:
    """Create a CheckpointManager instance.

    Args:
        job_store: Optional JobStore instance. If None, creates a new one.

    Returns:
        Configured CheckpointManager instance.
    """
    if job_store is None:
        job_store = JobStore()
    return CheckpointManager(job_store)


def get_voice_preview_generator(synthesizer: VoiceSynthesizer | None = None) -> VoicePreviewGenerator:
    """Create a VoicePreviewGenerator instance.

    Args:
        synthesizer: Optional VoiceSynthesizer instance. If None, creates a new one.

    Returns:
        Configured VoicePreviewGenerator instance.
    """
    if synthesizer is None:
        synthesizer = VoiceSynthesizer()
    return VoicePreviewGenerator(synthesizer)


def get_downloader() -> VideoDownloader:
    """Create a VideoDownloader instance."""
    return VideoDownloader()


def get_audio_extractor() -> AudioExtractor:
    """Create an AudioExtractor instance."""
    return AudioExtractor()


def get_vocal_isolator() -> VocalIsolator:
    """Create a VocalIsolator instance."""
    return VocalIsolator()


def get_speech_recognizer() -> SpeechRecognizer:
    """Create a SpeechRecognizer instance."""
    return SpeechRecognizer()


def get_translator() -> Translator:
    """Create a Translator instance."""
    return Translator()


def get_voice_synthesizer() -> VoiceSynthesizer:
    """Create a VoiceSynthesizer instance."""
    return VoiceSynthesizer()


def get_video_composer() -> VideoComposer:
    """Create a VideoComposer instance."""
    return VideoComposer()
