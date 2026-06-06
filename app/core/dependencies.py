"""Dependency injection for all application services.

Provides factory functions to create service instances with proper
configuration. Used by the pipeline, Celery tasks, and API layer.
"""

from app.services.audio_extractor import AudioExtractor
from app.services.downloader import VideoDownloader
from app.services.job_store import JobStore
from app.services.speech_recognizer import SpeechRecognizer
from app.services.translator import Translator
from app.services.video_composer import VideoComposer
from app.services.vocal_isolator import VocalIsolator
from app.services.voice_synthesizer import VoiceSynthesizer


def get_job_store() -> JobStore:
    """Create a JobStore instance connected to Redis."""
    return JobStore()


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
