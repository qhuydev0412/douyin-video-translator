"""Pipeline orchestrator for the Douyin video translation workflow."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol

from app.models.job import CheckpointType, JobState, JobStatus, PipelineStep
from app.models.pipeline import (
    DownloadResult,
    SynthesisResult,
    TranscriptionResult,
    TranscriptionSegment,
    TranslatedSegment,
    TranslationResult,
    VocalIsolationResult,
)
from app.services.audio_extractor import AudioExtractor
from app.services.downloader import VideoDownloader
from app.services.speech_recognizer import SpeechRecognizer
from app.services.translator import Translator
from app.services.video_composer import VideoComposer
from app.services.vocal_isolator import VocalIsolator
from app.services.voice_synthesizer import VoiceSynthesizer

if TYPE_CHECKING:
    from app.services.checkpoint_manager import CheckpointManager
    from app.services.gender_detector import GenderDetector
    from app.services.subtitle_extractor import SubtitleExtractor
    from app.services.voice_preview import VoicePreviewGenerator

logger = logging.getLogger(__name__)


# Pipeline step order with progress percentages
STEP_PROGRESS: dict[PipelineStep, tuple[int, int]] = {
    PipelineStep.DOWNLOADING: (0, 15),
    PipelineStep.EXTRACTING_AUDIO: (15, 25),
    PipelineStep.ISOLATING_VOCALS: (25, 40),
    PipelineStep.RECOGNIZING_SPEECH: (40, 60),
    PipelineStep.TRANSLATING: (60, 75),
    PipelineStep.SYNTHESIZING_VOICE: (75, 90),
    PipelineStep.COMPOSING_VIDEO: (90, 100),
}

STEP_ORDER: list[PipelineStep] = [
    PipelineStep.DOWNLOADING,
    PipelineStep.EXTRACTING_AUDIO,
    PipelineStep.ISOLATING_VOCALS,
    PipelineStep.RECOGNIZING_SPEECH,
    PipelineStep.TRANSLATING,
    PipelineStep.SYNTHESIZING_VOICE,
    PipelineStep.COMPOSING_VIDEO,
]

# Mapping of pipeline steps that trigger a checkpoint pause after completion
CHECKPOINT_AFTER_STEP: dict[PipelineStep, CheckpointType] = {
    PipelineStep.TRANSLATING: CheckpointType.TRANSLATION,
}


class PipelineError(Exception):
    """Raised when a pipeline step fails."""

    def __init__(self, step: PipelineStep, message: str, retryable: bool = True):
        super().__init__(message)
        self.step = step
        self.message = message
        self.retryable = retryable


class CancellationError(Exception):
    """Raised when a job is cancelled during execution."""

    def __init__(self, job_id: str, step: PipelineStep):
        super().__init__(f"Job {job_id} cancelled at step {step.value}")
        self.job_id = job_id
        self.step = step


class CheckpointPauseSignal(Exception):
    """Raised when pipeline pauses at a checkpoint. Not an error — signals clean task completion."""

    def __init__(self, job_id: str, checkpoint_type: CheckpointType):
        super().__init__(f"Pipeline paused at checkpoint {checkpoint_type.value} for job {job_id}")
        self.job_id = job_id
        self.checkpoint_type = checkpoint_type


@dataclass
class PipelineResult:
    """Result of a completed pipeline execution."""

    output_path: Path
    job_id: str
    artifacts: dict[str, str] = field(default_factory=dict)


class JobStoreProtocol(Protocol):
    """Protocol for job state persistence (Redis-backed in production)."""

    def get_job(self, job_id: str) -> JobState: ...

    def update_job(self, job_id: str, **kwargs: object) -> None: ...

    def delete_job(self, job_id: str) -> None: ...

    def list_awaiting_confirmation_job_ids(self) -> list[str]: ...


class TranslationPipeline:
    """Orchestrates the full video translation pipeline.

    Executes 7 sequential steps: download → extract audio → isolate vocals →
    recognize speech → translate → synthesize voice → compose video.

    Each step updates job state in the store and checks for cancellation.
    On failure, artifacts from completed steps are preserved.
    """

    def __init__(
        self,
        downloader: VideoDownloader,
        extractor: AudioExtractor,
        isolator: VocalIsolator,
        recognizer: SpeechRecognizer,
        translator: Translator,
        synthesizer: VoiceSynthesizer,
        composer: VideoComposer,
        job_store: JobStoreProtocol,
        checkpoint_manager: Optional[CheckpointManager] = None,
        voice_preview_generator: Optional[VoicePreviewGenerator] = None,
        gender_detector: Optional[GenderDetector] = None,
        subtitle_extractor: Optional[SubtitleExtractor] = None,
    ) -> None:
        self._downloader = downloader
        self._extractor = extractor
        self._isolator = isolator
        self._recognizer = recognizer
        self._translator = translator
        self._synthesizer = synthesizer
        self._composer = composer
        self._job_store = job_store
        self._checkpoint_manager = checkpoint_manager
        self._voice_preview_generator = voice_preview_generator
        self._gender_detector = gender_detector
        self._subtitle_extractor = subtitle_extractor

    async def execute(self, job_id: str, url: str) -> Optional[PipelineResult]:
        """Execute the full translation pipeline with progress tracking.

        Args:
            job_id: Unique job identifier.
            url: Douyin video URL to process.

        Returns:
            PipelineResult with the output video path and all artifacts,
            or None if the pipeline paused at a checkpoint.

        Raises:
            PipelineError: If any step fails.
            CancellationError: If the job is cancelled during execution.
        """
        job_state = self._job_store.get_job(job_id)
        work_dir = Path(job_state.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        artifacts = dict(job_state.artifacts)

        try:
            return await self._run_steps(
                job_id=job_id,
                url=url,
                work_dir=work_dir,
                from_step_index=0,
                artifacts=artifacts,
            )
        except CheckpointPauseSignal:
            # Pipeline paused at a checkpoint — persist current artifacts and return None
            self._job_store.update_job(job_id, artifacts=artifacts)
            return None

    async def resume(self, job_id: str, from_step: str) -> Optional[PipelineResult]:
        """Resume pipeline from a specific step (after checkpoint confirmation or retry).

        Loads artifacts from job state and continues from the specified step.
        When resuming after a checkpoint confirmation, the CheckpointManager has
        already applied any user edits to the artifact files on disk, so the
        pipeline naturally uses the edited data when loading them.

        Args:
            job_id: Unique job identifier.
            from_step: PipelineStep value to resume from.

        Returns:
            PipelineResult with the output video path and all artifacts,
            or None if the pipeline paused at another checkpoint.

        Raises:
            PipelineError: If any step fails.
            CancellationError: If the job is cancelled during execution.
            ValueError: If from_step is not a valid pipeline step.
        """
        # Validate and find the step index
        try:
            target_step = PipelineStep(from_step)
        except ValueError:
            raise ValueError(f"Invalid pipeline step: {from_step}")

        from_step_index = STEP_ORDER.index(target_step)

        job_state = self._job_store.get_job(job_id)
        work_dir = Path(job_state.work_dir)
        url = job_state.url

        artifacts = dict(job_state.artifacts)

        try:
            return await self._run_steps(
                job_id=job_id,
                url=url,
                work_dir=work_dir,
                from_step_index=from_step_index,
                artifacts=artifacts,
            )
        except CheckpointPauseSignal:
            # Pipeline paused at another checkpoint — persist current artifacts and return None
            self._job_store.update_job(job_id, artifacts=artifacts)
            return None

    async def _run_steps(
        self,
        job_id: str,
        url: str,
        work_dir: Path,
        from_step_index: int,
        artifacts: dict[str, str],
    ) -> PipelineResult:
        """Run pipeline steps starting from a given index.

        Args:
            job_id: Unique job identifier.
            url: Douyin video URL.
            work_dir: Working directory for this job.
            from_step_index: Index in STEP_ORDER to start from.
            artifacts: Pre-existing artifacts from completed steps.

        Returns:
            PipelineResult on successful completion.

        Raises:
            PipelineError: If a step fails.
            CancellationError: If the job is cancelled.
        """
        # Check initial cancellation before starting
        self._check_cancellation(job_id, STEP_ORDER[from_step_index])

        self._job_store.update_job(job_id, status=JobStatus.PROCESSING)

        for i in range(from_step_index, len(STEP_ORDER)):
            step = STEP_ORDER[i]

            # Check cancellation before each step (skip first if already checked above)
            if i > from_step_index:
                self._check_cancellation(job_id, step)

            # Update progress to step start percentage
            start_percent, end_percent = STEP_PROGRESS[step]
            self._job_store.update_job(
                job_id,
                current_step=step,
                progress_percent=start_percent,
            )

            try:
                await self._execute_step(step, job_id, url, work_dir, artifacts)
            except (PipelineError, CancellationError, CheckpointPauseSignal):
                raise
            except Exception as e:
                logger.error(
                    "Pipeline step %s failed for job %s: %s",
                    step.value,
                    job_id,
                    str(e),
                )
                raise PipelineError(
                    step=step,
                    message=str(e),
                    retryable=True,
                )

            # Update progress to step end percentage
            self._job_store.update_job(
                job_id,
                progress_percent=end_percent,
                artifacts=artifacts,
            )

        # Pipeline completed successfully
        output_path = Path(artifacts["output_video"])
        self._job_store.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress_percent=100,
        )

        return PipelineResult(
            output_path=output_path,
            job_id=job_id,
            artifacts=artifacts,
        )

    async def _execute_step(
        self,
        step: PipelineStep,
        job_id: str,
        url: str,
        work_dir: Path,
        artifacts: dict[str, str],
    ) -> None:
        """Execute a single pipeline step and store its artifact.

        Args:
            step: The pipeline step to execute.
            job_id: Unique job identifier.
            url: Douyin video URL.
            work_dir: Working directory for this job.
            artifacts: Dict to store artifact paths (mutated in place).
        """
        if step == PipelineStep.DOWNLOADING:
            result: DownloadResult = self._downloader.download(url, work_dir)
            artifacts["video_path"] = str(result.video_path)

        elif step == PipelineStep.EXTRACTING_AUDIO:
            video_path = Path(artifacts["video_path"])
            audio_path: Path = self._extractor.extract(video_path, work_dir)
            artifacts["audio_path"] = str(audio_path)

        elif step == PipelineStep.ISOLATING_VOCALS:
            audio_path_iso = Path(artifacts["audio_path"])
            isolation_result: VocalIsolationResult = self._isolator.isolate(
                audio_path_iso, work_dir
            )
            artifacts["vocals_path"] = str(isolation_result.vocals_path)
            artifacts["background_path"] = str(isolation_result.background_path)

        elif step == PipelineStep.RECOGNIZING_SPEECH:
            vocals_path = Path(artifacts["vocals_path"])
            video_path = Path(artifacts["video_path"])

            # Try subtitles first (faster, more accurate than Whisper)
            transcription: TranscriptionResult | None = None
            if self._subtitle_extractor is not None:
                transcription = self._subtitle_extractor.try_extract(video_path, work_dir)

            if transcription is None:
                transcription = self._recognizer.recognize(vocals_path)

            # Detect speaker gender via pitch analysis (best-effort, non-blocking)
            if self._gender_detector is not None:
                try:
                    speaker_genders = self._gender_detector.detect(vocals_path, transcription)
                    gender_path = work_dir / "speaker_genders.json"
                    gender_path.write_text(
                        json.dumps(speaker_genders, ensure_ascii=False), encoding="utf-8"
                    )
                    artifacts["speaker_genders_path"] = str(gender_path)
                    logger.info("Speaker genders for job %s: %s", job_id, speaker_genders)
                except Exception as exc:
                    logger.warning("Gender detection failed for job %s: %s", job_id, exc)

            # Serialize transcription to JSON for later steps
            transcription_path = work_dir / "transcription.json"
            self._save_transcription(transcription, transcription_path)
            artifacts["transcription_path"] = str(transcription_path)

        elif step == PipelineStep.TRANSLATING:
            transcription_path = Path(artifacts["transcription_path"])
            transcription_data = self._load_transcription(transcription_path)
            translation: TranslationResult = self._translator.translate(transcription_data)
            # Serialize translation to JSON for voice synthesis
            translation_path = work_dir / "translation.json"
            self._save_translation(translation, translation_path)
            artifacts["translation_path"] = str(translation_path)

        elif step == PipelineStep.SYNTHESIZING_VOICE:
            translation_path = Path(artifacts["translation_path"])
            translation_data = self._load_translation(translation_path)

            # Load speaker gender map (built during RECOGNIZING_SPEECH)
            speaker_genders: dict[str, str] = {}
            if "speaker_genders_path" in artifacts:
                gender_path = Path(artifacts["speaker_genders_path"])
                if gender_path.exists():
                    speaker_genders = json.loads(gender_path.read_text(encoding="utf-8"))

            # Detect whether content has multiple distinct speakers
            unique_speakers = {seg.speaker for seg in translation_data.segments if seg.speaker}
            has_multiple_speakers = len(unique_speakers) > 1

            # Always let user pick voice (single or multi-speaker)
            if (
                "selected_voice_id" not in artifacts
                and self._voice_preview_generator is not None
            ):
                voice_options = await self._voice_preview_generator.generate_previews(
                    translation=translation_data,
                    work_dir=work_dir,
                    job_id=job_id,
                )
                previews_dir = work_dir / "voice_previews"
                artifacts["voice_previews_dir"] = str(previews_dir)
                self._job_store.update_job(job_id, voice_options=voice_options)
                if self._checkpoint_manager is not None:
                    self._checkpoint_manager.pause_at_checkpoint(
                        job_id, CheckpointType.VOICE_SELECTION
                    )
                    raise CheckpointPauseSignal(
                        job_id=job_id, checkpoint_type=CheckpointType.VOICE_SELECTION
                    )
            else:
                selected_voice = artifacts.get("selected_voice_id")

                # Load per-segment voices saved from a previous synthesis pass
                per_segment_voices: list[str | None] | None = None
                voices_path = work_dir / "segment_voices.json"
                if voices_path.exists() and "audio_preview_confirmed" in artifacts:
                    try:
                        per_segment_voices = json.loads(voices_path.read_text(encoding="utf-8"))
                    except Exception as exc:
                        logger.warning("Could not load segment_voices.json for job %s: %s", job_id, exc)

                synthesis: SynthesisResult = await self._synthesizer.synthesize(
                    translation_data,
                    work_dir,
                    voice=selected_voice,
                    speaker_gender_map=speaker_genders if has_multiple_speakers else None,
                    per_segment_voices=per_segment_voices,
                )
                artifacts["vietnamese_audio"] = str(synthesis.audio_path)

                # Always save per-segment voices for the audio preview UI
                voices_path.write_text(
                    json.dumps(synthesis.segment_voices, ensure_ascii=False), encoding="utf-8"
                )
                artifacts["segment_voices_path"] = str(voices_path)

                # Pause for audio preview (once only — skip if already confirmed)
                if (
                    "audio_preview_confirmed" not in artifacts
                    and self._checkpoint_manager is not None
                ):
                    self._checkpoint_manager.pause_at_checkpoint(
                        job_id, CheckpointType.AUDIO_PREVIEW
                    )
                    raise CheckpointPauseSignal(
                        job_id=job_id, checkpoint_type=CheckpointType.AUDIO_PREVIEW
                    )

        elif step == PipelineStep.COMPOSING_VIDEO:
            video_path_comp = Path(artifacts["video_path"])
            vietnamese_audio = Path(artifacts["vietnamese_audio"])
            background_audio = Path(artifacts["background_path"])
            output_dir = work_dir / "output"
            output_path: Path = self._composer.compose(
                video_path_comp, vietnamese_audio, background_audio, output_dir
            )
            artifacts["output_video"] = str(output_path)

        # Check if the completed step triggers a checkpoint pause
        if self._checkpoint_manager is not None and step in CHECKPOINT_AFTER_STEP:
            checkpoint_type = CHECKPOINT_AFTER_STEP[step]
            self._checkpoint_manager.pause_at_checkpoint(job_id, checkpoint_type)
            raise CheckpointPauseSignal(job_id=job_id, checkpoint_type=checkpoint_type)

    def _check_cancellation(self, job_id: str, step: PipelineStep) -> None:
        """Check if the job has been cancelled.

        Args:
            job_id: Unique job identifier.
            step: Current step (for error reporting).

        Raises:
            CancellationError: If the job status is CANCELLED.
        """
        job_state = self._job_store.get_job(job_id)
        if job_state.status == JobStatus.CANCELLED:
            raise CancellationError(job_id=job_id, step=step)

    def _save_transcription(
        self, transcription: TranscriptionResult, path: Path
    ) -> None:
        """Serialize TranscriptionResult to a JSON file."""
        data: dict[str, Any] = {
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker": seg.speaker,
                }
                for seg in transcription.segments
            ],
            "full_text": transcription.full_text,
            "language": transcription.language,
            "confidence": transcription.confidence,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_transcription(self, path: Path) -> TranscriptionResult:
        """Deserialize TranscriptionResult from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        segments = [
            TranscriptionSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                speaker=seg.get("speaker"),
            )
            for seg in data["segments"]
        ]
        return TranscriptionResult(
            segments=segments,
            full_text=data["full_text"],
            language=data["language"],
            confidence=data["confidence"],
        )

    def _save_translation(self, translation: TranslationResult, path: Path) -> None:
        """Serialize TranslationResult to a JSON file."""
        data: dict[str, Any] = {
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "original_text": seg.original_text,
                    "translated_text": seg.translated_text,
                    "speaker": seg.speaker,
                }
                for seg in translation.segments
            ],
            "full_text_original": translation.full_text_original,
            "full_text_translated": translation.full_text_translated,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_translation(self, path: Path) -> TranslationResult:
        """Deserialize TranslationResult from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        segments = [
            TranslatedSegment(
                start=seg["start"],
                end=seg["end"],
                original_text=seg["original_text"],
                translated_text=seg["translated_text"],
                speaker=seg.get("speaker"),
            )
            for seg in data["segments"]
        ]
        return TranslationResult(
            segments=segments,
            full_text_original=data["full_text_original"],
            full_text_translated=data["full_text_translated"],
        )
