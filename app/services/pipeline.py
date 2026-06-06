"""Pipeline orchestrator for the Douyin video translation workflow."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from app.models.job import JobState, JobStatus, PipelineStep
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
    ) -> None:
        self._downloader = downloader
        self._extractor = extractor
        self._isolator = isolator
        self._recognizer = recognizer
        self._translator = translator
        self._synthesizer = synthesizer
        self._composer = composer
        self._job_store = job_store

    async def execute(self, job_id: str, url: str) -> PipelineResult:
        """Execute the full translation pipeline with progress tracking.

        Args:
            job_id: Unique job identifier.
            url: Douyin video URL to process.

        Returns:
            PipelineResult with the output video path and all artifacts.

        Raises:
            PipelineError: If any step fails.
            CancellationError: If the job is cancelled during execution.
        """
        job_state = self._job_store.get_job(job_id)
        work_dir = Path(job_state.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        return await self._run_steps(
            job_id=job_id,
            url=url,
            work_dir=work_dir,
            from_step_index=0,
            artifacts=dict(job_state.artifacts),
        )

    async def resume(self, job_id: str, from_step: str) -> PipelineResult:
        """Resume pipeline from a specific step (for retry after failure).

        Skips steps that already have artifacts stored in job state.

        Args:
            job_id: Unique job identifier.
            from_step: PipelineStep value to resume from.

        Returns:
            PipelineResult with the output video path and all artifacts.

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

        return await self._run_steps(
            job_id=job_id,
            url=url,
            work_dir=work_dir,
            from_step_index=from_step_index,
            artifacts=dict(job_state.artifacts),
        )

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
            except (PipelineError, CancellationError):
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
            transcription: TranscriptionResult = self._recognizer.recognize(vocals_path)
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
            synthesis: SynthesisResult = await self._synthesizer.synthesize(
                translation_data, work_dir
            )
            artifacts["vietnamese_audio"] = str(synthesis.audio_path)

        elif step == PipelineStep.COMPOSING_VIDEO:
            video_path_comp = Path(artifacts["video_path"])
            vietnamese_audio = Path(artifacts["vietnamese_audio"])
            background_audio = Path(artifacts["background_path"])
            output_dir = work_dir / "output"
            output_path: Path = self._composer.compose(
                video_path_comp, vietnamese_audio, background_audio, output_dir
            )
            artifacts["output_video"] = str(output_path)

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
