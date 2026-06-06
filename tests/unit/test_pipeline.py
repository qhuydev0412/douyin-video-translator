"""Unit tests for the TranslationPipeline orchestrator."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.job import JobState, JobStatus, PipelineStep
from app.models.pipeline import (
    DownloadResult,
    SynthesisResult,
    TranscriptionResult,
    TranscriptionSegment,
    TranslatedSegment,
    TranslationResult,
    VocalIsolationResult,
    VideoInfo,
)
from app.services.pipeline import (
    CancellationError,
    JobStoreProtocol,
    PipelineError,
    PipelineResult,
    STEP_ORDER,
    STEP_PROGRESS,
    TranslationPipeline,
)


# --- Fixtures ---


class FakeJobStore:
    """In-memory job store for testing."""

    def __init__(self, job_state: JobState):
        self._state = job_state
        self.updates: list[dict] = []

    def get_job(self, job_id: str) -> JobState:
        return self._state

    def update_job(self, job_id: str, **kwargs: object) -> None:
        self.updates.append({"job_id": job_id, **kwargs})
        # Apply updates to internal state for subsequent reads
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)


def _make_job_state(
    job_id: str = "test-job-123",
    url: str = "https://www.douyin.com/video/123",
    status: JobStatus = JobStatus.QUEUED,
    work_dir: str = "",
    artifacts: dict | None = None,
) -> JobState:
    now = datetime.now()
    return JobState(
        job_id=job_id,
        url=url,
        status=status,
        current_step=None,
        progress_percent=0,
        created_at=now,
        updated_at=now,
        work_dir=work_dir,
        artifacts=artifacts or {},
    )


def _make_pipeline(
    job_store: FakeJobStore,
    downloader: MagicMock | None = None,
    extractor: MagicMock | None = None,
    isolator: MagicMock | None = None,
    recognizer: MagicMock | None = None,
    translator: MagicMock | None = None,
    synthesizer: MagicMock | None = None,
    composer: MagicMock | None = None,
) -> TranslationPipeline:
    return TranslationPipeline(
        downloader=downloader or MagicMock(),
        extractor=extractor or MagicMock(),
        isolator=isolator or MagicMock(),
        recognizer=recognizer or MagicMock(),
        translator=translator or MagicMock(),
        synthesizer=synthesizer or AsyncMock(),
        composer=composer or MagicMock(),
        job_store=job_store,
    )


def _make_transcription() -> TranscriptionResult:
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(start=0.0, end=2.0, text="你好世界", speaker="speaker_1"),
        ],
        full_text="你好世界",
        language="zh",
        confidence=0.95,
    )


def _make_translation() -> TranslationResult:
    return TranslationResult(
        segments=[
            TranslatedSegment(
                start=0.0,
                end=2.0,
                original_text="你好世界",
                translated_text="Xin chào thế giới",
                speaker="speaker_1",
            ),
        ],
        full_text_original="你好世界",
        full_text_translated="Xin chào thế giới",
    )


# --- Tests ---


class TestPipelineExecuteFullRun:
    """Test execute() runs all 7 steps successfully."""

    @pytest.mark.asyncio
    async def test_execute_completes_all_steps(self, tmp_path: Path):
        """Full pipeline execution updates job state through all steps."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        job_state = _make_job_state(work_dir=str(work_dir))
        store = FakeJobStore(job_state)

        # Set up mocks for all services
        downloader = MagicMock()
        video_path = work_dir / "original.mp4"
        video_path.touch()
        downloader.download.return_value = DownloadResult(
            video_path=video_path,
            video_info=VideoInfo(
                duration_seconds=30.0,
                file_size_bytes=1000000,
                resolution="1080x1920",
            ),
        )

        extractor = MagicMock()
        audio_path = work_dir / "audio_full.wav"
        audio_path.touch()
        extractor.extract.return_value = audio_path

        isolator = MagicMock()
        vocals_path = work_dir / "vocals.wav"
        vocals_path.touch()
        background_path = work_dir / "background.wav"
        background_path.touch()
        isolator.isolate.return_value = VocalIsolationResult(
            vocals_path=vocals_path,
            background_path=background_path,
        )

        recognizer = MagicMock()
        recognizer.recognize.return_value = _make_transcription()

        translator = MagicMock()
        translator.translate.return_value = _make_translation()

        synthesizer = AsyncMock()
        vietnamese_audio = work_dir / "vietnamese_audio.wav"
        vietnamese_audio.touch()
        synthesizer.synthesize.return_value = SynthesisResult(
            audio_path=vietnamese_audio,
            segment_audios=[],
        )

        composer = MagicMock()
        output_dir = work_dir / "output"
        output_dir.mkdir(parents=True)
        output_path = output_dir / "output.mp4"
        output_path.touch()
        composer.compose.return_value = output_path

        pipeline = _make_pipeline(
            job_store=store,
            downloader=downloader,
            extractor=extractor,
            isolator=isolator,
            recognizer=recognizer,
            translator=translator,
            synthesizer=synthesizer,
            composer=composer,
        )

        result = await pipeline.execute("test-job-123", "https://www.douyin.com/video/123")

        assert isinstance(result, PipelineResult)
        assert result.output_path == output_path
        assert result.job_id == "test-job-123"
        assert "output_video" in result.artifacts
        assert job_state.status == JobStatus.COMPLETED
        assert job_state.progress_percent == 100

    @pytest.mark.asyncio
    async def test_execute_updates_progress_per_step(self, tmp_path: Path):
        """Each step updates progress_percent in the job store."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        job_state = _make_job_state(work_dir=str(work_dir))
        store = FakeJobStore(job_state)

        # Set up mocks
        downloader = MagicMock()
        video_path = work_dir / "original.mp4"
        video_path.touch()
        downloader.download.return_value = DownloadResult(
            video_path=video_path,
            video_info=VideoInfo(
                duration_seconds=10.0, file_size_bytes=500, resolution="720x1280"
            ),
        )

        extractor = MagicMock()
        audio_path = work_dir / "audio_full.wav"
        audio_path.touch()
        extractor.extract.return_value = audio_path

        isolator = MagicMock()
        vocals = work_dir / "vocals.wav"
        vocals.touch()
        bg = work_dir / "background.wav"
        bg.touch()
        isolator.isolate.return_value = VocalIsolationResult(
            vocals_path=vocals, background_path=bg
        )

        recognizer = MagicMock()
        recognizer.recognize.return_value = _make_transcription()

        translator = MagicMock()
        translator.translate.return_value = _make_translation()

        synthesizer = AsyncMock()
        vi_audio = work_dir / "vietnamese_audio.wav"
        vi_audio.touch()
        synthesizer.synthesize.return_value = SynthesisResult(
            audio_path=vi_audio, segment_audios=[]
        )

        composer = MagicMock()
        out_dir = work_dir / "output"
        out_dir.mkdir()
        out = out_dir / "output.mp4"
        out.touch()
        composer.compose.return_value = out

        pipeline = _make_pipeline(
            job_store=store,
            downloader=downloader,
            extractor=extractor,
            isolator=isolator,
            recognizer=recognizer,
            translator=translator,
            synthesizer=synthesizer,
            composer=composer,
        )

        await pipeline.execute("test-job-123", "https://www.douyin.com/video/123")

        # Check that progress was updated for each step
        progress_updates = [
            u["progress_percent"] for u in store.updates if "progress_percent" in u
        ]
        # Should have start and end percentages for each of the 7 steps + final 100
        assert 0 in progress_updates  # DOWNLOADING start
        assert 15 in progress_updates  # DOWNLOADING end / EXTRACTING start
        assert 25 in progress_updates  # EXTRACTING end
        assert 40 in progress_updates  # ISOLATING end
        assert 60 in progress_updates  # RECOGNIZING end
        assert 75 in progress_updates  # TRANSLATING end
        assert 90 in progress_updates  # SYNTHESIZING end
        assert 100 in progress_updates  # COMPOSING end + final completion


class TestPipelineCancellation:
    """Test cancellation check between steps."""

    @pytest.mark.asyncio
    async def test_cancelled_before_step_raises_error(self, tmp_path: Path):
        """Pipeline raises CancellationError if job is cancelled."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        job_state = _make_job_state(
            status=JobStatus.CANCELLED,
            work_dir=str(work_dir),
        )
        store = FakeJobStore(job_state)
        pipeline = _make_pipeline(job_store=store)

        with pytest.raises(CancellationError) as exc_info:
            await pipeline.execute("test-job-123", "https://www.douyin.com/video/123")

        assert exc_info.value.step == PipelineStep.DOWNLOADING

    @pytest.mark.asyncio
    async def test_cancelled_mid_pipeline(self, tmp_path: Path):
        """Pipeline stops when job becomes cancelled between steps."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        job_state = _make_job_state(work_dir=str(work_dir))
        store = FakeJobStore(job_state)

        # Download succeeds, then cancel before extracting
        downloader = MagicMock()
        video_path = work_dir / "original.mp4"
        video_path.touch()
        downloader.download.return_value = DownloadResult(
            video_path=video_path,
            video_info=VideoInfo(
                duration_seconds=10.0, file_size_bytes=500, resolution="720x1280"
            ),
        )

        call_count = 0
        original_get_job = store.get_job

        def get_job_with_cancel(job_id: str) -> JobState:
            nonlocal call_count
            call_count += 1
            # Cancel after the first cancellation check (DOWNLOADING passes),
            # so the next check for EXTRACTING_AUDIO triggers cancel.
            # Call 1: execute() reads initial state
            # Call 2: _check_cancellation for DOWNLOADING (should pass)
            # Call 3: _check_cancellation for EXTRACTING_AUDIO (should cancel)
            if call_count >= 3:
                job_state.status = JobStatus.CANCELLED
            return original_get_job(job_id)

        store.get_job = get_job_with_cancel

        pipeline = _make_pipeline(job_store=store, downloader=downloader)

        with pytest.raises(CancellationError) as exc_info:
            await pipeline.execute("test-job-123", "https://www.douyin.com/video/123")

        assert exc_info.value.step == PipelineStep.EXTRACTING_AUDIO


class TestPipelineFailure:
    """Test error handling and artifact preservation on failure."""

    @pytest.mark.asyncio
    async def test_step_failure_raises_pipeline_error(self, tmp_path: Path):
        """A failing step raises PipelineError with step info."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        job_state = _make_job_state(work_dir=str(work_dir))
        store = FakeJobStore(job_state)

        downloader = MagicMock()
        downloader.download.side_effect = RuntimeError("Network timeout")

        pipeline = _make_pipeline(job_store=store, downloader=downloader)

        with pytest.raises(PipelineError) as exc_info:
            await pipeline.execute("test-job-123", "https://www.douyin.com/video/123")

        assert exc_info.value.step == PipelineStep.DOWNLOADING
        assert "Network timeout" in exc_info.value.message
        assert exc_info.value.retryable is True

    @pytest.mark.asyncio
    async def test_failure_preserves_prior_artifacts(self, tmp_path: Path):
        """Artifacts from completed steps are preserved when a later step fails."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        job_state = _make_job_state(work_dir=str(work_dir))
        store = FakeJobStore(job_state)

        # Download succeeds
        downloader = MagicMock()
        video_path = work_dir / "original.mp4"
        video_path.touch()
        downloader.download.return_value = DownloadResult(
            video_path=video_path,
            video_info=VideoInfo(
                duration_seconds=10.0, file_size_bytes=500, resolution="720x1280"
            ),
        )

        # Extractor fails
        extractor = MagicMock()
        extractor.extract.side_effect = RuntimeError("FFmpeg not found")

        pipeline = _make_pipeline(
            job_store=store,
            downloader=downloader,
            extractor=extractor,
        )

        with pytest.raises(PipelineError) as exc_info:
            await pipeline.execute("test-job-123", "https://www.douyin.com/video/123")

        assert exc_info.value.step == PipelineStep.EXTRACTING_AUDIO

        # The video_path artifact from downloading should have been stored
        # via the update_job call after the download step
        artifact_updates = [u for u in store.updates if "artifacts" in u]
        assert len(artifact_updates) >= 1
        assert "video_path" in artifact_updates[0]["artifacts"]


class TestPipelineResume:
    """Test resume() functionality for retry from a failed step."""

    @pytest.mark.asyncio
    async def test_resume_from_specific_step(self, tmp_path: Path):
        """resume() starts from the specified step, not from beginning."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        # Pre-existing artifacts from previous partial run
        video_path = work_dir / "original.mp4"
        video_path.touch()
        audio_path = work_dir / "audio_full.wav"
        audio_path.touch()

        artifacts = {
            "video_path": str(video_path),
            "audio_path": str(audio_path),
        }

        job_state = _make_job_state(
            work_dir=str(work_dir),
            artifacts=artifacts,
        )
        store = FakeJobStore(job_state)

        # Isolator succeeds on resume
        isolator = MagicMock()
        vocals = work_dir / "vocals.wav"
        vocals.touch()
        bg = work_dir / "background.wav"
        bg.touch()
        isolator.isolate.return_value = VocalIsolationResult(
            vocals_path=vocals, background_path=bg
        )

        recognizer = MagicMock()
        recognizer.recognize.return_value = _make_transcription()

        translator = MagicMock()
        translator.translate.return_value = _make_translation()

        synthesizer = AsyncMock()
        vi_audio = work_dir / "vietnamese_audio.wav"
        vi_audio.touch()
        synthesizer.synthesize.return_value = SynthesisResult(
            audio_path=vi_audio, segment_audios=[]
        )

        composer = MagicMock()
        out_dir = work_dir / "output"
        out_dir.mkdir()
        out = out_dir / "output.mp4"
        out.touch()
        composer.compose.return_value = out

        # Downloader should NOT be called since we resume from isolating_vocals
        downloader = MagicMock()
        extractor = MagicMock()

        pipeline = _make_pipeline(
            job_store=store,
            downloader=downloader,
            extractor=extractor,
            isolator=isolator,
            recognizer=recognizer,
            translator=translator,
            synthesizer=synthesizer,
            composer=composer,
        )

        result = await pipeline.resume("test-job-123", "isolating_vocals")

        assert isinstance(result, PipelineResult)
        assert result.output_path == out
        # Downloader and extractor should NOT have been called
        downloader.download.assert_not_called()
        extractor.extract.assert_not_called()
        # Isolator should have been called
        isolator.isolate.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_invalid_step_raises_value_error(self, tmp_path: Path):
        """resume() raises ValueError for invalid step name."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        job_state = _make_job_state(work_dir=str(work_dir))
        store = FakeJobStore(job_state)
        pipeline = _make_pipeline(job_store=store)

        with pytest.raises(ValueError, match="Invalid pipeline step"):
            await pipeline.resume("test-job-123", "invalid_step")


class TestPipelineRealTimeStatusUpdates:
    """Test real-time status updates per requirement 7.2."""

    @pytest.mark.asyncio
    async def test_current_step_updated_for_each_step(self, tmp_path: Path):
        """Job store receives current_step updates for every pipeline step."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        job_state = _make_job_state(work_dir=str(work_dir))
        store = FakeJobStore(job_state)

        # Set up mocks for all services
        downloader = MagicMock()
        video_path = work_dir / "original.mp4"
        video_path.touch()
        downloader.download.return_value = DownloadResult(
            video_path=video_path,
            video_info=VideoInfo(
                duration_seconds=10.0, file_size_bytes=500, resolution="720x1280"
            ),
        )

        extractor = MagicMock()
        audio_path = work_dir / "audio_full.wav"
        audio_path.touch()
        extractor.extract.return_value = audio_path

        isolator = MagicMock()
        vocals = work_dir / "vocals.wav"
        vocals.touch()
        bg = work_dir / "background.wav"
        bg.touch()
        isolator.isolate.return_value = VocalIsolationResult(
            vocals_path=vocals, background_path=bg
        )

        recognizer = MagicMock()
        recognizer.recognize.return_value = _make_transcription()

        translator = MagicMock()
        translator.translate.return_value = _make_translation()

        synthesizer = AsyncMock()
        vi_audio = work_dir / "vietnamese_audio.wav"
        vi_audio.touch()
        synthesizer.synthesize.return_value = SynthesisResult(
            audio_path=vi_audio, segment_audios=[]
        )

        composer = MagicMock()
        out_dir = work_dir / "output"
        out_dir.mkdir()
        out = out_dir / "output.mp4"
        out.touch()
        composer.compose.return_value = out

        pipeline = _make_pipeline(
            job_store=store,
            downloader=downloader,
            extractor=extractor,
            isolator=isolator,
            recognizer=recognizer,
            translator=translator,
            synthesizer=synthesizer,
            composer=composer,
        )

        await pipeline.execute("test-job-123", "https://www.douyin.com/video/123")

        # Extract all current_step updates
        step_updates = [
            u["current_step"]
            for u in store.updates
            if "current_step" in u
        ]
        # All 7 steps should be reported in order
        assert step_updates == list(STEP_ORDER)

    @pytest.mark.asyncio
    async def test_status_transitions_queued_to_processing_to_completed(self, tmp_path: Path):
        """Job status transitions: QUEUED → PROCESSING → COMPLETED."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        job_state = _make_job_state(work_dir=str(work_dir))
        store = FakeJobStore(job_state)

        # Set up all mocks
        downloader = MagicMock()
        video_path = work_dir / "original.mp4"
        video_path.touch()
        downloader.download.return_value = DownloadResult(
            video_path=video_path,
            video_info=VideoInfo(
                duration_seconds=10.0, file_size_bytes=500, resolution="720x1280"
            ),
        )

        extractor = MagicMock()
        audio_path = work_dir / "audio_full.wav"
        audio_path.touch()
        extractor.extract.return_value = audio_path

        isolator = MagicMock()
        vocals = work_dir / "vocals.wav"
        vocals.touch()
        bg = work_dir / "background.wav"
        bg.touch()
        isolator.isolate.return_value = VocalIsolationResult(
            vocals_path=vocals, background_path=bg
        )

        recognizer = MagicMock()
        recognizer.recognize.return_value = _make_transcription()

        translator = MagicMock()
        translator.translate.return_value = _make_translation()

        synthesizer = AsyncMock()
        vi_audio = work_dir / "vietnamese_audio.wav"
        vi_audio.touch()
        synthesizer.synthesize.return_value = SynthesisResult(
            audio_path=vi_audio, segment_audios=[]
        )

        composer = MagicMock()
        out_dir = work_dir / "output"
        out_dir.mkdir()
        out = out_dir / "output.mp4"
        out.touch()
        composer.compose.return_value = out

        pipeline = _make_pipeline(
            job_store=store,
            downloader=downloader,
            extractor=extractor,
            isolator=isolator,
            recognizer=recognizer,
            translator=translator,
            synthesizer=synthesizer,
            composer=composer,
        )

        await pipeline.execute("test-job-123", "https://www.douyin.com/video/123")

        # Extract status updates
        status_updates = [
            u["status"] for u in store.updates if "status" in u
        ]
        # Should transition to PROCESSING at start, then COMPLETED at end
        assert JobStatus.PROCESSING in status_updates
        assert JobStatus.COMPLETED in status_updates
        assert status_updates[0] == JobStatus.PROCESSING
        assert status_updates[-1] == JobStatus.COMPLETED


class TestPipelineCancellationAtVariousSteps:
    """Test cancellation at different pipeline steps (extends requirement 7.3)."""

    @pytest.mark.asyncio
    async def test_cancelled_before_translating(self, tmp_path: Path):
        """Pipeline stops when cancelled before the translating step."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        job_state = _make_job_state(work_dir=str(work_dir))
        store = FakeJobStore(job_state)

        # Set up mocks for steps before translating
        downloader = MagicMock()
        video_path = work_dir / "original.mp4"
        video_path.touch()
        downloader.download.return_value = DownloadResult(
            video_path=video_path,
            video_info=VideoInfo(
                duration_seconds=10.0, file_size_bytes=500, resolution="720x1280"
            ),
        )

        extractor = MagicMock()
        audio_path = work_dir / "audio_full.wav"
        audio_path.touch()
        extractor.extract.return_value = audio_path

        isolator = MagicMock()
        vocals = work_dir / "vocals.wav"
        vocals.touch()
        bg = work_dir / "background.wav"
        bg.touch()
        isolator.isolate.return_value = VocalIsolationResult(
            vocals_path=vocals, background_path=bg
        )

        recognizer = MagicMock()
        recognizer.recognize.return_value = _make_transcription()

        # Cancel after recognizing_speech completes, before translating
        call_count = 0
        original_get_job = store.get_job

        def get_job_with_cancel(job_id: str) -> JobState:
            nonlocal call_count
            call_count += 1
            # get_job calls:
            # 1: execute() initial read
            # 2: _check_cancellation for DOWNLOADING
            # 3: _check_cancellation for EXTRACTING_AUDIO
            # 4: _check_cancellation for ISOLATING_VOCALS
            # 5: _check_cancellation for RECOGNIZING_SPEECH
            # 6: _check_cancellation for TRANSLATING  <-- cancel here
            if call_count >= 6:
                job_state.status = JobStatus.CANCELLED
            return original_get_job(job_id)

        store.get_job = get_job_with_cancel

        translator = MagicMock()
        pipeline = _make_pipeline(
            job_store=store,
            downloader=downloader,
            extractor=extractor,
            isolator=isolator,
            recognizer=recognizer,
            translator=translator,
        )

        with pytest.raises(CancellationError) as exc_info:
            await pipeline.execute("test-job-123", "https://www.douyin.com/video/123")

        assert exc_info.value.step == PipelineStep.TRANSLATING
        # Translator should NOT have been called
        translator.translate.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancelled_before_composing_video(self, tmp_path: Path):
        """Pipeline stops when cancelled before the final composing step."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        job_state = _make_job_state(work_dir=str(work_dir))
        store = FakeJobStore(job_state)

        # Set up all mocks
        downloader = MagicMock()
        video_path = work_dir / "original.mp4"
        video_path.touch()
        downloader.download.return_value = DownloadResult(
            video_path=video_path,
            video_info=VideoInfo(
                duration_seconds=10.0, file_size_bytes=500, resolution="720x1280"
            ),
        )

        extractor = MagicMock()
        audio_path = work_dir / "audio_full.wav"
        audio_path.touch()
        extractor.extract.return_value = audio_path

        isolator = MagicMock()
        vocals = work_dir / "vocals.wav"
        vocals.touch()
        bg = work_dir / "background.wav"
        bg.touch()
        isolator.isolate.return_value = VocalIsolationResult(
            vocals_path=vocals, background_path=bg
        )

        recognizer = MagicMock()
        recognizer.recognize.return_value = _make_transcription()

        translator = MagicMock()
        translator.translate.return_value = _make_translation()

        synthesizer = AsyncMock()
        vi_audio = work_dir / "vietnamese_audio.wav"
        vi_audio.touch()
        synthesizer.synthesize.return_value = SynthesisResult(
            audio_path=vi_audio, segment_audios=[]
        )

        # Cancel after synthesizing completes, before composing
        call_count = 0
        original_get_job = store.get_job

        def get_job_with_cancel(job_id: str) -> JobState:
            nonlocal call_count
            call_count += 1
            # get_job calls:
            # 1: execute() initial read
            # 2: _check_cancellation for DOWNLOADING
            # 3: _check_cancellation for EXTRACTING_AUDIO
            # 4: _check_cancellation for ISOLATING_VOCALS
            # 5: _check_cancellation for RECOGNIZING_SPEECH
            # 6: _check_cancellation for TRANSLATING
            # 7: _check_cancellation for SYNTHESIZING_VOICE
            # 8: _check_cancellation for COMPOSING_VIDEO  <-- cancel here
            if call_count >= 8:
                job_state.status = JobStatus.CANCELLED
            return original_get_job(job_id)

        store.get_job = get_job_with_cancel

        composer = MagicMock()
        pipeline = _make_pipeline(
            job_store=store,
            downloader=downloader,
            extractor=extractor,
            isolator=isolator,
            recognizer=recognizer,
            translator=translator,
            synthesizer=synthesizer,
            composer=composer,
        )

        with pytest.raises(CancellationError) as exc_info:
            await pipeline.execute("test-job-123", "https://www.douyin.com/video/123")

        assert exc_info.value.step == PipelineStep.COMPOSING_VIDEO
        composer.compose.assert_not_called()


class TestPipelineResumeFromEachStep:
    """Test resume from various steps to verify retry works at each point."""

    @pytest.mark.asyncio
    async def test_resume_from_extracting_audio(self, tmp_path: Path):
        """Resume from extracting_audio skips downloading."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        video_path = work_dir / "original.mp4"
        video_path.touch()

        artifacts = {"video_path": str(video_path)}

        job_state = _make_job_state(work_dir=str(work_dir), artifacts=artifacts)
        store = FakeJobStore(job_state)

        extractor = MagicMock()
        audio_path = work_dir / "audio_full.wav"
        audio_path.touch()
        extractor.extract.return_value = audio_path

        isolator = MagicMock()
        vocals = work_dir / "vocals.wav"
        vocals.touch()
        bg = work_dir / "background.wav"
        bg.touch()
        isolator.isolate.return_value = VocalIsolationResult(
            vocals_path=vocals, background_path=bg
        )

        recognizer = MagicMock()
        recognizer.recognize.return_value = _make_transcription()

        translator = MagicMock()
        translator.translate.return_value = _make_translation()

        synthesizer = AsyncMock()
        vi_audio = work_dir / "vietnamese_audio.wav"
        vi_audio.touch()
        synthesizer.synthesize.return_value = SynthesisResult(
            audio_path=vi_audio, segment_audios=[]
        )

        composer = MagicMock()
        out_dir = work_dir / "output"
        out_dir.mkdir()
        out = out_dir / "output.mp4"
        out.touch()
        composer.compose.return_value = out

        downloader = MagicMock()

        pipeline = _make_pipeline(
            job_store=store,
            downloader=downloader,
            extractor=extractor,
            isolator=isolator,
            recognizer=recognizer,
            translator=translator,
            synthesizer=synthesizer,
            composer=composer,
        )

        result = await pipeline.resume("test-job-123", "extracting_audio")

        assert isinstance(result, PipelineResult)
        downloader.download.assert_not_called()
        extractor.extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_from_composing_video(self, tmp_path: Path):
        """Resume from composing_video only runs the final step."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        video_path = work_dir / "original.mp4"
        video_path.touch()
        vi_audio = work_dir / "vietnamese_audio.wav"
        vi_audio.touch()
        bg = work_dir / "background.wav"
        bg.touch()

        artifacts = {
            "video_path": str(video_path),
            "audio_path": str(work_dir / "audio.wav"),
            "vocals_path": str(work_dir / "vocals.wav"),
            "background_path": str(bg),
            "transcription_path": str(work_dir / "transcription.json"),
            "translation_path": str(work_dir / "translation.json"),
            "vietnamese_audio": str(vi_audio),
        }

        job_state = _make_job_state(work_dir=str(work_dir), artifacts=artifacts)
        store = FakeJobStore(job_state)

        composer = MagicMock()
        out_dir = work_dir / "output"
        out_dir.mkdir()
        out = out_dir / "output.mp4"
        out.touch()
        composer.compose.return_value = out

        downloader = MagicMock()
        extractor = MagicMock()
        isolator = MagicMock()
        recognizer = MagicMock()
        translator = MagicMock()
        synthesizer = AsyncMock()

        pipeline = _make_pipeline(
            job_store=store,
            downloader=downloader,
            extractor=extractor,
            isolator=isolator,
            recognizer=recognizer,
            translator=translator,
            synthesizer=synthesizer,
            composer=composer,
        )

        result = await pipeline.resume("test-job-123", "composing_video")

        assert isinstance(result, PipelineResult)
        assert result.output_path == out
        # No earlier steps should be called
        downloader.download.assert_not_called()
        extractor.extract.assert_not_called()
        isolator.isolate.assert_not_called()
        recognizer.recognize.assert_not_called()
        translator.translate.assert_not_called()
        synthesizer.synthesize.assert_not_called()
        # Only composer should be called
        composer.compose.assert_called_once()


class TestPipelineProgressTracking:
    """Test progress tracking details per requirement 7.1."""

    @pytest.mark.asyncio
    async def test_progress_never_decreases(self, tmp_path: Path):
        """Progress percent is monotonically non-decreasing during execution."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        job_state = _make_job_state(work_dir=str(work_dir))
        store = FakeJobStore(job_state)

        # Set up all mocks
        downloader = MagicMock()
        video_path = work_dir / "original.mp4"
        video_path.touch()
        downloader.download.return_value = DownloadResult(
            video_path=video_path,
            video_info=VideoInfo(
                duration_seconds=10.0, file_size_bytes=500, resolution="720x1280"
            ),
        )

        extractor = MagicMock()
        audio_path = work_dir / "audio_full.wav"
        audio_path.touch()
        extractor.extract.return_value = audio_path

        isolator = MagicMock()
        vocals = work_dir / "vocals.wav"
        vocals.touch()
        bg = work_dir / "background.wav"
        bg.touch()
        isolator.isolate.return_value = VocalIsolationResult(
            vocals_path=vocals, background_path=bg
        )

        recognizer = MagicMock()
        recognizer.recognize.return_value = _make_transcription()

        translator = MagicMock()
        translator.translate.return_value = _make_translation()

        synthesizer = AsyncMock()
        vi_audio = work_dir / "vietnamese_audio.wav"
        vi_audio.touch()
        synthesizer.synthesize.return_value = SynthesisResult(
            audio_path=vi_audio, segment_audios=[]
        )

        composer = MagicMock()
        out_dir = work_dir / "output"
        out_dir.mkdir()
        out = out_dir / "output.mp4"
        out.touch()
        composer.compose.return_value = out

        pipeline = _make_pipeline(
            job_store=store,
            downloader=downloader,
            extractor=extractor,
            isolator=isolator,
            recognizer=recognizer,
            translator=translator,
            synthesizer=synthesizer,
            composer=composer,
        )

        await pipeline.execute("test-job-123", "https://www.douyin.com/video/123")

        # Collect all progress updates in order
        progress_values = [
            u["progress_percent"]
            for u in store.updates
            if "progress_percent" in u
        ]
        # Progress should never decrease
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i - 1], (
                f"Progress decreased from {progress_values[i-1]} to {progress_values[i]}"
            )
        # Final progress should be 100
        assert progress_values[-1] == 100

    @pytest.mark.asyncio
    async def test_each_step_reports_start_and_end_progress(self, tmp_path: Path):
        """Each step reports both its start and end progress percentage."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        job_state = _make_job_state(work_dir=str(work_dir))
        store = FakeJobStore(job_state)

        # Set up all mocks
        downloader = MagicMock()
        video_path = work_dir / "original.mp4"
        video_path.touch()
        downloader.download.return_value = DownloadResult(
            video_path=video_path,
            video_info=VideoInfo(
                duration_seconds=10.0, file_size_bytes=500, resolution="720x1280"
            ),
        )

        extractor = MagicMock()
        audio_path = work_dir / "audio_full.wav"
        audio_path.touch()
        extractor.extract.return_value = audio_path

        isolator = MagicMock()
        vocals = work_dir / "vocals.wav"
        vocals.touch()
        bg = work_dir / "background.wav"
        bg.touch()
        isolator.isolate.return_value = VocalIsolationResult(
            vocals_path=vocals, background_path=bg
        )

        recognizer = MagicMock()
        recognizer.recognize.return_value = _make_transcription()

        translator = MagicMock()
        translator.translate.return_value = _make_translation()

        synthesizer = AsyncMock()
        vi_audio = work_dir / "vietnamese_audio.wav"
        vi_audio.touch()
        synthesizer.synthesize.return_value = SynthesisResult(
            audio_path=vi_audio, segment_audios=[]
        )

        composer = MagicMock()
        out_dir = work_dir / "output"
        out_dir.mkdir()
        out = out_dir / "output.mp4"
        out.touch()
        composer.compose.return_value = out

        pipeline = _make_pipeline(
            job_store=store,
            downloader=downloader,
            extractor=extractor,
            isolator=isolator,
            recognizer=recognizer,
            translator=translator,
            synthesizer=synthesizer,
            composer=composer,
        )

        await pipeline.execute("test-job-123", "https://www.douyin.com/video/123")

        # Each step should have start and end progress in the STEP_PROGRESS map
        progress_values = [
            u["progress_percent"]
            for u in store.updates
            if "progress_percent" in u
        ]
        for step in STEP_ORDER:
            start, end = STEP_PROGRESS[step]
            assert start in progress_values, f"Missing start progress {start} for {step}"
            assert end in progress_values, f"Missing end progress {end} for {step}"


class TestPipelineFailureReporting:
    """Test failed step reporting and retry allowance per requirement 7.4."""

    @pytest.mark.asyncio
    async def test_failure_at_translation_reports_correct_step(self, tmp_path: Path):
        """Failure at translation step reports PipelineStep.TRANSLATING."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        job_state = _make_job_state(work_dir=str(work_dir))
        store = FakeJobStore(job_state)

        downloader = MagicMock()
        video_path = work_dir / "original.mp4"
        video_path.touch()
        downloader.download.return_value = DownloadResult(
            video_path=video_path,
            video_info=VideoInfo(
                duration_seconds=10.0, file_size_bytes=500, resolution="720x1280"
            ),
        )

        extractor = MagicMock()
        audio_path = work_dir / "audio_full.wav"
        audio_path.touch()
        extractor.extract.return_value = audio_path

        isolator = MagicMock()
        vocals = work_dir / "vocals.wav"
        vocals.touch()
        bg = work_dir / "background.wav"
        bg.touch()
        isolator.isolate.return_value = VocalIsolationResult(
            vocals_path=vocals, background_path=bg
        )

        recognizer = MagicMock()
        recognizer.recognize.return_value = _make_transcription()

        # Translator fails
        translator = MagicMock()
        translator.translate.side_effect = RuntimeError("Google API quota exceeded")

        pipeline = _make_pipeline(
            job_store=store,
            downloader=downloader,
            extractor=extractor,
            isolator=isolator,
            recognizer=recognizer,
            translator=translator,
        )

        with pytest.raises(PipelineError) as exc_info:
            await pipeline.execute("test-job-123", "https://www.douyin.com/video/123")

        assert exc_info.value.step == PipelineStep.TRANSLATING
        assert "Google API quota exceeded" in exc_info.value.message
        assert exc_info.value.retryable is True

    @pytest.mark.asyncio
    async def test_resume_after_failure_skips_completed_steps(self, tmp_path: Path):
        """After failure at translation, resume from translating uses prior artifacts."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        # Artifacts from steps before translation
        video_path = work_dir / "original.mp4"
        video_path.touch()
        audio_path = work_dir / "audio_full.wav"
        audio_path.touch()
        vocals = work_dir / "vocals.wav"
        vocals.touch()
        bg = work_dir / "background.wav"
        bg.touch()

        # Create transcription artifact
        transcription_path = work_dir / "transcription.json"
        import json
        transcription_data = {
            "segments": [{"start": 0.0, "end": 2.0, "text": "你好", "speaker": None}],
            "full_text": "你好",
            "language": "zh",
            "confidence": 0.9,
        }
        transcription_path.write_text(
            json.dumps(transcription_data, ensure_ascii=False), encoding="utf-8"
        )

        artifacts = {
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "vocals_path": str(vocals),
            "background_path": str(bg),
            "transcription_path": str(transcription_path),
        }

        job_state = _make_job_state(work_dir=str(work_dir), artifacts=artifacts)
        store = FakeJobStore(job_state)

        # All prior steps should not be called
        downloader = MagicMock()
        extractor = MagicMock()
        isolator = MagicMock()
        recognizer = MagicMock()

        # Translation succeeds on retry
        translator = MagicMock()
        translator.translate.return_value = _make_translation()

        synthesizer = AsyncMock()
        vi_audio = work_dir / "vietnamese_audio.wav"
        vi_audio.touch()
        synthesizer.synthesize.return_value = SynthesisResult(
            audio_path=vi_audio, segment_audios=[]
        )

        composer = MagicMock()
        out_dir = work_dir / "output"
        out_dir.mkdir()
        out = out_dir / "output.mp4"
        out.touch()
        composer.compose.return_value = out

        pipeline = _make_pipeline(
            job_store=store,
            downloader=downloader,
            extractor=extractor,
            isolator=isolator,
            recognizer=recognizer,
            translator=translator,
            synthesizer=synthesizer,
            composer=composer,
        )

        result = await pipeline.resume("test-job-123", "translating")

        assert isinstance(result, PipelineResult)
        # Steps before translation should NOT be called
        downloader.download.assert_not_called()
        extractor.extract.assert_not_called()
        isolator.isolate.assert_not_called()
        recognizer.recognize.assert_not_called()
        # Translation and later steps should be called
        translator.translate.assert_called_once()
        synthesizer.synthesize.assert_called_once()
        composer.compose.assert_called_once()


class TestPipelineStepOrder:
    """Test pipeline step ordering and progress mapping."""

    def test_step_order_has_7_steps(self):
        """Pipeline has exactly 7 steps."""
        assert len(STEP_ORDER) == 7

    def test_step_progress_is_monotonic(self):
        """Progress percentages increase monotonically."""
        prev_end = 0
        for step in STEP_ORDER:
            start, end = STEP_PROGRESS[step]
            assert start == prev_end, f"Step {step}: start {start} != prev_end {prev_end}"
            assert end > start, f"Step {step}: end {end} <= start {start}"
            prev_end = end
        assert prev_end == 100

    def test_all_steps_have_progress(self):
        """Every step in STEP_ORDER has a corresponding progress mapping."""
        for step in STEP_ORDER:
            assert step in STEP_PROGRESS


class TestPipelineResumeAdditionalSteps:
    """Test resume from remaining steps not covered by TestPipelineResumeFromEachStep."""

    @pytest.mark.asyncio
    async def test_resume_from_recognizing_speech(self, tmp_path: Path):
        """Resume from recognizing_speech skips download, extract, and isolate."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        video_path = work_dir / "original.mp4"
        video_path.touch()
        vocals = work_dir / "vocals.wav"
        vocals.touch()
        bg = work_dir / "background.wav"
        bg.touch()

        artifacts = {
            "video_path": str(video_path),
            "audio_path": str(work_dir / "audio.wav"),
            "vocals_path": str(vocals),
            "background_path": str(bg),
        }

        job_state = _make_job_state(work_dir=str(work_dir), artifacts=artifacts)
        store = FakeJobStore(job_state)

        recognizer = MagicMock()
        recognizer.recognize.return_value = _make_transcription()

        translator = MagicMock()
        translator.translate.return_value = _make_translation()

        synthesizer = AsyncMock()
        vi_audio = work_dir / "vietnamese_audio.wav"
        vi_audio.touch()
        synthesizer.synthesize.return_value = SynthesisResult(
            audio_path=vi_audio, segment_audios=[]
        )

        composer = MagicMock()
        out_dir = work_dir / "output"
        out_dir.mkdir()
        out = out_dir / "output.mp4"
        out.touch()
        composer.compose.return_value = out

        downloader = MagicMock()
        extractor = MagicMock()
        isolator = MagicMock()

        pipeline = _make_pipeline(
            job_store=store,
            downloader=downloader,
            extractor=extractor,
            isolator=isolator,
            recognizer=recognizer,
            translator=translator,
            synthesizer=synthesizer,
            composer=composer,
        )

        result = await pipeline.resume("test-job-123", "recognizing_speech")

        assert isinstance(result, PipelineResult)
        downloader.download.assert_not_called()
        extractor.extract.assert_not_called()
        isolator.isolate.assert_not_called()
        recognizer.recognize.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_from_synthesizing_voice(self, tmp_path: Path):
        """Resume from synthesizing_voice skips all earlier steps."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        video_path = work_dir / "original.mp4"
        video_path.touch()
        bg = work_dir / "background.wav"
        bg.touch()

        # Create translation artifact
        translation_path = work_dir / "translation.json"
        translation_data = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "original_text": "你好",
                    "translated_text": "Xin chào",
                    "speaker": None,
                }
            ],
            "full_text_original": "你好",
            "full_text_translated": "Xin chào",
        }
        translation_path.write_text(
            json.dumps(translation_data, ensure_ascii=False), encoding="utf-8"
        )

        artifacts = {
            "video_path": str(video_path),
            "audio_path": str(work_dir / "audio.wav"),
            "vocals_path": str(work_dir / "vocals.wav"),
            "background_path": str(bg),
            "transcription_path": str(work_dir / "transcription.json"),
            "translation_path": str(translation_path),
        }

        job_state = _make_job_state(work_dir=str(work_dir), artifacts=artifacts)
        store = FakeJobStore(job_state)

        synthesizer = AsyncMock()
        vi_audio = work_dir / "vietnamese_audio.wav"
        vi_audio.touch()
        synthesizer.synthesize.return_value = SynthesisResult(
            audio_path=vi_audio, segment_audios=[]
        )

        composer = MagicMock()
        out_dir = work_dir / "output"
        out_dir.mkdir()
        out = out_dir / "output.mp4"
        out.touch()
        composer.compose.return_value = out

        downloader = MagicMock()
        extractor = MagicMock()
        isolator = MagicMock()
        recognizer = MagicMock()
        translator = MagicMock()

        pipeline = _make_pipeline(
            job_store=store,
            downloader=downloader,
            extractor=extractor,
            isolator=isolator,
            recognizer=recognizer,
            translator=translator,
            synthesizer=synthesizer,
            composer=composer,
        )

        result = await pipeline.resume("test-job-123", "synthesizing_voice")

        assert isinstance(result, PipelineResult)
        downloader.download.assert_not_called()
        extractor.extract.assert_not_called()
        isolator.isolate.assert_not_called()
        recognizer.recognize.assert_not_called()
        translator.translate.assert_not_called()
        synthesizer.synthesize.assert_called_once()
        composer.compose.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_from_downloading(self, tmp_path: Path):
        """Resume from downloading runs all steps (equivalent to full execute)."""
        work_dir = tmp_path / "job-work"
        work_dir.mkdir(parents=True)

        job_state = _make_job_state(work_dir=str(work_dir))
        store = FakeJobStore(job_state)

        downloader = MagicMock()
        video_path = work_dir / "original.mp4"
        video_path.touch()
        downloader.download.return_value = DownloadResult(
            video_path=video_path,
            video_info=VideoInfo(
                duration_seconds=10.0, file_size_bytes=500, resolution="720x1280"
            ),
        )

        extractor = MagicMock()
        audio_path = work_dir / "audio_full.wav"
        audio_path.touch()
        extractor.extract.return_value = audio_path

        isolator = MagicMock()
        vocals = work_dir / "vocals.wav"
        vocals.touch()
        bg = work_dir / "background.wav"
        bg.touch()
        isolator.isolate.return_value = VocalIsolationResult(
            vocals_path=vocals, background_path=bg
        )

        recognizer = MagicMock()
        recognizer.recognize.return_value = _make_transcription()

        translator = MagicMock()
        translator.translate.return_value = _make_translation()

        synthesizer = AsyncMock()
        vi_audio = work_dir / "vietnamese_audio.wav"
        vi_audio.touch()
        synthesizer.synthesize.return_value = SynthesisResult(
            audio_path=vi_audio, segment_audios=[]
        )

        composer = MagicMock()
        out_dir = work_dir / "output"
        out_dir.mkdir()
        out = out_dir / "output.mp4"
        out.touch()
        composer.compose.return_value = out

        pipeline = _make_pipeline(
            job_store=store,
            downloader=downloader,
            extractor=extractor,
            isolator=isolator,
            recognizer=recognizer,
            translator=translator,
            synthesizer=synthesizer,
            composer=composer,
        )

        result = await pipeline.resume("test-job-123", "downloading")

        assert isinstance(result, PipelineResult)
        # All steps should be called
        downloader.download.assert_called_once()
        extractor.extract.assert_called_once()
        isolator.isolate.assert_called_once()
        recognizer.recognize.assert_called_once()
        translator.translate.assert_called_once()
        synthesizer.synthesize.assert_called_once()
        composer.compose.assert_called_once()
