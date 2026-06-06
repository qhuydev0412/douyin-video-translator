"""Property-based tests for pipeline orchestration.

Feature: douyin-video-translator

Tests cover:
- Property 8: Pipeline Step Progression is Monotonic
- Property 9: Cancellation From Any Active Step
- Property 10: Pipeline Resume Preserves Prior Artifacts
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

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
from app.services.pipeline import (
    PipelineError,
    PipelineResult,
    STEP_ORDER,
    STEP_PROGRESS,
    TranslationPipeline,
)


# --- Helpers ---


class FakeJobStore:
    """In-memory job store for property testing."""

    def __init__(self, job_state: JobState):
        self._state = job_state
        self.updates: list[dict] = []

    def get_job(self, job_id: str) -> JobState:
        return self._state

    def update_job(self, job_id: str, **kwargs: object) -> None:
        self.updates.append({"job_id": job_id, **kwargs})
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)


def _make_job_state(
    job_id: str = "prop-test-job",
    url: str = "https://www.douyin.com/video/999",
    status: JobStatus = JobStatus.PROCESSING,
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


# Mapping of step index to the artifact keys that step produces
STEP_ARTIFACT_KEYS: dict[int, list[str]] = {
    0: ["video_path"],                          # DOWNLOADING
    1: ["audio_path"],                          # EXTRACTING_AUDIO
    2: ["vocals_path", "background_path"],      # ISOLATING_VOCALS
    3: ["transcription_path"],                  # RECOGNIZING_SPEECH
    4: ["translation_path"],                    # TRANSLATING
    5: ["vietnamese_audio"],                    # SYNTHESIZING_VOICE
    6: ["output_video"],                        # COMPOSING_VIDEO
}

# All artifact keys needed up to (but not including) a given step index
def _artifacts_before_step(step_index: int) -> dict[str, str]:
    """Return artifact keys for all steps before step_index with placeholder paths."""
    artifacts: dict[str, str] = {}
    for i in range(step_index):
        for key in STEP_ARTIFACT_KEYS[i]:
            artifacts[key] = f"/fake/path/{key}.file"
    return artifacts


def _create_artifact_files(work_dir: Path, artifacts: dict[str, str]) -> None:
    """Create placeholder files on disk for all artifact paths."""
    for key, path_str in artifacts.items():
        file_path = Path(path_str)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if key == "transcription_path":
            # Write a valid transcription JSON
            data = {
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": "ä½ å¥½", "speaker": "speaker_1"}
                ],
                "full_text": "ä½ å¥½",
                "language": "zh",
                "confidence": 0.95,
            }
            file_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        elif key == "translation_path":
            # Write a valid translation JSON
            data = {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "original_text": "ä½ å¥½",
                        "translated_text": "Xin chÃ o",
                        "speaker": "speaker_1",
                    }
                ],
                "full_text_original": "ä½ å¥½",
                "full_text_translated": "Xin chÃ o",
            }
            file_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        else:
            file_path.write_text("fake content", encoding="utf-8")


def _setup_service_mocks(work_dir: Path) -> dict[str, MagicMock | AsyncMock]:
    """Create service mocks that produce valid artifacts."""
    from app.models.job import VideoInfo

    video_path = work_dir / "original.mp4"
    audio_path = work_dir / "audio_full.wav"
    vocals_path = work_dir / "vocals.wav"
    background_path = work_dir / "background.wav"
    vietnamese_audio = work_dir / "vietnamese_audio.wav"
    output_path = work_dir / "output" / "output.mp4"

    downloader = MagicMock()
    downloader.download.return_value = DownloadResult(
        video_path=video_path,
        video_info=VideoInfo(
            duration_seconds=10.0, file_size_bytes=1000, resolution="720x1280"
        ),
    )

    extractor = MagicMock()
    extractor.extract.return_value = audio_path

    isolator = MagicMock()
    isolator.isolate.return_value = VocalIsolationResult(
        vocals_path=vocals_path, background_path=background_path
    )

    recognizer = MagicMock()
    recognizer.recognize.return_value = TranscriptionResult(
        segments=[
            TranscriptionSegment(start=0.0, end=2.0, text="ä½ å¥½", speaker="speaker_1")
        ],
        full_text="ä½ å¥½",
        language="zh",
        confidence=0.95,
    )

    translator = MagicMock()
    translator.translate.return_value = TranslationResult(
        segments=[
            TranslatedSegment(
                start=0.0,
                end=2.0,
                original_text="ä½ å¥½",
                translated_text="Xin chÃ o",
                speaker="speaker_1",
            )
        ],
        full_text_original="ä½ å¥½",
        full_text_translated="Xin chÃ o",
    )

    synthesizer = AsyncMock()
    synthesizer.synthesize.return_value = SynthesisResult(
        audio_path=vietnamese_audio, segment_audios=[]
    )

    composer = MagicMock()
    composer.compose.return_value = output_path

    return {
        "downloader": downloader,
        "extractor": extractor,
        "isolator": isolator,
        "recognizer": recognizer,
        "translator": translator,
        "synthesizer": synthesizer,
        "composer": composer,
    }


# Service call checkers - maps step index to the mock method that should be called
def _get_service_call_method(mocks: dict, step_index: int) -> MagicMock | AsyncMock:
    """Get the mock service method corresponding to a step index."""
    methods = [
        mocks["downloader"].download,
        mocks["extractor"].extract,
        mocks["isolator"].isolate,
        mocks["recognizer"].recognize,
        mocks["translator"].translate,
        mocks["synthesizer"].synthesize,
        mocks["composer"].compose,
    ]
    return methods[step_index]


# --- Property 8: Pipeline Step Progression is Monotonic ---


@pytest.mark.property
@pytest.mark.asyncio
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(num_successful_steps=st.integers(min_value=1, max_value=7))
async def test_progress_values_are_monotonically_non_decreasing(
    num_successful_steps: int, tmp_path: Path
):
    """Feature: douyin-video-translator, Property 8: Pipeline Step Progression is Monotonic

    For any pipeline execution that completes N steps, progress values recorded
    via update_job are monotonically non-decreasing.

    **Validates: Requirements 7.1**
    """
    work_dir = tmp_path / f"job-monotonic-{num_successful_steps}"
    work_dir.mkdir(parents=True, exist_ok=True)

    job_state = _make_job_state(
        work_dir=str(work_dir), status=JobStatus.QUEUED
    )
    store = FakeJobStore(job_state)

    mocks = _setup_service_mocks(work_dir)

    # Ensure all placeholder files exist
    for f in [
        work_dir / "original.mp4",
        work_dir / "audio_full.wav",
        work_dir / "vocals.wav",
        work_dir / "background.wav",
        work_dir / "vietnamese_audio.wav",
    ]:
        f.touch()
    (work_dir / "output").mkdir(exist_ok=True)
    (work_dir / "output" / "output.mp4").touch()

    # If num_successful_steps < 7, make the step after the last successful one fail
    if num_successful_steps < 7:
        failing_method = _get_service_call_method(mocks, num_successful_steps)
        failing_method.side_effect = RuntimeError("Simulated failure")

    pipeline = TranslationPipeline(
        downloader=mocks["downloader"],
        extractor=mocks["extractor"],
        isolator=mocks["isolator"],
        recognizer=mocks["recognizer"],
        translator=mocks["translator"],
        synthesizer=mocks["synthesizer"],
        composer=mocks["composer"],
        job_store=store,
    )

    try:
        await pipeline.execute("prop-test-job", "https://www.douyin.com/video/999")
    except PipelineError:
        pass  # Expected if num_successful_steps < 7

    # Extract all progress_percent values from update calls in order
    progress_values = [
        u["progress_percent"]
        for u in store.updates
        if "progress_percent" in u
    ]

    # Verify monotonicity: each value >= previous value
    for i in range(1, len(progress_values)):
        assert progress_values[i] >= progress_values[i - 1], (
            f"Progress decreased from {progress_values[i - 1]} to {progress_values[i]} "
            f"at index {i}. Full sequence: {progress_values}"
        )


@pytest.mark.property
@pytest.mark.asyncio
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(num_successful_steps=st.integers(min_value=1, max_value=7))
async def test_step_transitions_follow_step_order(
    num_successful_steps: int, tmp_path: Path
):
    """Feature: douyin-video-translator, Property 8: Pipeline Step Progression is Monotonic

    Step transitions recorded via update_job follow STEP_ORDER exactly:
    downloading â†’ extracting_audio â†’ isolating_vocals â†’ recognizing_speech â†’
    translating â†’ synthesizing_voice â†’ composing_video.

    **Validates: Requirements 7.1**
    """
    work_dir = tmp_path / f"job-step-order-{num_successful_steps}"
    work_dir.mkdir(parents=True, exist_ok=True)

    job_state = _make_job_state(
        work_dir=str(work_dir), status=JobStatus.QUEUED
    )
    store = FakeJobStore(job_state)

    mocks = _setup_service_mocks(work_dir)

    # Ensure all placeholder files exist
    for f in [
        work_dir / "original.mp4",
        work_dir / "audio_full.wav",
        work_dir / "vocals.wav",
        work_dir / "background.wav",
        work_dir / "vietnamese_audio.wav",
    ]:
        f.touch()
    (work_dir / "output").mkdir(exist_ok=True)
    (work_dir / "output" / "output.mp4").touch()

    # If num_successful_steps < 7, make the step after the last successful one fail
    if num_successful_steps < 7:
        failing_method = _get_service_call_method(mocks, num_successful_steps)
        failing_method.side_effect = RuntimeError("Simulated failure")

    pipeline = TranslationPipeline(
        downloader=mocks["downloader"],
        extractor=mocks["extractor"],
        isolator=mocks["isolator"],
        recognizer=mocks["recognizer"],
        translator=mocks["translator"],
        synthesizer=mocks["synthesizer"],
        composer=mocks["composer"],
        job_store=store,
    )

    try:
        await pipeline.execute("prop-test-job", "https://www.douyin.com/video/999")
    except PipelineError:
        pass  # Expected if num_successful_steps < 7

    # Extract current_step values from update calls
    step_transitions = [
        u["current_step"]
        for u in store.updates
        if "current_step" in u
    ]

    # Steps should follow STEP_ORDER exactly for the steps that were reached
    expected_steps = STEP_ORDER[:num_successful_steps + (1 if num_successful_steps < 7 else 0)]
    # The failing step is also entered (current_step is set before execution)
    # but only up to num_successful_steps steps complete. The (num_successful_steps+1)th
    # step is entered but fails. For full completion, all 7 steps are entered.
    if num_successful_steps < 7:
        # Steps 0..num_successful_steps are all entered (including the failing one)
        expected_steps = STEP_ORDER[: num_successful_steps + 1]
    else:
        expected_steps = STEP_ORDER[:]

    assert step_transitions == expected_steps, (
        f"Step transitions {[s.value for s in step_transitions]} don't match "
        f"expected {[s.value for s in expected_steps]}"
    )


@pytest.mark.property
@pytest.mark.asyncio
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(num_successful_steps=st.just(7))
async def test_final_progress_is_100_on_completion(
    num_successful_steps: int, tmp_path: Path
):
    """Feature: douyin-video-translator, Property 8: Pipeline Step Progression is Monotonic

    For any pipeline execution that completes all steps, the final progress
    value is 100.

    **Validates: Requirements 7.1**
    """
    work_dir = tmp_path / "job-final-progress"
    work_dir.mkdir(parents=True, exist_ok=True)

    job_state = _make_job_state(
        work_dir=str(work_dir), status=JobStatus.QUEUED
    )
    store = FakeJobStore(job_state)

    mocks = _setup_service_mocks(work_dir)

    # Ensure all placeholder files exist
    for f in [
        work_dir / "original.mp4",
        work_dir / "audio_full.wav",
        work_dir / "vocals.wav",
        work_dir / "background.wav",
        work_dir / "vietnamese_audio.wav",
    ]:
        f.touch()
    (work_dir / "output").mkdir(exist_ok=True)
    (work_dir / "output" / "output.mp4").touch()

    pipeline = TranslationPipeline(
        downloader=mocks["downloader"],
        extractor=mocks["extractor"],
        isolator=mocks["isolator"],
        recognizer=mocks["recognizer"],
        translator=mocks["translator"],
        synthesizer=mocks["synthesizer"],
        composer=mocks["composer"],
        job_store=store,
    )

    await pipeline.execute("prop-test-job", "https://www.douyin.com/video/999")

    # Extract all progress_percent values
    progress_values = [
        u["progress_percent"]
        for u in store.updates
        if "progress_percent" in u
    ]

    # Final progress value should be 100
    assert progress_values[-1] == 100, (
        f"Final progress should be 100, got {progress_values[-1]}. "
        f"Full sequence: {progress_values}"
    )


# --- Property 10: Pipeline Resume Preserves Prior Artifacts ---


@pytest.mark.property
@pytest.mark.asyncio
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(resume_from_step=st.integers(min_value=1, max_value=6))
async def test_resume_does_not_call_services_for_prior_steps(
    resume_from_step: int, tmp_path: Path
):
    """Feature: douyin-video-translator, Property 10: Pipeline Resume Preserves Prior Artifacts

    For any step index K (1..6), resuming from step K does NOT call services
    for steps 0..K-1.

    **Validates: Requirements 6.5, 7.4**
    """
    work_dir = tmp_path / f"job-resume-{resume_from_step}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Pre-populate artifacts for steps 0..resume_from_step-1
    prior_artifacts = _artifacts_before_step(resume_from_step)

    # Update paths to use real work_dir
    real_artifacts: dict[str, str] = {}
    for key, _ in prior_artifacts.items():
        real_artifacts[key] = str(work_dir / f"{key}.file")

    # Create actual files on disk for the artifacts
    _create_artifact_files(work_dir, real_artifacts)

    job_state = _make_job_state(work_dir=str(work_dir), artifacts=real_artifacts)
    store = FakeJobStore(job_state)

    mocks = _setup_service_mocks(work_dir)

    # Ensure files exist for steps that will execute
    for f in [
        work_dir / "original.mp4",
        work_dir / "audio_full.wav",
        work_dir / "vocals.wav",
        work_dir / "background.wav",
        work_dir / "vietnamese_audio.wav",
    ]:
        f.touch()
    (work_dir / "output").mkdir(exist_ok=True)
    (work_dir / "output" / "output.mp4").touch()

    pipeline = TranslationPipeline(
        downloader=mocks["downloader"],
        extractor=mocks["extractor"],
        isolator=mocks["isolator"],
        recognizer=mocks["recognizer"],
        translator=mocks["translator"],
        synthesizer=mocks["synthesizer"],
        composer=mocks["composer"],
        job_store=store,
    )

    target_step = STEP_ORDER[resume_from_step]
    await pipeline.resume("prop-test-job", target_step.value)

    # Verify steps BEFORE resume_from_step were NOT called
    for i in range(resume_from_step):
        method = _get_service_call_method(mocks, i)
        assert not method.called, (
            f"Step {i} ({STEP_ORDER[i].value}) should NOT have been called "
            f"when resuming from step {resume_from_step} ({target_step.value})"
        )


@pytest.mark.property
@pytest.mark.asyncio
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(resume_from_step=st.integers(min_value=1, max_value=6))
async def test_resume_calls_services_from_resume_step_onwards(
    resume_from_step: int, tmp_path: Path
):
    """Feature: douyin-video-translator, Property 10: Pipeline Resume Preserves Prior Artifacts

    When resuming from step K, services for steps K..6 ARE called (if they succeed).

    **Validates: Requirements 6.5, 7.4**
    """
    work_dir = tmp_path / f"job-resume-fwd-{resume_from_step}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Pre-populate artifacts for steps 0..resume_from_step-1
    prior_artifacts = _artifacts_before_step(resume_from_step)

    real_artifacts: dict[str, str] = {}
    for key, _ in prior_artifacts.items():
        real_artifacts[key] = str(work_dir / f"{key}.file")

    _create_artifact_files(work_dir, real_artifacts)

    job_state = _make_job_state(work_dir=str(work_dir), artifacts=real_artifacts)
    store = FakeJobStore(job_state)

    mocks = _setup_service_mocks(work_dir)

    # Ensure files exist for the pipeline steps
    for f in [
        work_dir / "original.mp4",
        work_dir / "audio_full.wav",
        work_dir / "vocals.wav",
        work_dir / "background.wav",
        work_dir / "vietnamese_audio.wav",
    ]:
        f.touch()
    (work_dir / "output").mkdir(exist_ok=True)
    (work_dir / "output" / "output.mp4").touch()

    pipeline = TranslationPipeline(
        downloader=mocks["downloader"],
        extractor=mocks["extractor"],
        isolator=mocks["isolator"],
        recognizer=mocks["recognizer"],
        translator=mocks["translator"],
        synthesizer=mocks["synthesizer"],
        composer=mocks["composer"],
        job_store=store,
    )

    target_step = STEP_ORDER[resume_from_step]
    await pipeline.resume("prop-test-job", target_step.value)

    # Verify steps FROM resume_from_step onwards WERE called
    for i in range(resume_from_step, len(STEP_ORDER)):
        method = _get_service_call_method(mocks, i)
        assert method.called, (
            f"Step {i} ({STEP_ORDER[i].value}) should have been called "
            f"when resuming from step {resume_from_step} ({target_step.value})"
        )


@pytest.mark.property
@pytest.mark.asyncio
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(resume_from_step=st.integers(min_value=1, max_value=6))
async def test_resume_preserves_prior_artifacts_on_disk(
    resume_from_step: int, tmp_path: Path
):
    """Feature: douyin-video-translator, Property 10: Pipeline Resume Preserves Prior Artifacts

    Pre-existing artifacts in job state are preserved and accessible during
    resumed execution.

    **Validates: Requirements 6.5, 7.4**
    """
    work_dir = tmp_path / f"job-resume-preserve-{resume_from_step}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Pre-populate artifacts for steps 0..resume_from_step-1
    prior_artifacts = _artifacts_before_step(resume_from_step)

    real_artifacts: dict[str, str] = {}
    for key, _ in prior_artifacts.items():
        real_artifacts[key] = str(work_dir / f"{key}.file")

    _create_artifact_files(work_dir, real_artifacts)

    job_state = _make_job_state(work_dir=str(work_dir), artifacts=real_artifacts)
    store = FakeJobStore(job_state)

    mocks = _setup_service_mocks(work_dir)

    # Ensure files for all steps
    for f in [
        work_dir / "original.mp4",
        work_dir / "audio_full.wav",
        work_dir / "vocals.wav",
        work_dir / "background.wav",
        work_dir / "vietnamese_audio.wav",
    ]:
        f.touch()
    (work_dir / "output").mkdir(exist_ok=True)
    (work_dir / "output" / "output.mp4").touch()

    pipeline = TranslationPipeline(
        downloader=mocks["downloader"],
        extractor=mocks["extractor"],
        isolator=mocks["isolator"],
        recognizer=mocks["recognizer"],
        translator=mocks["translator"],
        synthesizer=mocks["synthesizer"],
        composer=mocks["composer"],
        job_store=store,
    )

    target_step = STEP_ORDER[resume_from_step]
    result = await pipeline.resume("prop-test-job", target_step.value)

    # Verify ALL prior artifact files still exist on disk
    for key, path_str in real_artifacts.items():
        assert Path(path_str).exists(), (
            f"Artifact '{key}' at '{path_str}' should still exist after resume "
            f"from step {resume_from_step} ({target_step.value})"
        )

    # Verify prior artifacts are still in the result
    for key in real_artifacts:
        assert key in result.artifacts, (
            f"Prior artifact '{key}' should be present in pipeline result "
            f"after resuming from step {resume_from_step}"
        )


@pytest.mark.property
@pytest.mark.asyncio
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(fail_at_step=st.integers(min_value=1, max_value=6))
async def test_failure_at_step_n_identifies_correct_step_in_error(
    fail_at_step: int, tmp_path: Path
):
    """Feature: douyin-video-translator, Property 10: Pipeline Resume Preserves Prior Artifacts

    On failure at step N, PipelineError.step correctly identifies step N.

    **Validates: Requirements 6.5, 7.4**
    """
    work_dir = tmp_path / f"job-fail-{fail_at_step}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Pre-populate artifacts for steps before the failing step
    prior_artifacts = _artifacts_before_step(fail_at_step)

    real_artifacts: dict[str, str] = {}
    for key, _ in prior_artifacts.items():
        real_artifacts[key] = str(work_dir / f"{key}.file")

    _create_artifact_files(work_dir, real_artifacts)

    job_state = _make_job_state(work_dir=str(work_dir), artifacts=real_artifacts)
    store = FakeJobStore(job_state)

    mocks = _setup_service_mocks(work_dir)

    # Ensure files for the pipeline
    for f in [
        work_dir / "original.mp4",
        work_dir / "audio_full.wav",
        work_dir / "vocals.wav",
        work_dir / "background.wav",
        work_dir / "vietnamese_audio.wav",
    ]:
        f.touch()
    (work_dir / "output").mkdir(exist_ok=True)
    (work_dir / "output" / "output.mp4").touch()

    # Make the specific step fail
    error_message = f"Simulated failure at step {fail_at_step}"
    failing_method = _get_service_call_method(mocks, fail_at_step)
    failing_method.side_effect = RuntimeError(error_message)

    pipeline = TranslationPipeline(
        downloader=mocks["downloader"],
        extractor=mocks["extractor"],
        isolator=mocks["isolator"],
        recognizer=mocks["recognizer"],
        translator=mocks["translator"],
        synthesizer=mocks["synthesizer"],
        composer=mocks["composer"],
        job_store=store,
    )

    # Resume from the failing step
    target_step = STEP_ORDER[fail_at_step]
    with pytest.raises(PipelineError) as exc_info:
        await pipeline.resume("prop-test-job", target_step.value)

    # Verify the error correctly identifies the failed step
    assert exc_info.value.step == target_step, (
        f"PipelineError.step should be {target_step.value} but got "
        f"{exc_info.value.step.value}"
    )
    assert error_message in exc_info.value.message


# --- Property 9: Cancellation From Any Active Step ---


class CancellableFakeJobStore:
    """Job store that simulates cancellation at a specific step.

    Changes the job status to CANCELLED when `_check_cancellation` is called
    for the target step. The pipeline calls `get_job` for cancellation checks
    at specific points, so we count calls and trigger at the right one.
    """

    def __init__(self, job_state: JobState, cancel_at_step: int):
        self._state = job_state
        self._cancel_at_step = cancel_at_step
        self._cancellation_check_count = 0
        self.updates: list[dict] = []

    def get_job(self, job_id: str) -> JobState:
        """Return job state, marking CANCELLED when the target step's check fires.

        The pipeline calls _check_cancellation(job_id, step) which calls get_job.
        The first cancellation check is for step 0 (DOWNLOADING) before the loop,
        then for steps 1..6 inside the loop iteration.
        So cancellation_check i corresponds to step i.
        """
        # The pipeline checks cancellation:
        # - Once before the loop starts (for the from_step, i.e., step 0)
        # - Then before each subsequent step (for steps 1..6)
        # Additionally, execute() calls get_job once at the beginning to read job state.
        # We need to distinguish the initial get_job from cancellation checks.
        #
        # Flow of get_job calls in execute():
        # 1. execute() -> self._job_store.get_job(job_id) to read work_dir etc.
        # 2. _run_steps -> _check_cancellation(job_id, STEP_ORDER[0]) -> get_job
        # 3. After step 0 executes -> _check_cancellation(job_id, STEP_ORDER[1]) -> get_job
        # ...
        # So get_job call 1 is the initial read, calls 2+ are cancellation checks.
        # Cancellation check N corresponds to step N-1 (0-indexed after subtracting 1).
        self._cancellation_check_count += 1

        # Call 1 is the initial state read in execute(). 
        # Calls 2+ are cancellation checks. Call 2 = step 0, call 3 = step 1, etc.
        if self._cancellation_check_count >= 2:
            check_index = self._cancellation_check_count - 2  # 0-based step index
            if check_index >= self._cancel_at_step:
                self._state.status = JobStatus.CANCELLED

        return self._state

    def update_job(self, job_id: str, **kwargs: object) -> None:
        self.updates.append({"job_id": job_id, **kwargs})
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)


@pytest.mark.property
@pytest.mark.asyncio
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(cancel_at_step=st.integers(min_value=0, max_value=6))
async def test_cancellation_raises_at_correct_step(
    cancel_at_step: int, tmp_path: Path
):
    """Feature: douyin-video-translator, Property 9: Cancellation From Any Active Step

    For any step index K (0..6), if job is cancelled before step K,
    pipeline raises CancellationError at step K.

    **Validates: Requirements 7.3**
    """
    work_dir = tmp_path / f"job-cancel-{cancel_at_step}"
    work_dir.mkdir(parents=True, exist_ok=True)

    job_state = _make_job_state(
        status=JobStatus.QUEUED,
        work_dir=str(work_dir),
    )
    store = CancellableFakeJobStore(job_state, cancel_at_step=cancel_at_step)

    mocks = _setup_service_mocks(work_dir)

    # Ensure files exist for potential step execution
    for f in [
        work_dir / "original.mp4",
        work_dir / "audio_full.wav",
        work_dir / "vocals.wav",
        work_dir / "background.wav",
        work_dir / "vietnamese_audio.wav",
    ]:
        f.touch()
    (work_dir / "output").mkdir(exist_ok=True)
    (work_dir / "output" / "output.mp4").touch()

    pipeline = TranslationPipeline(
        downloader=mocks["downloader"],
        extractor=mocks["extractor"],
        isolator=mocks["isolator"],
        recognizer=mocks["recognizer"],
        translator=mocks["translator"],
        synthesizer=mocks["synthesizer"],
        composer=mocks["composer"],
        job_store=store,
    )

    from app.services.pipeline import CancellationError

    with pytest.raises(CancellationError) as exc_info:
        await pipeline.execute("prop-test-job", "https://www.douyin.com/video/999")

    # CancellationError.step should be the step where cancellation was detected
    expected_step = STEP_ORDER[cancel_at_step]
    assert exc_info.value.step == expected_step, (
        f"Expected CancellationError at step {expected_step.value} "
        f"(index {cancel_at_step}), but got {exc_info.value.step.value}"
    )


@pytest.mark.property
@pytest.mark.asyncio
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(cancel_at_step=st.integers(min_value=0, max_value=6))
async def test_cancellation_prevents_subsequent_steps(
    cancel_at_step: int, tmp_path: Path
):
    """Feature: douyin-video-translator, Property 9: Cancellation From Any Active Step

    No services after the cancelled step are called.
    Services at and after the cancel step index should NOT be invoked.

    **Validates: Requirements 7.3**
    """
    work_dir = tmp_path / f"job-cancel-no-call-{cancel_at_step}"
    work_dir.mkdir(parents=True, exist_ok=True)

    job_state = _make_job_state(
        status=JobStatus.QUEUED,
        work_dir=str(work_dir),
    )
    store = CancellableFakeJobStore(job_state, cancel_at_step=cancel_at_step)

    mocks = _setup_service_mocks(work_dir)

    # Ensure files exist for potential step execution
    for f in [
        work_dir / "original.mp4",
        work_dir / "audio_full.wav",
        work_dir / "vocals.wav",
        work_dir / "background.wav",
        work_dir / "vietnamese_audio.wav",
    ]:
        f.touch()
    (work_dir / "output").mkdir(exist_ok=True)
    (work_dir / "output" / "output.mp4").touch()

    pipeline = TranslationPipeline(
        downloader=mocks["downloader"],
        extractor=mocks["extractor"],
        isolator=mocks["isolator"],
        recognizer=mocks["recognizer"],
        translator=mocks["translator"],
        synthesizer=mocks["synthesizer"],
        composer=mocks["composer"],
        job_store=store,
    )

    from app.services.pipeline import CancellationError

    with pytest.raises(CancellationError):
        await pipeline.execute("prop-test-job", "https://www.douyin.com/video/999")

    # Verify that services at cancel_at_step and after are NOT called
    for i in range(cancel_at_step, len(STEP_ORDER)):
        method = _get_service_call_method(mocks, i)
        assert not method.called, (
            f"Step {i} ({STEP_ORDER[i].value}) should NOT have been called "
            f"when job was cancelled at step {cancel_at_step} ({STEP_ORDER[cancel_at_step].value})"
        )
