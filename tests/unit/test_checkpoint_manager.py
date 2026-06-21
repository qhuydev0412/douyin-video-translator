"""Unit tests for the CheckpointManager service."""

from datetime import datetime, timezone

import pytest

from app.models.job import CheckpointType, JobState, JobStatus, PipelineStep
from app.services.checkpoint_manager import (
    CHECKPOINT_NEXT_STEP,
    CheckpointError,
    CheckpointManager,
    ConfirmationInProgressError,
    NotAwaitingConfirmationError,
    WrongCheckpointError,
)


# --- Test Helpers ---


class FakeJobStore:
    """In-memory job store for testing CheckpointManager."""

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
    job_id: str = "job-001",
    status: JobStatus = JobStatus.PROCESSING,
    checkpoint_type: CheckpointType | None = None,
    checkpoint_entered_at: datetime | None = None,
    confirmation_lock: bool = False,
) -> JobState:
    now = datetime.now(timezone.utc)
    return JobState(
        job_id=job_id,
        url="https://www.douyin.com/video/123",
        status=status,
        created_at=now,
        updated_at=now,
        work_dir="/tmp/jobs/job-001",
        checkpoint_type=checkpoint_type,
        checkpoint_entered_at=checkpoint_entered_at,
        confirmation_lock=confirmation_lock,
    )


# --- Tests for pause_at_checkpoint ---


class TestPauseAtCheckpoint:
    """Tests for CheckpointManager.pause_at_checkpoint."""

    def test_pause_sets_awaiting_confirmation_status(self):
        """pause_at_checkpoint transitions status to AWAITING_CONFIRMATION."""
        job = _make_job_state(status=JobStatus.PROCESSING)
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        manager.pause_at_checkpoint("job-001", CheckpointType.TRANSCRIPTION)

        assert job.status == JobStatus.AWAITING_CONFIRMATION

    def test_pause_stores_checkpoint_type(self):
        """pause_at_checkpoint stores the correct checkpoint_type."""
        job = _make_job_state(status=JobStatus.PROCESSING)
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        manager.pause_at_checkpoint("job-001", CheckpointType.TRANSLATION)

        assert job.checkpoint_type == CheckpointType.TRANSLATION

    def test_pause_stores_checkpoint_entered_at(self):
        """pause_at_checkpoint records the time the checkpoint was entered."""
        job = _make_job_state(status=JobStatus.PROCESSING)
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        manager.pause_at_checkpoint("job-001", CheckpointType.VOICE_SELECTION)

        assert job.checkpoint_entered_at is not None
        # Should be a recent UTC datetime
        assert (datetime.now(timezone.utc) - job.checkpoint_entered_at).total_seconds() < 2

    def test_pause_calls_update_job_with_correct_fields(self):
        """pause_at_checkpoint passes all required fields to update_job."""
        job = _make_job_state(status=JobStatus.PROCESSING)
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        manager.pause_at_checkpoint("job-001", CheckpointType.TRANSCRIPTION)

        assert len(store.updates) == 1
        update = store.updates[0]
        assert update["job_id"] == "job-001"
        assert update["status"] == JobStatus.AWAITING_CONFIRMATION
        assert update["checkpoint_type"] == CheckpointType.TRANSCRIPTION
        assert "checkpoint_entered_at" in update


# --- Tests for validate_confirmation ---


class TestValidateConfirmation:
    """Tests for CheckpointManager.validate_confirmation."""

    def test_validate_success_acquires_lock(self):
        """validate_confirmation acquires confirmation_lock on valid state."""
        job = _make_job_state(
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSCRIPTION,
            checkpoint_entered_at=datetime.now(timezone.utc),
        )
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        result = manager.validate_confirmation("job-001", CheckpointType.TRANSCRIPTION)

        assert result.confirmation_lock is True

    def test_validate_returns_job_state(self):
        """validate_confirmation returns the updated JobState."""
        job = _make_job_state(
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSLATION,
            checkpoint_entered_at=datetime.now(timezone.utc),
        )
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        result = manager.validate_confirmation("job-001", CheckpointType.TRANSLATION)

        assert isinstance(result, JobState)
        assert result.job_id == "job-001"

    def test_validate_raises_not_awaiting_when_processing(self):
        """validate_confirmation raises NotAwaitingConfirmationError for wrong status."""
        job = _make_job_state(status=JobStatus.PROCESSING)
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        with pytest.raises(NotAwaitingConfirmationError) as exc_info:
            manager.validate_confirmation("job-001", CheckpointType.TRANSCRIPTION)

        assert exc_info.value.current_status == JobStatus.PROCESSING
        assert exc_info.value.job_id == "job-001"

    def test_validate_raises_not_awaiting_when_completed(self):
        """validate_confirmation raises for COMPLETED status."""
        job = _make_job_state(status=JobStatus.COMPLETED)
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        with pytest.raises(NotAwaitingConfirmationError) as exc_info:
            manager.validate_confirmation("job-001", CheckpointType.TRANSCRIPTION)

        assert exc_info.value.current_status == JobStatus.COMPLETED

    def test_validate_raises_wrong_checkpoint(self):
        """validate_confirmation raises WrongCheckpointError for mismatched checkpoint."""
        job = _make_job_state(
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSCRIPTION,
            checkpoint_entered_at=datetime.now(timezone.utc),
        )
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        with pytest.raises(WrongCheckpointError) as exc_info:
            manager.validate_confirmation("job-001", CheckpointType.TRANSLATION)

        assert exc_info.value.expected == CheckpointType.TRANSLATION
        assert exc_info.value.actual == CheckpointType.TRANSCRIPTION

    def test_validate_raises_confirmation_in_progress(self):
        """validate_confirmation raises ConfirmationInProgressError when lock is held."""
        job = _make_job_state(
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSCRIPTION,
            checkpoint_entered_at=datetime.now(timezone.utc),
            confirmation_lock=True,
        )
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        with pytest.raises(ConfirmationInProgressError) as exc_info:
            manager.validate_confirmation("job-001", CheckpointType.TRANSCRIPTION)

        assert exc_info.value.job_id == "job-001"

    def test_validate_error_hierarchy(self):
        """All custom exceptions inherit from CheckpointError."""
        assert issubclass(NotAwaitingConfirmationError, CheckpointError)
        assert issubclass(WrongCheckpointError, CheckpointError)
        assert issubclass(ConfirmationInProgressError, CheckpointError)


# --- Tests for confirm_and_resume ---


class TestConfirmAndResume:
    """Tests for CheckpointManager.confirm_and_resume."""

    def test_confirm_transcription_returns_translating(self):
        """confirm_and_resume after TRANSCRIPTION checkpoint returns TRANSLATING."""
        job = _make_job_state(
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSCRIPTION,
            checkpoint_entered_at=datetime.now(timezone.utc),
            confirmation_lock=True,
        )
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        next_step = manager.confirm_and_resume("job-001")

        assert next_step == PipelineStep.TRANSLATING

    def test_confirm_translation_returns_synthesizing_voice(self):
        """confirm_and_resume after TRANSLATION checkpoint returns SYNTHESIZING_VOICE."""
        job = _make_job_state(
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSLATION,
            checkpoint_entered_at=datetime.now(timezone.utc),
            confirmation_lock=True,
        )
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        next_step = manager.confirm_and_resume("job-001")

        assert next_step == PipelineStep.SYNTHESIZING_VOICE

    def test_confirm_voice_selection_returns_synthesizing_voice(self):
        """confirm_and_resume after VOICE_SELECTION checkpoint returns SYNTHESIZING_VOICE.

        The pipeline must re-run SYNTHESIZING_VOICE to perform the actual TTS
        using the selected voice (preview generation only happened on the first run).
        """
        job = _make_job_state(
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.VOICE_SELECTION,
            checkpoint_entered_at=datetime.now(timezone.utc),
            confirmation_lock=True,
        )
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        next_step = manager.confirm_and_resume("job-001")

        assert next_step == PipelineStep.SYNTHESIZING_VOICE

    def test_confirm_sets_status_to_processing(self):
        """confirm_and_resume transitions job status to PROCESSING."""
        job = _make_job_state(
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSCRIPTION,
            checkpoint_entered_at=datetime.now(timezone.utc),
            confirmation_lock=True,
        )
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        manager.confirm_and_resume("job-001")

        assert job.status == JobStatus.PROCESSING

    def test_confirm_releases_lock(self):
        """confirm_and_resume releases the confirmation lock."""
        job = _make_job_state(
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSLATION,
            checkpoint_entered_at=datetime.now(timezone.utc),
            confirmation_lock=True,
        )
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        manager.confirm_and_resume("job-001")

        assert job.confirmation_lock is False

    def test_confirm_clears_checkpoint_fields(self):
        """confirm_and_resume clears checkpoint_type and checkpoint_entered_at."""
        job = _make_job_state(
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.VOICE_SELECTION,
            checkpoint_entered_at=datetime.now(timezone.utc),
            confirmation_lock=True,
        )
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        manager.confirm_and_resume("job-001")

        assert job.checkpoint_type is None
        assert job.checkpoint_entered_at is None


# --- Tests for CHECKPOINT_NEXT_STEP mapping ---


class TestCheckpointNextStepMapping:
    """Verify the CHECKPOINT_NEXT_STEP mapping covers all checkpoint types."""

    def test_all_checkpoint_types_have_next_step(self):
        """Every CheckpointType has a corresponding next step."""
        for ct in CheckpointType:
            assert ct in CHECKPOINT_NEXT_STEP

    def test_mapping_values_are_valid_pipeline_steps(self):
        """All mapped next steps are valid PipelineStep values."""
        for next_step in CHECKPOINT_NEXT_STEP.values():
            assert isinstance(next_step, PipelineStep)


# --- Tests for apply_transcription_edits ---

import json
import tempfile
from pathlib import Path

from app.models.confirmation_schemas import SegmentEdit, TranslationEdit
from app.services.checkpoint_manager import InvalidSegmentIndexError


def _make_job_state_with_artifacts(
    job_id: str = "job-001",
    transcription_path: str | None = None,
    translation_path: str | None = None,
) -> JobState:
    """Create a JobState with artifact paths set."""
    now = datetime.now(timezone.utc)
    artifacts: dict[str, str] = {}
    if transcription_path:
        artifacts["transcription_path"] = transcription_path
    if translation_path:
        artifacts["translation_path"] = translation_path
    return JobState(
        job_id=job_id,
        url="https://www.douyin.com/video/123",
        status=JobStatus.AWAITING_CONFIRMATION,
        checkpoint_type=CheckpointType.TRANSCRIPTION,
        checkpoint_entered_at=now,
        created_at=now,
        updated_at=now,
        work_dir="/tmp/jobs/job-001",
        artifacts=artifacts,
    )


class TestApplyTranscriptionEdits:
    """Tests for CheckpointManager.apply_transcription_edits."""

    def test_applies_single_edit(self, tmp_path):
        """apply_transcription_edits replaces text at specified index."""
        transcription = {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "你好", "speaker": None},
                {"start": 2.5, "end": 5.0, "text": "世界", "speaker": None},
            ],
            "full_text": "你好 世界",
            "language": "zh",
            "confidence": 0.95,
        }
        file_path = tmp_path / "transcription.json"
        file_path.write_text(json.dumps(transcription), encoding="utf-8")

        job = _make_job_state_with_artifacts(transcription_path=str(file_path))
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        edits = [SegmentEdit(index=0, text="你们好")]
        manager.apply_transcription_edits("job-001", edits)

        result = json.loads(file_path.read_text(encoding="utf-8"))
        assert result["segments"][0]["text"] == "你们好"
        assert result["segments"][1]["text"] == "世界"

    def test_applies_multiple_edits(self, tmp_path):
        """apply_transcription_edits applies all edits in the list."""
        transcription = {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "你好", "speaker": None},
                {"start": 2.5, "end": 5.0, "text": "世界", "speaker": None},
                {"start": 5.0, "end": 7.5, "text": "再见", "speaker": None},
            ],
            "full_text": "你好 世界 再见",
            "language": "zh",
            "confidence": 0.95,
        }
        file_path = tmp_path / "transcription.json"
        file_path.write_text(json.dumps(transcription), encoding="utf-8")

        job = _make_job_state_with_artifacts(transcription_path=str(file_path))
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        edits = [
            SegmentEdit(index=0, text="大家好"),
            SegmentEdit(index=2, text="拜拜"),
        ]
        manager.apply_transcription_edits("job-001", edits)

        result = json.loads(file_path.read_text(encoding="utf-8"))
        assert result["segments"][0]["text"] == "大家好"
        assert result["segments"][1]["text"] == "世界"
        assert result["segments"][2]["text"] == "拜拜"

    def test_excludes_whitespace_only_segments(self, tmp_path):
        """apply_transcription_edits removes segments with whitespace-only text."""
        transcription = {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "你好", "speaker": None},
                {"start": 2.5, "end": 5.0, "text": "世界", "speaker": None},
            ],
            "full_text": "你好 世界",
            "language": "zh",
            "confidence": 0.95,
        }
        file_path = tmp_path / "transcription.json"
        file_path.write_text(json.dumps(transcription), encoding="utf-8")

        job = _make_job_state_with_artifacts(transcription_path=str(file_path))
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        edits = [SegmentEdit(index=0, text="   ")]
        manager.apply_transcription_edits("job-001", edits)

        result = json.loads(file_path.read_text(encoding="utf-8"))
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "世界"

    def test_raises_invalid_segment_index_for_out_of_bounds(self, tmp_path):
        """apply_transcription_edits raises InvalidSegmentIndexError for bad index."""
        transcription = {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "你好", "speaker": None},
            ],
            "full_text": "你好",
            "language": "zh",
            "confidence": 0.95,
        }
        file_path = tmp_path / "transcription.json"
        file_path.write_text(json.dumps(transcription), encoding="utf-8")

        job = _make_job_state_with_artifacts(transcription_path=str(file_path))
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        edits = [SegmentEdit(index=5, text="test")]
        with pytest.raises(InvalidSegmentIndexError) as exc_info:
            manager.apply_transcription_edits("job-001", edits)

        assert exc_info.value.index == 5
        assert exc_info.value.segment_count == 1
        assert exc_info.value.job_id == "job-001"

    def test_no_edits_applied_on_invalid_index(self, tmp_path):
        """If any index is invalid, no edits are applied (validation-first)."""
        transcription = {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "你好", "speaker": None},
                {"start": 2.5, "end": 5.0, "text": "世界", "speaker": None},
            ],
            "full_text": "你好 世界",
            "language": "zh",
            "confidence": 0.95,
        }
        file_path = tmp_path / "transcription.json"
        file_path.write_text(json.dumps(transcription), encoding="utf-8")

        job = _make_job_state_with_artifacts(transcription_path=str(file_path))
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        # First edit is valid, second is out of bounds
        edits = [
            SegmentEdit(index=0, text="changed"),
            SegmentEdit(index=10, text="bad"),
        ]
        with pytest.raises(InvalidSegmentIndexError):
            manager.apply_transcription_edits("job-001", edits)

        # Verify original data is unchanged
        result = json.loads(file_path.read_text(encoding="utf-8"))
        assert result["segments"][0]["text"] == "你好"

    def test_preserves_unedited_segments(self, tmp_path):
        """Segments not in the edit list remain unchanged."""
        transcription = {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "你好", "speaker": None},
                {"start": 2.5, "end": 5.0, "text": "世界", "speaker": None},
                {"start": 5.0, "end": 7.5, "text": "再见", "speaker": None},
            ],
            "full_text": "你好 世界 再见",
            "language": "zh",
            "confidence": 0.95,
        }
        file_path = tmp_path / "transcription.json"
        file_path.write_text(json.dumps(transcription), encoding="utf-8")

        job = _make_job_state_with_artifacts(transcription_path=str(file_path))
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        # Only edit index 1
        edits = [SegmentEdit(index=1, text="地球")]
        manager.apply_transcription_edits("job-001", edits)

        result = json.loads(file_path.read_text(encoding="utf-8"))
        assert result["segments"][0]["text"] == "你好"
        assert result["segments"][1]["text"] == "地球"
        assert result["segments"][2]["text"] == "再见"


# --- Tests for apply_translation_edits ---


class TestApplyTranslationEdits:
    """Tests for CheckpointManager.apply_translation_edits."""

    def test_applies_single_edit(self, tmp_path):
        """apply_translation_edits replaces translated_text at specified index."""
        translation = {
            "segments": [
                {"start": 0.0, "end": 2.5, "original_text": "你好", "translated_text": "Xin chào", "speaker": None},
                {"start": 2.5, "end": 5.0, "original_text": "世界", "translated_text": "Thế giới", "speaker": None},
            ],
            "full_text_original": "你好 世界",
            "full_text_translated": "Xin chào Thế giới",
        }
        file_path = tmp_path / "translation.json"
        file_path.write_text(json.dumps(translation), encoding="utf-8")

        job = _make_job_state_with_artifacts(translation_path=str(file_path))
        job.checkpoint_type = CheckpointType.TRANSLATION
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        edits = [TranslationEdit(index=0, translated_text="Chào bạn")]
        manager.apply_translation_edits("job-001", edits)

        result = json.loads(file_path.read_text(encoding="utf-8"))
        assert result["segments"][0]["translated_text"] == "Chào bạn"
        assert result["segments"][1]["translated_text"] == "Thế giới"

    def test_applies_multiple_edits(self, tmp_path):
        """apply_translation_edits applies all edits in the list."""
        translation = {
            "segments": [
                {"start": 0.0, "end": 2.5, "original_text": "你好", "translated_text": "Xin chào", "speaker": None},
                {"start": 2.5, "end": 5.0, "original_text": "世界", "translated_text": "Thế giới", "speaker": None},
                {"start": 5.0, "end": 7.5, "original_text": "再见", "translated_text": "Tạm biệt", "speaker": None},
            ],
            "full_text_original": "你好 世界 再见",
            "full_text_translated": "Xin chào Thế giới Tạm biệt",
        }
        file_path = tmp_path / "translation.json"
        file_path.write_text(json.dumps(translation), encoding="utf-8")

        job = _make_job_state_with_artifacts(translation_path=str(file_path))
        job.checkpoint_type = CheckpointType.TRANSLATION
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        edits = [
            TranslationEdit(index=0, translated_text="Chào các bạn"),
            TranslationEdit(index=2, translated_text="Hẹn gặp lại"),
        ]
        manager.apply_translation_edits("job-001", edits)

        result = json.loads(file_path.read_text(encoding="utf-8"))
        assert result["segments"][0]["translated_text"] == "Chào các bạn"
        assert result["segments"][1]["translated_text"] == "Thế giới"
        assert result["segments"][2]["translated_text"] == "Hẹn gặp lại"

    def test_excludes_whitespace_only_segments(self, tmp_path):
        """apply_translation_edits removes segments with whitespace-only translated_text."""
        translation = {
            "segments": [
                {"start": 0.0, "end": 2.5, "original_text": "你好", "translated_text": "Xin chào", "speaker": None},
                {"start": 2.5, "end": 5.0, "original_text": "世界", "translated_text": "Thế giới", "speaker": None},
            ],
            "full_text_original": "你好 世界",
            "full_text_translated": "Xin chào Thế giới",
        }
        file_path = tmp_path / "translation.json"
        file_path.write_text(json.dumps(translation), encoding="utf-8")

        job = _make_job_state_with_artifacts(translation_path=str(file_path))
        job.checkpoint_type = CheckpointType.TRANSLATION
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        edits = [TranslationEdit(index=1, translated_text="  \t\n  ")]
        manager.apply_translation_edits("job-001", edits)

        result = json.loads(file_path.read_text(encoding="utf-8"))
        assert len(result["segments"]) == 1
        assert result["segments"][0]["translated_text"] == "Xin chào"

    def test_raises_invalid_segment_index_for_out_of_bounds(self, tmp_path):
        """apply_translation_edits raises InvalidSegmentIndexError for bad index."""
        translation = {
            "segments": [
                {"start": 0.0, "end": 2.5, "original_text": "你好", "translated_text": "Xin chào", "speaker": None},
            ],
            "full_text_original": "你好",
            "full_text_translated": "Xin chào",
        }
        file_path = tmp_path / "translation.json"
        file_path.write_text(json.dumps(translation), encoding="utf-8")

        job = _make_job_state_with_artifacts(translation_path=str(file_path))
        job.checkpoint_type = CheckpointType.TRANSLATION
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        edits = [TranslationEdit(index=3, translated_text="test")]
        with pytest.raises(InvalidSegmentIndexError) as exc_info:
            manager.apply_translation_edits("job-001", edits)

        assert exc_info.value.index == 3
        assert exc_info.value.segment_count == 1
        assert exc_info.value.job_id == "job-001"

    def test_no_edits_applied_on_invalid_index(self, tmp_path):
        """If any index is invalid, no edits are applied (validation-first)."""
        translation = {
            "segments": [
                {"start": 0.0, "end": 2.5, "original_text": "你好", "translated_text": "Xin chào", "speaker": None},
                {"start": 2.5, "end": 5.0, "original_text": "世界", "translated_text": "Thế giới", "speaker": None},
            ],
            "full_text_original": "你好 世界",
            "full_text_translated": "Xin chào Thế giới",
        }
        file_path = tmp_path / "translation.json"
        file_path.write_text(json.dumps(translation), encoding="utf-8")

        job = _make_job_state_with_artifacts(translation_path=str(file_path))
        job.checkpoint_type = CheckpointType.TRANSLATION
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        edits = [
            TranslationEdit(index=0, translated_text="changed"),
            TranslationEdit(index=10, translated_text="bad"),
        ]
        with pytest.raises(InvalidSegmentIndexError):
            manager.apply_translation_edits("job-001", edits)

        # Verify original data is unchanged
        result = json.loads(file_path.read_text(encoding="utf-8"))
        assert result["segments"][0]["translated_text"] == "Xin chào"

    def test_preserves_unedited_segments(self, tmp_path):
        """Segments not in the edit list remain unchanged."""
        translation = {
            "segments": [
                {"start": 0.0, "end": 2.5, "original_text": "你好", "translated_text": "Xin chào", "speaker": None},
                {"start": 2.5, "end": 5.0, "original_text": "世界", "translated_text": "Thế giới", "speaker": None},
                {"start": 5.0, "end": 7.5, "original_text": "再见", "translated_text": "Tạm biệt", "speaker": None},
            ],
            "full_text_original": "你好 世界 再见",
            "full_text_translated": "Xin chào Thế giới Tạm biệt",
        }
        file_path = tmp_path / "translation.json"
        file_path.write_text(json.dumps(translation), encoding="utf-8")

        job = _make_job_state_with_artifacts(translation_path=str(file_path))
        job.checkpoint_type = CheckpointType.TRANSLATION
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        # Only edit index 1
        edits = [TranslationEdit(index=1, translated_text="Trái Đất")]
        manager.apply_translation_edits("job-001", edits)

        result = json.loads(file_path.read_text(encoding="utf-8"))
        assert result["segments"][0]["translated_text"] == "Xin chào"
        assert result["segments"][1]["translated_text"] == "Trái Đất"
        assert result["segments"][2]["translated_text"] == "Tạm biệt"


# --- Tests for InvalidSegmentIndexError ---


class TestInvalidSegmentIndexError:
    """Tests for InvalidSegmentIndexError exception."""

    def test_inherits_from_checkpoint_error(self):
        """InvalidSegmentIndexError is a subclass of CheckpointError."""
        assert issubclass(InvalidSegmentIndexError, CheckpointError)

    def test_stores_index_and_count(self):
        """InvalidSegmentIndexError stores index and segment_count."""
        err = InvalidSegmentIndexError(job_id="j1", index=5, segment_count=3)
        assert err.index == 5
        assert err.segment_count == 3
        assert err.job_id == "j1"

    def test_error_message_format(self):
        """InvalidSegmentIndexError has a descriptive message."""
        err = InvalidSegmentIndexError(job_id="j1", index=5, segment_count=3)
        assert "5" in str(err)
        assert "0 to 2" in str(err)
