"""Property-based tests for checkpoint confirmation edit logic.

Feature: pipeline-preview-confirm

Tests cover:
- Property 3: Confirmation without edits preserves original data
- Property 4: Confirmation with edits replaces stored result
- Property 5: Partial edits preserve unmodified segments
- Property 6: Whitespace-only text exclusion
- Property 7: Segment text length validation
- Property 8: Invalid segment index rejection
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from pydantic import ValidationError

from app.models.confirmation_schemas import SegmentEdit, TranslationEdit
from app.models.job import CheckpointType, JobState, JobStatus, PipelineStep
from app.services.checkpoint_manager import CheckpointManager, InvalidSegmentIndexError


# --- Helpers ---


class FakeJobStore:
    """In-memory job store for property testing of checkpoint logic."""

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

    def delete_job(self, job_id: str) -> None:
        pass

    def list_awaiting_confirmation_job_ids(self) -> list[str]:
        if self._state.status == JobStatus.AWAITING_CONFIRMATION:
            return [self._state.job_id]
        return []


def _make_job_state(
    job_id: str = "prop-test-job",
    url: str = "https://www.douyin.com/video/999",
    status: JobStatus = JobStatus.AWAITING_CONFIRMATION,
    work_dir: str = "",
    artifacts: dict | None = None,
    checkpoint_type: CheckpointType | None = CheckpointType.TRANSCRIPTION,
) -> JobState:
    now = datetime.now(timezone.utc)
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
        checkpoint_type=checkpoint_type,
        checkpoint_entered_at=now,
        confirmation_lock=False,
    )


# --- Hypothesis strategies ---


def _non_whitespace_text():
    """Generate text that is NOT whitespace-only (has at least one non-whitespace char)."""
    return st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)),
        min_size=1,
        max_size=50,
    ).filter(lambda t: t.strip() != "")


def _transcription_segments_strategy(min_segments: int = 1, max_segments: int = 20):
    """Generate a list of transcription segments with random non-whitespace text."""
    segment = st.fixed_dictionaries({
        "start": st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        "end": st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        "text": _non_whitespace_text(),
        "speaker": st.just("speaker_1"),
    })
    return st.lists(segment, min_size=min_segments, max_size=max_segments)


def _translation_segments_strategy(min_segments: int = 1, max_segments: int = 20):
    """Generate a list of translation segments with random non-whitespace text."""
    segment = st.fixed_dictionaries({
        "start": st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        "end": st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        "original_text": _non_whitespace_text(),
        "translated_text": _non_whitespace_text(),
        "speaker": st.just("speaker_1"),
    })
    return st.lists(segment, min_size=min_segments, max_size=max_segments)


# --- Property 3: Confirmation without edits preserves original data ---


@pytest.mark.property
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(segments=_transcription_segments_strategy())
def test_confirm_without_edits_preserves_transcription(
    segments: list[dict], tmp_path: Path
):
    """Feature: pipeline-preview-confirm, Property 3: Confirmation without edits preserves original data

    For any job at the transcription checkpoint, submitting a confirmation
    with no edits SHALL result in the pipeline resuming with the stored
    transcription result unchanged.

    **Validates: Requirements 1.8, 2.2**
    """
    # Write transcription JSON to a temp file
    transcription_data = {
        "segments": segments,
        "full_text": " ".join(seg["text"] for seg in segments),
        "language": "zh",
        "confidence": 0.95,
    }
    transcription_path = tmp_path / "transcription.json"
    transcription_path.write_text(
        json.dumps(transcription_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Create job state with artifact pointing to the file
    job_state = _make_job_state(
        work_dir=str(tmp_path),
        artifacts={"transcription_path": str(transcription_path)},
        checkpoint_type=CheckpointType.TRANSCRIPTION,
    )
    store = FakeJobStore(job_state)
    manager = CheckpointManager(job_store=store)

    # Save original file content
    original_content = transcription_path.read_bytes()

    # Call confirm_and_resume WITHOUT applying any edits
    next_step = manager.confirm_and_resume("prop-test-job")

    # Verify file contents are unchanged
    assert transcription_path.read_bytes() == original_content
    assert next_step == PipelineStep.TRANSLATING


@pytest.mark.property
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(segments=_translation_segments_strategy())
def test_confirm_without_edits_preserves_translation(
    segments: list[dict], tmp_path: Path
):
    """Feature: pipeline-preview-confirm, Property 3: Confirmation without edits preserves original data

    For any job at the translation checkpoint, submitting a confirmation
    with no edits SHALL result in the pipeline resuming with the stored
    translation result unchanged.

    **Validates: Requirements 1.8, 3.2**
    """
    # Write translation JSON to a temp file
    translation_data = {
        "segments": segments,
        "full_text_original": " ".join(seg["original_text"] for seg in segments),
        "full_text_translated": " ".join(seg["translated_text"] for seg in segments),
    }
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(
        json.dumps(translation_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Create job state with artifact pointing to the file
    job_state = _make_job_state(
        work_dir=str(tmp_path),
        artifacts={"translation_path": str(translation_path)},
        checkpoint_type=CheckpointType.TRANSLATION,
    )
    store = FakeJobStore(job_state)
    manager = CheckpointManager(job_store=store)

    # Save original file content
    original_content = translation_path.read_bytes()

    # Call confirm_and_resume WITHOUT applying any edits
    next_step = manager.confirm_and_resume("prop-test-job")

    # Verify file contents are unchanged
    assert translation_path.read_bytes() == original_content
    assert next_step == PipelineStep.SYNTHESIZING_VOICE


# --- Property 4: Confirmation with edits replaces stored result ---


@pytest.mark.property
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data())
def test_transcription_edits_replace_stored_result(
    data: st.DataObject, tmp_path: Path
):
    """Feature: pipeline-preview-confirm, Property 4: Confirmation with edits replaces stored result

    For any valid set of transcription edits, the stored result SHALL be
    updated to reflect exactly the edits.

    **Validates: Requirements 2.3**
    """
    # Generate random segments
    segments = data.draw(
        _transcription_segments_strategy(min_segments=1, max_segments=15)
    )
    num_segments = len(segments)

    # Generate random edits: random subset of indices with replacement text
    num_edits = data.draw(st.integers(min_value=1, max_value=num_segments))
    edit_indices = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=num_segments - 1),
            min_size=num_edits,
            max_size=num_edits,
            unique=True,
        )
    )
    edit_texts = data.draw(
        st.lists(
            _non_whitespace_text(),
            min_size=num_edits,
            max_size=num_edits,
        )
    )

    # Write transcription JSON
    transcription_data = {
        "segments": segments,
        "full_text": " ".join(seg["text"] for seg in segments),
        "language": "zh",
        "confidence": 0.95,
    }
    transcription_path = tmp_path / "transcription.json"
    transcription_path.write_text(
        json.dumps(transcription_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Create job state
    job_state = _make_job_state(
        work_dir=str(tmp_path),
        artifacts={"transcription_path": str(transcription_path)},
        checkpoint_type=CheckpointType.TRANSCRIPTION,
    )
    store = FakeJobStore(job_state)
    manager = CheckpointManager(job_store=store)

    # Build SegmentEdit list
    edits = [
        SegmentEdit(index=idx, text=txt)
        for idx, txt in zip(edit_indices, edit_texts)
    ]

    # Apply edits
    manager.apply_transcription_edits("prop-test-job", edits)

    # Load the result and verify edits were applied
    with open(transcription_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    # Build expected segments: apply edits to original, then filter whitespace-only
    expected_segments = [dict(seg) for seg in segments]
    for idx, txt in zip(edit_indices, edit_texts):
        expected_segments[idx]["text"] = txt
    expected_segments = [seg for seg in expected_segments if seg["text"].strip()]

    assert len(result_data["segments"]) == len(expected_segments)
    for i, (actual, expected) in enumerate(
        zip(result_data["segments"], expected_segments)
    ):
        assert actual["text"] == expected["text"], (
            f"Segment {i}: expected text '{expected['text']}' but got '{actual['text']}'"
        )


@pytest.mark.property
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data())
def test_translation_edits_replace_stored_result(
    data: st.DataObject, tmp_path: Path
):
    """Feature: pipeline-preview-confirm, Property 4: Confirmation with edits replaces stored result

    For any valid set of translation edits, the stored result SHALL be
    updated to reflect exactly the edits.

    **Validates: Requirements 3.3**
    """
    # Generate random segments
    segments = data.draw(
        _translation_segments_strategy(min_segments=1, max_segments=15)
    )
    num_segments = len(segments)

    # Generate random edits
    num_edits = data.draw(st.integers(min_value=1, max_value=num_segments))
    edit_indices = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=num_segments - 1),
            min_size=num_edits,
            max_size=num_edits,
            unique=True,
        )
    )
    edit_texts = data.draw(
        st.lists(
            _non_whitespace_text(),
            min_size=num_edits,
            max_size=num_edits,
        )
    )

    # Write translation JSON
    translation_data = {
        "segments": segments,
        "full_text_original": " ".join(seg["original_text"] for seg in segments),
        "full_text_translated": " ".join(seg["translated_text"] for seg in segments),
    }
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(
        json.dumps(translation_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Create job state
    job_state = _make_job_state(
        work_dir=str(tmp_path),
        artifacts={"translation_path": str(translation_path)},
        checkpoint_type=CheckpointType.TRANSLATION,
    )
    store = FakeJobStore(job_state)
    manager = CheckpointManager(job_store=store)

    # Build TranslationEdit list
    edits = [
        TranslationEdit(index=idx, translated_text=txt)
        for idx, txt in zip(edit_indices, edit_texts)
    ]

    # Apply edits
    manager.apply_translation_edits("prop-test-job", edits)

    # Load the result and verify edits were applied
    with open(translation_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    # Build expected segments: apply edits to original, then filter whitespace-only
    expected_segments = [dict(seg) for seg in segments]
    for idx, txt in zip(edit_indices, edit_texts):
        expected_segments[idx]["translated_text"] = txt
    expected_segments = [
        seg for seg in expected_segments if seg["translated_text"].strip()
    ]

    assert len(result_data["segments"]) == len(expected_segments)
    for i, (actual, expected) in enumerate(
        zip(result_data["segments"], expected_segments)
    ):
        assert actual["translated_text"] == expected["translated_text"], (
            f"Segment {i}: expected translated_text '{expected['translated_text']}' "
            f"but got '{actual['translated_text']}'"
        )


# --- Property 5: Partial edits preserve unmodified segments ---


@pytest.mark.property
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data())
def test_partial_transcription_edits_preserve_unmodified_segments(
    data: st.DataObject, tmp_path: Path
):
    """Feature: pipeline-preview-confirm, Property 5: Partial edits preserve unmodified segments

    For any transcription with N segments and a subset S of edited indices,
    segments NOT in S SHALL remain byte-for-byte identical.

    **Validates: Requirements 2.4**
    """
    # Generate segments (at least 2 so we can have edited and unedited)
    segments = data.draw(
        _transcription_segments_strategy(min_segments=2, max_segments=15)
    )
    num_segments = len(segments)

    # Choose a random strict subset of indices to edit (not all indices)
    max_edits = max(1, num_segments - 1)  # Leave at least 1 unedited
    num_edits = data.draw(st.integers(min_value=1, max_value=max_edits))
    edit_indices = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=num_segments - 1),
            min_size=num_edits,
            max_size=num_edits,
            unique=True,
        )
    )
    edit_texts = data.draw(
        st.lists(
            _non_whitespace_text(),
            min_size=num_edits,
            max_size=num_edits,
        )
    )

    # Write transcription JSON
    transcription_data = {
        "segments": segments,
        "full_text": " ".join(seg["text"] for seg in segments),
        "language": "zh",
        "confidence": 0.95,
    }
    transcription_path = tmp_path / "transcription.json"
    transcription_path.write_text(
        json.dumps(transcription_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Create job state
    job_state = _make_job_state(
        work_dir=str(tmp_path),
        artifacts={"transcription_path": str(transcription_path)},
        checkpoint_type=CheckpointType.TRANSCRIPTION,
    )
    store = FakeJobStore(job_state)
    manager = CheckpointManager(job_store=store)

    # Build edits
    edits = [
        SegmentEdit(index=idx, text=txt)
        for idx, txt in zip(edit_indices, edit_texts)
    ]

    # Apply edits
    manager.apply_transcription_edits("prop-test-job", edits)

    # Load result
    with open(transcription_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    # Build the expected full list after edits (before whitespace filtering)
    expected_all = [dict(seg) for seg in segments]
    for idx, txt in zip(edit_indices, edit_texts):
        expected_all[idx]["text"] = txt

    # After filtering, unedited segments that were not whitespace-only should be preserved
    edit_indices_set = set(edit_indices)
    # Track which original indices survive the whitespace filter
    surviving_segments = [
        (i, seg) for i, seg in enumerate(expected_all) if seg["text"].strip()
    ]

    # For each surviving segment that was NOT edited, verify it's identical
    result_idx = 0
    for orig_idx, expected_seg in surviving_segments:
        actual_seg = result_data["segments"][result_idx]
        if orig_idx not in edit_indices_set:
            # This segment was NOT edited — should be byte-for-byte identical
            assert actual_seg["text"] == segments[orig_idx]["text"], (
                f"Unmodified segment at original index {orig_idx} changed: "
                f"expected '{segments[orig_idx]['text']}' but got '{actual_seg['text']}'"
            )
            assert actual_seg["start"] == segments[orig_idx]["start"]
            assert actual_seg["end"] == segments[orig_idx]["end"]
        result_idx += 1


@pytest.mark.property
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data())
def test_partial_translation_edits_preserve_unmodified_segments(
    data: st.DataObject, tmp_path: Path
):
    """Feature: pipeline-preview-confirm, Property 5: Partial edits preserve unmodified segments

    For any translation with N segments and a subset S of edited indices,
    segments NOT in S SHALL remain byte-for-byte identical.

    **Validates: Requirements 3.4**
    """
    # Generate segments (at least 2 so we can have edited and unedited)
    segments = data.draw(
        _translation_segments_strategy(min_segments=2, max_segments=15)
    )
    num_segments = len(segments)

    # Choose a random strict subset of indices to edit (not all indices)
    max_edits = max(1, num_segments - 1)
    num_edits = data.draw(st.integers(min_value=1, max_value=max_edits))
    edit_indices = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=num_segments - 1),
            min_size=num_edits,
            max_size=num_edits,
            unique=True,
        )
    )
    edit_texts = data.draw(
        st.lists(
            _non_whitespace_text(),
            min_size=num_edits,
            max_size=num_edits,
        )
    )

    # Write translation JSON
    translation_data = {
        "segments": segments,
        "full_text_original": " ".join(seg["original_text"] for seg in segments),
        "full_text_translated": " ".join(seg["translated_text"] for seg in segments),
    }
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(
        json.dumps(translation_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Create job state
    job_state = _make_job_state(
        work_dir=str(tmp_path),
        artifacts={"translation_path": str(translation_path)},
        checkpoint_type=CheckpointType.TRANSLATION,
    )
    store = FakeJobStore(job_state)
    manager = CheckpointManager(job_store=store)

    # Build edits
    edits = [
        TranslationEdit(index=idx, translated_text=txt)
        for idx, txt in zip(edit_indices, edit_texts)
    ]

    # Apply edits
    manager.apply_translation_edits("prop-test-job", edits)

    # Load result
    with open(translation_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    # Build expected full list after edits (before whitespace filtering)
    expected_all = [dict(seg) for seg in segments]
    for idx, txt in zip(edit_indices, edit_texts):
        expected_all[idx]["translated_text"] = txt

    # After filtering, unedited segments that were not whitespace-only should be preserved
    edit_indices_set = set(edit_indices)
    surviving_segments = [
        (i, seg)
        for i, seg in enumerate(expected_all)
        if seg["translated_text"].strip()
    ]

    # For each surviving segment that was NOT edited, verify it's identical
    result_idx = 0
    for orig_idx, expected_seg in surviving_segments:
        actual_seg = result_data["segments"][result_idx]
        if orig_idx not in edit_indices_set:
            # This segment was NOT edited — should be byte-for-byte identical
            assert actual_seg["translated_text"] == segments[orig_idx]["translated_text"], (
                f"Unmodified segment at original index {orig_idx} changed: "
                f"expected '{segments[orig_idx]['translated_text']}' "
                f"but got '{actual_seg['translated_text']}'"
            )
            assert actual_seg["original_text"] == segments[orig_idx]["original_text"]
            assert actual_seg["start"] == segments[orig_idx]["start"]
            assert actual_seg["end"] == segments[orig_idx]["end"]
        result_idx += 1


# --- Property 6: Whitespace-only text exclusion ---


@pytest.mark.property
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.filter_too_much])
@given(data=st.data())
def test_whitespace_only_transcription_segments_excluded(
    data: st.DataObject, tmp_path: Path
):
    """Feature: pipeline-preview-confirm, Property 6: Whitespace-only text exclusion

    For any string composed entirely of whitespace characters submitted as
    segment text, that segment SHALL be excluded from subsequent processing.

    **Validates: Requirements 2.5**
    """
    # Generate at least 2 segments so we have some remaining after exclusion
    segments = data.draw(
        _transcription_segments_strategy(min_segments=2, max_segments=10)
    )
    num_segments = len(segments)

    # Choose an index to replace with whitespace-only text
    whitespace_index = data.draw(
        st.integers(min_value=0, max_value=num_segments - 1)
    )

    # Generate whitespace-only text (spaces, tabs, newlines, etc.)
    whitespace_text = data.draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("Zs", "Cc")),
            min_size=1,
            max_size=20,
        ).filter(lambda t: t.strip() == "")
    )

    # Write transcription JSON
    transcription_data = {
        "segments": segments,
        "full_text": " ".join(seg["text"] for seg in segments),
        "language": "zh",
        "confidence": 0.95,
    }
    transcription_path = tmp_path / "transcription.json"
    transcription_path.write_text(
        json.dumps(transcription_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Create job state
    job_state = _make_job_state(
        work_dir=str(tmp_path),
        artifacts={"transcription_path": str(transcription_path)},
        checkpoint_type=CheckpointType.TRANSCRIPTION,
    )
    store = FakeJobStore(job_state)
    manager = CheckpointManager(job_store=store)

    # Apply edit with whitespace-only text
    edits = [SegmentEdit(index=whitespace_index, text=whitespace_text)]
    manager.apply_transcription_edits("prop-test-job", edits)

    # Load result
    with open(transcription_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    # The whitespace-only segment should NOT appear in the result
    for seg in result_data["segments"]:
        assert seg["text"].strip() != "", (
            f"Whitespace-only segment found in result: '{seg['text']}'"
        )

    # The result should have fewer segments than the original
    assert len(result_data["segments"]) == num_segments - 1, (
        f"Expected {num_segments - 1} segments after excluding whitespace, "
        f"got {len(result_data['segments'])}"
    )


@pytest.mark.property
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.filter_too_much])
@given(data=st.data())
def test_whitespace_only_translation_segments_excluded(
    data: st.DataObject, tmp_path: Path
):
    """Feature: pipeline-preview-confirm, Property 6: Whitespace-only text exclusion

    For any string composed entirely of whitespace characters submitted as
    translated_text, that segment SHALL be excluded from subsequent processing.

    **Validates: Requirements 3.5**
    """
    # Generate at least 2 segments so we have some remaining after exclusion
    segments = data.draw(
        _translation_segments_strategy(min_segments=2, max_segments=10)
    )
    num_segments = len(segments)

    # Choose an index to replace with whitespace-only text
    whitespace_index = data.draw(
        st.integers(min_value=0, max_value=num_segments - 1)
    )

    # Generate whitespace-only text (spaces, tabs, newlines, etc.)
    whitespace_text = data.draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("Zs", "Cc")),
            min_size=1,
            max_size=20,
        ).filter(lambda t: t.strip() == "")
    )

    # Write translation JSON
    translation_data = {
        "segments": segments,
        "full_text_original": " ".join(seg["original_text"] for seg in segments),
        "full_text_translated": " ".join(seg["translated_text"] for seg in segments),
    }
    translation_path = tmp_path / "translation.json"
    translation_path.write_text(
        json.dumps(translation_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Create job state
    job_state = _make_job_state(
        work_dir=str(tmp_path),
        artifacts={"translation_path": str(translation_path)},
        checkpoint_type=CheckpointType.TRANSLATION,
    )
    store = FakeJobStore(job_state)
    manager = CheckpointManager(job_store=store)

    # Apply edit with whitespace-only translated_text
    edits = [TranslationEdit(index=whitespace_index, translated_text=whitespace_text)]
    manager.apply_translation_edits("prop-test-job", edits)

    # Load result
    with open(translation_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    # The whitespace-only segment should NOT appear in the result
    for seg in result_data["segments"]:
        assert seg["translated_text"].strip() != "", (
            f"Whitespace-only segment found in result: '{seg['translated_text']}'"
        )

    # The result should have fewer segments than the original
    assert len(result_data["segments"]) == num_segments - 1, (
        f"Expected {num_segments - 1} segments after excluding whitespace, "
        f"got {len(result_data['segments'])}"
    )


# --- Property 7: Segment text length validation ---


# Feature: pipeline-preview-confirm, Property 7: Segment text length validation
@pytest.mark.property
class TestSegmentTextLengthValidation:
    """Property 7: Segment text length validation.

    For any transcription edit where segment text exceeds 500 characters,
    OR any translation edit where translated_text exceeds 5000 characters,
    the API SHALL reject the entire submission with a validation error,
    and the stored data SHALL remain unchanged.

    **Validates: Requirements 2.6, 3.3**
    """

    @given(long_text=st.text(min_size=501, max_size=600))
    @settings(max_examples=100)
    def test_segment_edit_rejects_text_over_500_chars(self, long_text: str):
        """SegmentEdit with text > 500 characters raises Pydantic ValidationError.

        # Feature: pipeline-preview-confirm, Property 7: Segment text length validation
        """
        with pytest.raises(ValidationError):
            SegmentEdit(index=0, text=long_text)

    @given(long_text=st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=5001, max_size=5100))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.large_base_example, HealthCheck.data_too_large])
    def test_translation_edit_rejects_text_over_5000_chars(self, long_text: str):
        """TranslationEdit with translated_text > 5000 characters raises Pydantic ValidationError.

        # Feature: pipeline-preview-confirm, Property 7: Segment text length validation
        """
        with pytest.raises(ValidationError):
            TranslationEdit(index=0, translated_text=long_text)

    @given(
        text=st.text(min_size=1, max_size=500).filter(lambda t: t.strip() != ""),
    )
    @settings(max_examples=100)
    def test_segment_edit_accepts_text_at_or_below_500_chars(self, text: str):
        """SegmentEdit with text <= 500 characters is accepted by Pydantic.

        # Feature: pipeline-preview-confirm, Property 7: Segment text length validation
        """
        edit = SegmentEdit(index=0, text=text)
        assert edit.text == text
        assert len(edit.text) <= 500

    @given(
        text=st.text(min_size=1, max_size=5000).filter(lambda t: t.strip() != ""),
    )
    @settings(max_examples=100)
    def test_translation_edit_accepts_text_at_or_below_5000_chars(self, text: str):
        """TranslationEdit with translated_text <= 5000 characters is accepted by Pydantic.

        # Feature: pipeline-preview-confirm, Property 7: Segment text length validation
        """
        edit = TranslationEdit(index=0, translated_text=text)
        assert edit.translated_text == text
        assert len(edit.translated_text) <= 5000


# --- Property 8: Invalid segment index rejection ---


# Feature: pipeline-preview-confirm, Property 8: Invalid segment index rejection
@pytest.mark.property
class TestInvalidSegmentIndexRejection:
    """Property 8: Invalid segment index rejection.

    For any edit submission referencing a segment index that is negative
    or >= the total number of segments, the API SHALL reject the entire
    submission with an error indicating the invalid index.

    **Validates: Requirements 3.6**
    """

    @given(
        segment_count=st.integers(min_value=1, max_value=20),
        data=st.data(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_transcription_edit_rejects_index_above_segment_count(
        self, segment_count: int, data, tmp_path_factory
    ):
        """apply_transcription_edits raises InvalidSegmentIndexError for index >= segment_count.

        # Feature: pipeline-preview-confirm, Property 8: Invalid segment index rejection
        """
        # Generate an invalid index >= segment_count
        invalid_index = data.draw(
            st.integers(min_value=segment_count, max_value=segment_count + 1000)
        )

        # Create segments
        segments = [
            {"start": float(i), "end": float(i + 1), "text": f"segment {i}", "speaker": None}
            for i in range(segment_count)
        ]

        tmp_path = tmp_path_factory.mktemp("prop8_trans")
        transcription_data = {"segments": segments, "language": "zh", "confidence": 0.95}
        file_path = tmp_path / "transcription.json"
        file_path.write_text(json.dumps(transcription_data, ensure_ascii=False), encoding="utf-8")

        original_content = file_path.read_text(encoding="utf-8")

        now = datetime.now(timezone.utc)
        job = JobState(
            job_id="job-prop8",
            url="https://www.douyin.com/video/123",
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSCRIPTION,
            checkpoint_entered_at=now,
            created_at=now,
            updated_at=now,
            work_dir=str(tmp_path),
            artifacts={"transcription_path": str(file_path)},
        )
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        edit = SegmentEdit(index=invalid_index, text="edited")
        with pytest.raises(InvalidSegmentIndexError) as exc_info:
            manager.apply_transcription_edits("job-prop8", [edit])

        assert exc_info.value.index == invalid_index
        assert exc_info.value.segment_count == segment_count

        # Verify stored data is unchanged
        assert file_path.read_text(encoding="utf-8") == original_content

    @given(
        segment_count=st.integers(min_value=1, max_value=20),
        data=st.data(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_translation_edit_rejects_index_above_segment_count(
        self, segment_count: int, data, tmp_path_factory
    ):
        """apply_translation_edits raises InvalidSegmentIndexError for index >= segment_count.

        # Feature: pipeline-preview-confirm, Property 8: Invalid segment index rejection
        """
        # Generate an invalid index >= segment_count
        invalid_index = data.draw(
            st.integers(min_value=segment_count, max_value=segment_count + 1000)
        )

        # Create segments
        segments = [
            {
                "start": float(i),
                "end": float(i + 1),
                "original_text": f"原文 {i}",
                "translated_text": f"Bản dịch {i}",
                "speaker": None,
            }
            for i in range(segment_count)
        ]

        tmp_path = tmp_path_factory.mktemp("prop8_transl")
        translation_data = {"segments": segments}
        file_path = tmp_path / "translation.json"
        file_path.write_text(json.dumps(translation_data, ensure_ascii=False), encoding="utf-8")

        original_content = file_path.read_text(encoding="utf-8")

        now = datetime.now(timezone.utc)
        job = JobState(
            job_id="job-prop8",
            url="https://www.douyin.com/video/123",
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSLATION,
            checkpoint_entered_at=now,
            created_at=now,
            updated_at=now,
            work_dir=str(tmp_path),
            artifacts={"translation_path": str(file_path)},
        )
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        edit = TranslationEdit(index=invalid_index, translated_text="edited")
        with pytest.raises(InvalidSegmentIndexError) as exc_info:
            manager.apply_translation_edits("job-prop8", [edit])

        assert exc_info.value.index == invalid_index
        assert exc_info.value.segment_count == segment_count

        # Verify stored data is unchanged
        assert file_path.read_text(encoding="utf-8") == original_content

    @given(
        segment_count=st.integers(min_value=1, max_value=20),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_negative_index_rejected_by_pydantic_schema(
        self, segment_count: int, data
    ):
        """SegmentEdit and TranslationEdit reject negative indices at schema level.

        # Feature: pipeline-preview-confirm, Property 8: Invalid segment index rejection
        """
        negative_index = data.draw(st.integers(min_value=-1000, max_value=-1))

        # SegmentEdit should reject negative index (ge=0 constraint)
        with pytest.raises(ValidationError):
            SegmentEdit(index=negative_index, text="test")

        # TranslationEdit should also reject negative index (ge=0 constraint)
        with pytest.raises(ValidationError):
            TranslationEdit(index=negative_index, translated_text="test")

    @given(
        segment_count=st.integers(min_value=1, max_value=20),
        data=st.data(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_mixed_valid_and_invalid_indices_rejects_all(
        self, segment_count: int, data, tmp_path_factory
    ):
        """When edits contain both valid and invalid indices, no edits are applied.

        # Feature: pipeline-preview-confirm, Property 8: Invalid segment index rejection
        """
        # Generate one valid and one invalid index
        valid_index = data.draw(st.integers(min_value=0, max_value=segment_count - 1))
        invalid_index = data.draw(
            st.integers(min_value=segment_count, max_value=segment_count + 1000)
        )

        # Create transcription segments
        segments = [
            {"start": float(i), "end": float(i + 1), "text": f"segment {i}", "speaker": None}
            for i in range(segment_count)
        ]

        tmp_path = tmp_path_factory.mktemp("prop8_mixed")
        transcription_data = {"segments": segments, "language": "zh", "confidence": 0.95}
        file_path = tmp_path / "transcription.json"
        file_path.write_text(json.dumps(transcription_data, ensure_ascii=False), encoding="utf-8")

        original_content = file_path.read_text(encoding="utf-8")

        now = datetime.now(timezone.utc)
        job = JobState(
            job_id="job-prop8",
            url="https://www.douyin.com/video/123",
            status=JobStatus.AWAITING_CONFIRMATION,
            checkpoint_type=CheckpointType.TRANSCRIPTION,
            checkpoint_entered_at=now,
            created_at=now,
            updated_at=now,
            work_dir=str(tmp_path),
            artifacts={"transcription_path": str(file_path)},
        )
        store = FakeJobStore(job)
        manager = CheckpointManager(store)

        edits = [
            SegmentEdit(index=valid_index, text="valid edit"),
            SegmentEdit(index=invalid_index, text="invalid edit"),
        ]
        with pytest.raises(InvalidSegmentIndexError):
            manager.apply_transcription_edits("job-prop8", edits)

        # Verify stored data is unchanged — no partial application
        assert file_path.read_text(encoding="utf-8") == original_content
