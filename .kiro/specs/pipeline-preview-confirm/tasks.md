# Implementation Plan: Pipeline Preview & Confirm

## Overview

Implement a checkpoint/pause system in the Douyin video translation pipeline that pauses after transcription, translation, and voice preview steps. Users can review results, make edits, and confirm before the pipeline resumes. The implementation extends existing models, adds new API endpoints, a CheckpointManager service, resume/expiry Celery tasks, and property-based tests.

## Tasks

- [x] 1. Extend job models and add checkpoint data structures
  - [x] 1.1 Add new statuses and checkpoint types to job model
    - Add `AWAITING_CONFIRMATION` and `EXPIRED` to `JobStatus` enum in `app/models/job.py`
    - Add `CheckpointType` enum with values: `TRANSCRIPTION`, `TRANSLATION`, `VOICE_SELECTION`
    - Add `VoiceOption` Pydantic model with fields: `voice_id`, `voice_name`, `preview_url`
    - Extend `JobState` with fields: `checkpoint_type`, `checkpoint_entered_at`, `confirmation_lock`, `voice_options`
    - _Requirements: 1.1, 1.2, 1.3, 6.1_

  - [x] 1.2 Add request/response schemas for confirmation endpoints
    - Create `SegmentEdit` model (index: int >= 0, text: str max 500 chars)
    - Create `TranscriptionConfirmRequest` model with optional list of `SegmentEdit`
    - Create `TranslationEdit` model (index: int >= 0, translated_text: str max 5000 chars)
    - Create `TranslationConfirmRequest` model with optional list of `TranslationEdit`
    - Create `VoiceConfirmRequest` model with `voice_id` (min_length=1)
    - Create `ConfirmResponse` model (job_id, status, next_step, message)
    - Add to `app/models/schemas.py` or create `app/models/confirmation_schemas.py`
    - _Requirements: 5.1, 5.2, 5.3, 2.6, 3.3_

  - [x] 1.3 Extend JobStatusResponse with checkpoint preview data
    - Create `TranscriptionPreviewSegment` model (index, start, end, text, confidence)
    - Create `TranslationPreviewSegment` model (index, start, end, original_text, translated_text)
    - Create `VoiceOptionResponse` model (voice_id, voice_name, preview_url)
    - Create `PreviewData` model containing optional lists of the above
    - Add `checkpoint_type` and `preview_data` fields to `JobStatusResponse`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.6_

- [x] 2. Implement CheckpointManager service
  - [x] 2.1 Create CheckpointManager with pause and validation logic
    - Create `app/services/checkpoint_manager.py`
    - Implement `__init__` accepting a `JobStoreProtocol`
    - Implement `pause_at_checkpoint(job_id, checkpoint_type)` — sets status to AWAITING_CONFIRMATION, stores checkpoint_type and checkpoint_entered_at
    - Implement `validate_confirmation(job_id, expected_checkpoint)` — checks job status, checkpoint match, acquires confirmation_lock; raises errors for wrong state/checkpoint/concurrent lock
    - Implement `confirm_and_resume(job_id)` — releases lock, sets status to PROCESSING, returns next PipelineStep
    - _Requirements: 1.4, 1.6, 1.8, 5.4, 5.7, 5.8_

  - [x] 2.2 Add transcription and translation edit methods to CheckpointManager
    - Implement `apply_transcription_edits(job_id, edits)` — loads transcription JSON from artifacts, applies edits by index, excludes whitespace-only segments, saves back
    - Implement `apply_translation_edits(job_id, edits)` — loads translation JSON from artifacts, applies edits by index, validates segment index bounds, excludes whitespace-only segments, saves back
    - Validate segment index is within range; raise error for out-of-bounds
    - _Requirements: 2.3, 2.4, 2.5, 3.3, 3.4, 3.5, 3.6_

  - [x] 2.3 Add expiration logic to CheckpointManager
    - Implement `check_expired_jobs()` — scans jobs with AWAITING_CONFIRMATION status, expires those past 24h since last timer reset, deletes working directory, removes from active set
    - Implement `reset_expiration(job_id)` — resets checkpoint_entered_at to now (called on status query)
    - _Requirements: 7.1, 7.2, 7.3_

  - [x] 2.4 Write property tests for CheckpointManager pause behavior
    - **Property 1: Checkpoint pause preserves results**
    - **Property 2: No progression while awaiting confirmation**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

  - [x] 2.5 Write property tests for confirmation edit logic
    - **Property 3: Confirmation without edits preserves original data**
    - **Property 4: Confirmation with edits replaces stored result**
    - **Property 5: Partial edits preserve unmodified segments**
    - **Property 6: Whitespace-only text exclusion**
    - **Validates: Requirements 1.8, 2.2, 2.3, 2.4, 2.5, 3.2, 3.3, 3.4, 3.5**

  - [x] 2.6 Write property tests for validation and error handling
    - **Property 7: Segment text length validation**
    - **Property 8: Invalid segment index rejection**
    - **Validates: Requirements 2.6, 3.3, 3.6**

- [x] 3. Modify pipeline to support checkpoints and resumption
  - [x] 3.1 Add CheckpointPauseSignal and checkpoint mapping to pipeline
    - Create `CheckpointPauseSignal` exception class in `app/services/pipeline.py`
    - Define `CHECKPOINT_AFTER_STEP` dict mapping `PipelineStep` → `CheckpointType` for RECOGNIZING_SPEECH, TRANSLATING, and SYNTHESIZING_VOICE
    - Modify `TranslationPipeline._execute_step` to check if completed step triggers a checkpoint; if so, call `CheckpointManager.pause_at_checkpoint()` and raise `CheckpointPauseSignal`
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 3.2 Implement voice preview generation step in pipeline
    - After TRANSLATING checkpoint confirmation, before full synthesis, generate voice previews
    - Select longest translated segment that does not exceed 15 seconds estimated speech duration
    - Generate preview samples for at least 3 voice options (5-15 seconds each)
    - Store previews in `{work_dir}/voice_previews/{voice_id}_preview.mp3`
    - Store `voice_options.json` metadata
    - Handle partial failures: proceed if >= 2 voice previews succeed, fail otherwise
    - _Requirements: 4.1, 4.2, 4.4, 4.6_

  - [x] 3.3 Modify pipeline execute/resume to handle CheckpointPauseSignal
    - In `TranslationPipeline.execute()`, catch `CheckpointPauseSignal` and return cleanly (task completes, not fails)
    - In `TranslationPipeline.resume()`, accept `from_step` parameter, load artifacts from Redis, and continue pipeline from the specified step
    - Ensure resume uses edited data when available (transcription/translation edits applied by CheckpointManager)
    - _Requirements: 1.6, 1.8, 4.3_

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement confirmation API endpoints
  - [x] 5.1 Create confirmation routes module with transcription endpoint
    - Create `app/api/confirmation_routes.py` with APIRouter (prefix="/api/v1/jobs")
    - Implement `POST /{job_id}/confirm/transcription` endpoint
    - Validate request body (SegmentEdit list, max 500 chars per segment text)
    - Call `CheckpointManager.validate_confirmation()` then `apply_transcription_edits()` then `confirm_and_resume()`
    - Enqueue `resume_pipeline_task` with next step
    - Return `ConfirmResponse` with updated status and next step
    - Handle all error cases: 404, 409 (wrong status, wrong checkpoint, concurrent), 410 (expired), 422 (validation)
    - _Requirements: 5.1, 5.4, 5.5, 5.6, 5.7, 5.8, 2.3, 2.6_

  - [x] 5.2 Add translation and voice confirmation endpoints
    - Implement `POST /{job_id}/confirm/translation` endpoint
    - Validate TranslationEdit (max 5000 chars, valid segment indices)
    - Implement `POST /{job_id}/confirm/voice` endpoint
    - Validate voice_id against available voice options, return 422 if unavailable
    - Store selected voice_id in job artifacts
    - Both endpoints follow same pattern: validate → apply edits → confirm → enqueue resume
    - _Requirements: 5.2, 5.3, 3.3, 3.6, 4.5_

  - [x] 5.3 Add voice preview audio endpoint
    - Implement `GET /{job_id}/preview/voice/{voice_id}` — serves preview audio file
    - Validate job exists and is at voice_selection checkpoint
    - Return audio file with appropriate content-type header
    - _Requirements: 4.2_

  - [x] 5.4 Register confirmation routes in FastAPI app
    - Import and include `confirmation_routes.router` in `app/main.py`
    - Configure dependencies (job_store, checkpoint_manager) for confirmation routes
    - _Requirements: 5.1, 5.2, 5.3_

  - [x] 5.5 Write property tests for confirmation API behavior
    - **Property 11: Confirmation for wrong job status returns 409**
    - **Property 12: Confirmation for wrong checkpoint type returns 409**
    - **Property 13: Valid confirmation returns updated status and enqueues resumption**
    - **Validates: Requirements 5.4, 5.6, 5.7**

- [x] 6. Update job status API to include checkpoint data
  - [x] 6.1 Modify GET /jobs/{id} to return checkpoint preview data
    - Update `get_job_status` route in `app/api/routes.py`
    - When status is AWAITING_CONFIRMATION, load preview data from artifacts and return in response
    - For transcription checkpoint: return segments with start, end, text, confidence
    - For translation checkpoint: return segments with start, end, original_text, translated_text
    - For voice_selection checkpoint: return voice options with voice_id, voice_name, preview_url
    - When status is not AWAITING_CONFIRMATION, return checkpoint_type and preview_data as null
    - Call `CheckpointManager.reset_expiration()` when job is at checkpoint (resets 24h timer)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 7.3_

  - [x] 6.2 Write property tests for status response correctness
    - **Property 9: Status response includes correct preview data at checkpoint**
    - **Property 10: Non-awaiting jobs omit checkpoint fields**
    - **Property 14: Status query resets expiration timer**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.6, 7.3**

- [x] 7. Implement Celery tasks for resume and expiry
  - [x] 7.1 Create resume_pipeline_task
    - Create `app/tasks/resume_task.py`
    - Implement `resume_pipeline_task(job_id, from_step)` Celery task
    - Instantiate pipeline with dependencies and call `pipeline.resume(job_id, from_step)`
    - Handle pipeline errors: update job status to "failed" with error details
    - _Requirements: 5.6, 1.6, 1.8_

  - [x] 7.2 Create checkpoint expiry periodic task
    - Create `app/tasks/expiry_task.py`
    - Implement `check_checkpoint_expiry_task()` Celery task
    - Call `CheckpointManager.check_expired_jobs()`
    - Register as Celery Beat periodic task with interval <= 60 seconds (use 30s as designed)
    - Configure Celery Beat schedule in `app/core/celery_app.py`
    - _Requirements: 7.1, 7.2, 7.5_

  - [x] 7.3 Write property test for voice preview segment selection
    - **Property 15: Preview segment selection uses longest under 15 seconds**
    - **Validates: Requirements 4.4**

- [x] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Integration wiring and final validation
  - [x] 9.1 Wire CheckpointManager into pipeline and route dependencies
    - Instantiate `CheckpointManager` in `app/core/dependencies.py`
    - Pass `CheckpointManager` to `TranslationPipeline` constructor
    - Pass `CheckpointManager` to confirmation routes via dependency injection
    - Ensure `resume_pipeline_task` can access CheckpointManager and pipeline
    - _Requirements: 1.1, 1.2, 1.3, 5.6_

  - [x] 9.2 Handle expired job confirmation in API
    - When confirmation endpoint receives request for expired job, return HTTP 410 with JOB_EXPIRED error
    - Ensure expired status check happens before confirmation_lock acquisition
    - _Requirements: 7.4_

  - [x] 9.3 Write integration tests for full pipeline checkpoint flow
    - Test complete flow: create job → transcription checkpoint → confirm with edits → translation checkpoint → confirm → voice checkpoint → select voice → completion
    - Test expiry scenario: create job → reach checkpoint → simulate 24h passage → verify expiration
    - Test concurrent confirmation rejection
    - _Requirements: 1.1–1.8, 5.4–5.8, 7.1–7.4_

- [x] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document using Hypothesis (already configured in project)
- Unit tests validate specific examples and edge cases
- The project uses Python with FastAPI, Celery, Redis, and Pydantic — all tasks use these existing technologies
- Voice preview generation requires integration with the existing `VoiceSynthesizer` service

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["2.1", "3.1"] },
    { "id": 3, "tasks": ["2.2", "2.3", "3.2", "3.3"] },
    { "id": 4, "tasks": ["2.4", "2.5", "2.6"] },
    { "id": 5, "tasks": ["5.1", "5.2", "5.3", "6.1"] },
    { "id": 6, "tasks": ["5.4", "5.5", "6.2"] },
    { "id": 7, "tasks": ["7.1", "7.2", "7.3"] },
    { "id": 8, "tasks": ["9.1", "9.2"] },
    { "id": 9, "tasks": ["9.3"] }
  ]
}
```
