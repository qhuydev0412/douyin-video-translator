# Implementation Plan: Douyin Video Translator

## Overview

Triển khai standalone Python backend service dịch video Douyin từ tiếng Trung sang tiếng Việt. Sử dụng FastAPI + Celery + Redis với pipeline: download → extract audio → isolate vocals → speech recognition → translate → TTS → compose video.

## Tasks

- [x] 1. Set up project structure and core dependencies
  - [x] 1.1 Initialize Python project with pyproject.toml and install core dependencies
    - Create `pyproject.toml` with dependencies: fastapi, uvicorn, celery, redis, yt-dlp, openai-whisper, demucs, edge-tts, google-cloud-translate, pydantic, httpx
    - Create directory structure: `app/`, `app/api/`, `app/core/`, `app/models/`, `app/services/`, `app/tasks/`, `app/utils/`, `storage/jobs/`, `tests/`
    - Create `app/__init__.py`, `app/api/__init__.py`, `app/core/__init__.py`, `app/models/__init__.py`, `app/services/__init__.py`, `app/tasks/__init__.py`, `app/utils/__init__.py`
    - _Requirements: 8.1_

  - [x] 1.2 Define data models and enums
    - Create `app/models/job.py` with `JobState`, `JobStatus`, `PipelineStep`, `VideoInfo`, `ErrorDetail` classes
    - Create `app/models/schemas.py` with `TranslateRequest`, `TranslateResponse`, `JobStatusResponse`, `CancelResponse`, `ErrorResponse` Pydantic models
    - Create `app/models/pipeline.py` with `DownloadResult`, `TranscriptionSegment`, `TranscriptionResult`, `TranslatedSegment`, `TranslationResult`, `VocalIsolationResult`, `SynthesisResult`, `SegmentAudio` dataclasses
    - _Requirements: 7.1, 8.2_

  - [x] 1.3 Create configuration and settings module
    - Create `app/core/config.py` with `Settings` class using pydantic-settings (Redis URL, storage path, max concurrent jobs, file expiry hours, retry settings, background volume level)
    - Create `app/core/celery_app.py` with Celery application setup pointing to Redis broker
    - Create `.env.example` with all configurable environment variables
    - _Requirements: 7.2, 8.5_

- [ ] 2. Implement Video Downloader module
  - [x] 2.1 Implement URL validation and video download logic
    - Create `app/services/downloader.py` with `VideoDownloader` class
    - Implement `validate_url(url: str) -> bool` to check if URL belongs to douyin.com domain (including subdomains: www.douyin.com, v.douyin.com)
    - Implement `download(url: str, output_dir: Path) -> DownloadResult` using yt-dlp with MP4 format
    - Implement `get_video_info(url: str) -> VideoInfo` for metadata extraction (duration, filesize, resolution)
    - Implement retry logic with max 3 attempts and exponential backoff (2s/4s/8s) for network failures
    - Handle specific errors: 404 (video removed), network timeout, invalid URL
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [-] 2.2 Write property test for URL validation (Property 1)
    - **Property 1: URL Validation — Only Douyin URLs Accepted**
    - **Validates: Requirements 1.2**

  - [-] 2.3 Write property test for retry logic (Property 2)
    - **Property 2: Network Retry Logic Bounded at 3 Attempts**
    - **Validates: Requirements 1.4**

  - [-] 2.4 Write unit tests for video downloader
    - Test URL validation with valid/invalid URLs
    - Test download with mocked yt-dlp responses
    - Test error handling for removed videos and network failures
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 3. Implement Audio Extraction and Vocal Isolation modules
  - [x] 3.1 Implement audio extractor service
    - Create `app/services/audio_extractor.py` with `AudioExtractor` class
    - Implement `has_audio_track(video_path: Path) -> bool` using FFmpeg probe
    - Implement `extract(video_path: Path, output_dir: Path) -> Path` to extract audio as WAV using FFmpeg
    - Handle error case: video without audio track returns descriptive error
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 3.2 Implement vocal isolator service
    - Create `app/services/vocal_isolator.py` with `VocalIsolator` class
    - Implement `isolate(audio_path: Path, output_dir: Path) -> VocalIsolationResult` using Demucs
    - Return separated vocals and background music paths
    - Handle graceful degradation: if isolation fails, proceed with full audio
    - _Requirements: 2.4_

  - [x] 3.3 Write unit tests for audio extraction and vocal isolation
    - Test audio track detection (with/without audio)
    - Test extraction produces valid WAV file
    - Test vocal isolation output structure
    - Test error handling for invalid video files
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 4. Implement Speech Recognition module
  - [x] 4.1 Implement speech recognizer service
    - Create `app/services/speech_recognizer.py` with `SpeechRecognizer` class
    - Implement `recognize(audio_path: Path) -> TranscriptionResult` using Whisper large-v3
    - Ensure output includes timestamps (start/end) for each segment
    - Implement speaker diarization for multi-speaker detection (speaker labeling)
    - Handle empty audio or no recognizable speech with descriptive error
    - Ensure segments maintain temporal ordering (start < end, non-overlapping)
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 4.2 Write property test for transcription segment ordering (Property 3)
    - **Property 3: Transcription Segments Temporal Ordering**
    - **Validates: Requirements 3.2**

  - [x] 4.3 Write unit tests for speech recognizer
    - Test transcription output structure with timestamps
    - Test empty/silent audio handling
    - Test multi-speaker labeling
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 5. Implement Translation module
  - [x] 5.1 Implement translator service
    - Create `app/services/translator.py` with `Translator` class
    - Implement `translate(transcription: TranscriptionResult) -> TranslationResult` using Google Cloud Translation API v2
    - Preserve segment structure and timestamps from transcription
    - Implement Chinese text validation (detect Unicode range \u4e00-\u9fff)
    - Handle empty/non-Chinese text with error "Không có nội dung tiếng Trung để dịch"
    - Implement retry with exponential backoff for API failures
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 5.2 Write property test for translation segment preservation (Property 4)
    - **Property 4: Translation Preserves Segment Structure and Timestamps**
    - **Validates: Requirements 4.2**

  - [x] 5.3 Write property test for Chinese text validation (Property 5)
    - **Property 5: Non-Chinese or Empty Text Validation**
    - **Validates: Requirements 4.4**

  - [x] 5.4 Write unit tests for translator
    - Test translation preserves timestamps
    - Test Chinese text detection
    - Test error handling for empty text and API failures
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 6. Checkpoint - Core processing modules
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement Voice Synthesis module
  - [x] 7.1 Implement voice synthesizer service
    - Create `app/services/voice_synthesizer.py` with `VoiceSynthesizer` class
    - Implement `synthesize(translation: TranslationResult, output_dir: Path) -> SynthesisResult` using edge-tts
    - Implement `select_voice(speaker: str | None) -> str` to assign distinct Vietnamese voices per speaker
    - Implement speed adjustment: if TTS duration exceeds target duration, adjust speed (max 2x) to fit
    - Handle graceful degradation: if multi-voice fails, use single default voice
    - Combine individual segment audios into final Vietnamese audio file with correct timing
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 7.2 Write property test for TTS duration matching (Property 6)
    - **Property 6: TTS Duration Matches Target Segment Duration**
    - **Validates: Requirements 5.2, 5.3**

  - [x] 7.3 Write property test for speaker-voice mapping (Property 7)
    - **Property 7: Distinct Speakers Receive Distinct Voices**
    - **Validates: Requirements 5.4**

  - [x] 7.4 Write unit tests for voice synthesizer
    - Test voice selection per speaker
    - Test speed adjustment logic
    - Test segment audio timing
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 8. Implement Video Composer module
  - [x] 8.1 Implement video composer service
    - Create `app/services/video_composer.py` with `VideoComposer` class
    - Implement `compose(video_path, vietnamese_audio, background_audio, output_dir, background_volume=0.2) -> Path`
    - Use FFmpeg to mix Vietnamese voiceover with background music at reduced volume
    - Output MP4 with H.264 codec, preserving original video quality
    - Handle disk space errors and FFmpeg failures
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 8.2 Write unit tests for video composer
    - Test output format is MP4 H.264
    - Test background volume mixing
    - Test error handling for FFmpeg failures
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 9. Implement Pipeline Orchestrator and Celery tasks
  - [x] 9.1 Implement pipeline orchestrator
    - Create `app/services/pipeline.py` with `TranslationPipeline` class
    - Implement `execute(job_id: str, url: str) -> PipelineResult` orchestrating all 7 steps sequentially
    - Implement `resume(job_id: str, from_step: str) -> PipelineResult` for retry from failed step
    - Update job state in Redis after each step (progress_percent, current_step)
    - Implement cancellation check between each step
    - Preserve artifacts from completed steps on failure
    - _Requirements: 7.1, 7.3, 7.4_

  - [x] 9.2 Implement Celery task definitions
    - Create `app/tasks/translation_task.py` with `translate_video_task` Celery task
    - Wire pipeline execution to Celery task with proper error handling
    - Implement task revocation for cancellation support
    - Create `app/tasks/cleanup_task.py` with `cleanup_expired_jobs` periodic task (24h expiry)
    - Configure Celery Beat schedule for cleanup task
    - _Requirements: 7.2, 7.3, 7.5_

  - [x] 9.3 Write property test for pipeline step progression (Property 8)
    - **Property 8: Pipeline Step Progression is Monotonic**
    - **Validates: Requirements 7.1**

  - [x] 9.4 Write property test for cancellation (Property 9)
    - **Property 9: Cancellation From Any Active Step**
    - **Validates: Requirements 7.3**

  - [x] 9.5 Write property test for pipeline resume (Property 10)
    - **Property 10: Pipeline Resume Preserves Prior Artifacts**
    - **Validates: Requirements 6.5, 7.4**

  - [x] 9.6 Write unit tests for pipeline orchestrator
    - Test full pipeline execution flow with mocked services
    - Test resume from each step
    - Test cancellation at various steps
    - Test progress tracking
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 10. Checkpoint - Pipeline and modules complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implement API Layer and Rate Limiting
  - [x] 11.1 Implement FastAPI endpoints and rate limiting
    - Create `app/api/routes.py` with REST API endpoints:
      - `POST /api/v1/translate` — accept Douyin URL, validate, create job, return HTTP 202 with job_id
      - `GET /api/v1/jobs/{job_id}` — return job status, progress, download URL
      - `DELETE /api/v1/jobs/{job_id}` — cancel job, return cancelled status
    - Create `app/api/dependencies.py` with rate limiting dependency (max 5 concurrent jobs per IP)
    - Return HTTP 429 with Retry-After header when rate limit exceeded
    - Create `app/main.py` with FastAPI app setup, CORS, error handlers, router includes
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 11.2 Implement Redis job state management
    - Create `app/core/job_store.py` with `JobStore` class for Redis operations
    - Implement `create_job`, `get_job`, `update_job`, `cancel_job`, `count_active_jobs` methods
    - Implement job serialization/deserialization with Pydantic
    - Handle Redis connection errors gracefully (HTTP 503)
    - _Requirements: 7.2, 8.2, 8.4_

  - [x] 11.3 Write property test for API response fields (Property 11)
    - **Property 11: API Response Contains Required Fields**
    - **Validates: Requirements 8.2**

  - [x] 11.4 Write property test for rate limiting (Property 12)
    - **Property 12: Rate Limiting Enforces Maximum 5 Concurrent Jobs**
    - **Validates: Requirements 8.5**

  - [x] 11.5 Write unit tests for API endpoints
    - Test POST /translate returns 202 with job_id
    - Test GET /jobs/:id returns correct status
    - Test DELETE /jobs/:id cancels job
    - Test rate limiting returns 429
    - Test invalid URL returns 400
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 12. Integration wiring and final setup
  - [x] 12.1 Wire all components together and create entry point
    - Update `app/main.py` to include all routers and middleware
    - Create `app/core/dependencies.py` with dependency injection for all services
    - Create `docker-compose.yml` with Redis service for local development
    - Create `Dockerfile` for the application
    - Create `README.md` with setup instructions, API documentation, and usage examples
    - _Requirements: 8.1_

  - [x] 12.2 Write integration tests for full pipeline
    - Test end-to-end job creation and status tracking via API
    - Test file cleanup after expiry
    - Test error propagation from pipeline to API response
    - _Requirements: 7.1, 7.2, 7.5, 8.1, 8.2_

- [x] 13. Final checkpoint - Full integration
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- External services (yt-dlp, Whisper, Demucs, Google Translate, edge-tts) should be mocked in unit tests
- Integration tests require actual service access and should be run separately
- Hypothesis library is used for property-based testing with minimum 100 iterations per property

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["2.1", "3.1", "3.2"] },
    { "id": 3, "tasks": ["2.2", "2.3", "2.4", "3.3", "4.1"] },
    { "id": 4, "tasks": ["4.2", "4.3", "5.1"] },
    { "id": 5, "tasks": ["5.2", "5.3", "5.4", "7.1"] },
    { "id": 6, "tasks": ["7.2", "7.3", "7.4", "8.1"] },
    { "id": 7, "tasks": ["8.2", "9.1"] },
    { "id": 8, "tasks": ["9.2", "9.3", "9.4", "9.5", "9.6"] },
    { "id": 9, "tasks": ["11.1", "11.2"] },
    { "id": 10, "tasks": ["11.3", "11.4", "11.5", "12.1"] },
    { "id": 11, "tasks": ["12.2"] }
  ]
}
```
