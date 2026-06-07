# Requirements Document

## Introduction

Pipeline Preview & Confirm là tính năng cho phép pipeline dịch video Douyin tạm dừng tại các bước quan trọng để người dùng xem trước kết quả và xác nhận trước khi tiếp tục. Pipeline sẽ dừng tại 3 điểm: (1) sau khi nhận dạng tiếng Trung — cho phép xem và chỉnh sửa transcript, (2) sau khi dịch sang tiếng Việt — cho phép chỉnh sửa bản dịch cho chính xác, và (3) khi tạo giọng nói tiếng Việt — cho phép nghe thử và chọn giọng phù hợp.

## Glossary

- **Pipeline**: Dịch vụ điều phối (TranslationPipeline) xử lý tuần tự các bước dịch video từ Douyin
- **Job**: Một phiên dịch video, được theo dõi trạng thái qua JobState trong Redis
- **Checkpoint**: Điểm dừng trong pipeline nơi hệ thống chờ người dùng xác nhận trước khi tiếp tục
- **Transcription_Preview**: Bản xem trước kết quả nhận dạng giọng nói tiếng Trung (segments với timestamp)
- **Translation_Preview**: Bản xem trước kết quả dịch tiếng Việt (segments gốc và dịch)
- **Voice_Preview**: Bản xem trước các tùy chọn giọng nói tiếng Việt được tổng hợp
- **Job_Store**: Dịch vụ lưu trữ trạng thái job trong Redis
- **Voice_Option**: Một mẫu giọng nói TTS với tên, ID, và audio preview ngắn

## Requirements

### Requirement 1: Pipeline Pause at Checkpoints

**User Story:** As a user, I want the pipeline to pause at key steps, so that I can review and correct results before the pipeline continues processing.

#### Acceptance Criteria

1. WHEN the speech recognition step completes, THE Pipeline SHALL transition the job status to "awaiting_confirmation", store the transcription result in the job artifacts, and include the transcription text in the job status response for user review
2. WHEN the translation step completes, THE Pipeline SHALL transition the job status to "awaiting_confirmation", store the translation result in the job artifacts, and include the translated text in the job status response for user review
3. WHEN the voice synthesis preview step completes, THE Pipeline SHALL transition the job status to "awaiting_confirmation", store the voice preview options in the job artifacts, and include the available voice preview references in the job status response for user selection
4. WHILE a job is in "awaiting_confirmation" status, THE Pipeline SHALL not proceed to the next step until a user confirmation request is received for that job via the API
5. WHILE a job is in "awaiting_confirmation" status, THE Job_Store SHALL preserve all artifacts from previously completed steps until the job reaches a terminal status or the job expiration time elapses
6. WHEN the user submits a confirmation with corrections for a checkpoint step, THE Pipeline SHALL replace the stored result for that step with the user-provided corrected content and resume processing from the next step
7. IF a job remains in "awaiting_confirmation" status for longer than 24 hours without user action, THEN THE Pipeline SHALL transition the job status to "failed" with an error indicating confirmation timeout
8. WHEN the user submits a confirmation without corrections for a checkpoint step, THE Pipeline SHALL resume processing from the next step using the existing stored result

### Requirement 2: Transcription Preview and Edit

**User Story:** As a user, I want to preview the detected Chinese text and edit it if needed, so that subsequent translation is based on accurate source text.

#### Acceptance Criteria

1. WHEN the job reaches the transcription checkpoint, THE API SHALL return the transcription segments each containing a start timestamp in seconds, an end timestamp in seconds, the transcribed text, and a confidence score between 0.0 and 1.0
2. WHEN the user submits a confirmation without edits, THE Pipeline SHALL proceed to the translation step using the original transcription
3. WHEN the user submits a confirmation with edited segments, THE Pipeline SHALL update the stored transcription with the user edits and proceed to the translation step using the edited version
4. THE API SHALL accept partial edits where only specific segments are modified while others remain unchanged, identifying segments by their index position in the returned list
5. IF the user submits a segment with empty or whitespace-only text, THEN THE Pipeline SHALL exclude that segment from subsequent processing
6. IF the user submits an edit where any segment text exceeds 500 characters, THEN THE API SHALL reject the request with an error indicating the maximum segment length has been exceeded

### Requirement 3: Translation Preview and Edit

**User Story:** As a user, I want to preview and edit the Vietnamese translation, so that I can ensure accuracy before voice synthesis.

#### Acceptance Criteria

1. WHEN the job reaches the translation checkpoint, THE API SHALL return the translation segments each containing the segment index, start time, end time, original Chinese text, and translated Vietnamese text
2. WHEN the user submits a confirmation without edits, THE Pipeline SHALL proceed to voice synthesis using the original translation
3. WHEN the user submits a confirmation with edited translations, THE Pipeline SHALL validate that each edited segment's translated_text does not exceed 5000 characters, update the stored translation with the user edits, and proceed to voice synthesis using the edited version
4. WHEN the user submits partial edits referencing specific segment indices, THE API SHALL apply edits only to the referenced segments while preserving all other segments unchanged
5. IF the user submits a translated segment with empty or whitespace-only text, THEN THE Pipeline SHALL exclude that segment from voice synthesis
6. IF the user submits an edit referencing a segment index that does not exist in the stored translation, THEN THE API SHALL return an error indicating the invalid segment index and reject the entire edit submission

### Requirement 4: Voice Preview and Selection

**User Story:** As a user, I want to hear multiple Vietnamese voice options before the final video is composed, so that I can choose the most suitable voice.

#### Acceptance Criteria

1. WHEN the job reaches the voice preview checkpoint, THE Voice_Synthesizer SHALL generate audio preview samples of 5 to 15 seconds in duration using at least 3 different voice options
2. WHEN the job reaches the voice preview checkpoint, THE API SHALL return a list of Voice_Option items each containing a voice ID, voice name, and a URL to the preview audio file
3. WHEN the user selects a voice option, THE Pipeline SHALL proceed to full voice synthesis using the selected voice for all segments
4. THE Voice_Synthesizer SHALL select the longest translated text segment that does not exceed 15 seconds of estimated speech duration to generate preview samples
5. IF the user selects a voice that is no longer available, THEN THE API SHALL return an error indicating the voice is unavailable and provide the current list of available voices
6. IF the Voice_Synthesizer fails to generate a preview for one or more voice options, THEN THE API SHALL return only the successfully generated voice options, provided at least 2 options are available

### Requirement 5: Confirmation API Endpoints

**User Story:** As a user, I want API endpoints to submit my confirmations and edits, so that I can interact with the pipeline at each checkpoint.

#### Acceptance Criteria

1. THE API SHALL expose a POST endpoint for confirming the transcription checkpoint, accepting an optional list of edited segments where each segment contains a segment index and the revised text (max 5000 characters per segment)
2. THE API SHALL expose a POST endpoint for confirming the translation checkpoint, accepting an optional list of edited translation segments where each segment contains a segment index and the revised translated text (max 5000 characters per segment)
3. THE API SHALL expose a POST endpoint for confirming the voice selection, accepting the selected voice ID as a non-empty string
4. IF a confirmation request is submitted for a job that is not in "awaiting_confirmation" status, THEN THE API SHALL return an error with HTTP 409 status indicating the job's current status and that confirmation is not expected
5. IF a confirmation request references a non-existent job, THEN THE API SHALL return an error with HTTP 404 status
6. WHEN a valid confirmation is received, THE API SHALL return the updated job status and the next pipeline step, and enqueue pipeline resumption within 5 seconds of the response
7. IF a confirmation request targets a checkpoint type that does not match the job's current checkpoint, THEN THE API SHALL return an error with HTTP 409 status indicating which checkpoint the job is currently awaiting
8. IF a confirmation request is received while another confirmation for the same job is already being processed, THEN THE API SHALL return an error with HTTP 409 status indicating the confirmation is already in progress

### Requirement 6: Job Status Reporting for Checkpoints

**User Story:** As a user, I want the job status API to clearly indicate when a job is waiting for my input, so that I know when to review and confirm.

#### Acceptance Criteria

1. WHILE a job is in "awaiting_confirmation" status, THE Job_Status_API SHALL include a "checkpoint_type" field with the value "transcription", "translation", or "voice_selection" corresponding to the active checkpoint
2. WHILE a job is in "awaiting_confirmation" status at the transcription checkpoint, THE Job_Status_API SHALL include preview data containing the transcription segments with start time, end time, text, and confidence score for each segment
3. WHILE a job is in "awaiting_confirmation" status at the translation checkpoint, THE Job_Status_API SHALL include preview data containing the translation segments with start time, end time, original Chinese text, and translated Vietnamese text for each segment
4. WHILE a job is in "awaiting_confirmation" status at the voice_selection checkpoint, THE Job_Status_API SHALL include preview data containing a list of Voice_Option items each with voice ID, voice name, and preview audio URL
5. WHEN a job transitions from "awaiting_confirmation" to "processing", THE Job_Status_API SHALL return the updated status as "processing" and the next pipeline step name in the subsequent GET response
6. IF a job is not in "awaiting_confirmation" status, THEN THE Job_Status_API SHALL omit the "checkpoint_type" and preview data fields from the response or return them as null

### Requirement 7: Checkpoint Timeout Handling

**User Story:** As a user, I want the system to handle cases where I don't respond to a checkpoint promptly, so that jobs don't remain stuck indefinitely.

#### Acceptance Criteria

1. WHILE a job is in "awaiting_confirmation" status, THE Pipeline SHALL keep the job available for user confirmation for a maximum duration of 24 hours from the most recent expiration timer start
2. IF a job remains in "awaiting_confirmation" status for more than 24 hours since the last expiration timer reset, THEN THE Pipeline SHALL transition the job status to "expired", delete the job's working directory and all associated artifact files, and remove the job from the active jobs set in the Job_Store
3. WHEN a user queries the status of a job that is in "awaiting_confirmation" status, THE Job_Store SHALL reset the job expiration timer to 24 hours from the time of the query
4. IF the user submits a confirmation for an expired job, THEN THE API SHALL return an error with HTTP 410 status indicating the job has expired
5. THE Pipeline SHALL check for expired "awaiting_confirmation" jobs at intervals no greater than 60 seconds
