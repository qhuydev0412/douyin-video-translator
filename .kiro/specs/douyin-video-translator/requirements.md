# Requirements Document

## Introduction

Công cụ Douyin Video Translator cho phép người dùng tải video từ Douyin (TikTok Trung Quốc), tách lời nói/giọng nói từ video, dịch nội dung lời nói sang tiếng Việt, và ghép lại giọng đọc tiếng Việt vào video gốc. Công cụ này hỗ trợ người dùng Việt Nam tiếp cận nội dung Douyin mà không cần biết tiếng Trung.

Đây là một project độc lập (standalone), không phụ thuộc vào các project khác trong workspace.

## Glossary

- **Hệ_thống (System)**: Ứng dụng Douyin Video Translator hoàn chỉnh bao gồm backend service và các thành phần xử lý
- **Trình_tải_video (Video_Downloader)**: Module chịu trách nhiệm tải video từ link Douyin
- **Trình_tách_âm (Audio_Extractor)**: Module chịu trách nhiệm tách âm thanh/giọng nói ra khỏi video
- **Trình_nhận_dạng (Speech_Recognizer)**: Module chịu trách nhiệm chuyển đổi âm thanh giọng nói thành văn bản (Speech-to-Text)
- **Trình_dịch (Translator)**: Module chịu trách nhiệm dịch văn bản từ tiếng Trung sang tiếng Việt
- **Trình_tổng_hợp_giọng (Voice_Synthesizer)**: Module chịu trách nhiệm tạo giọng đọc tiếng Việt từ văn bản đã dịch (Text-to-Speech)
- **Trình_ghép_video (Video_Composer)**: Module chịu trách nhiệm ghép giọng đọc tiếng Việt vào video gốc
- **Douyin_URL**: Đường link chia sẻ hợp lệ từ ứng dụng Douyin hoặc website douyin.com
- **Video_gốc**: File video được tải từ Douyin trước khi xử lý
- **Video_đầu_ra**: File video cuối cùng đã được ghép giọng đọc tiếng Việt

## Requirements

### Requirement 1: Tải video từ Douyin

**User Story:** As a người dùng, I want to tải video từ link Douyin, so that tôi có thể xử lý và dịch nội dung video đó sang tiếng Việt.

#### Acceptance Criteria

1. WHEN người dùng cung cấp một Douyin_URL hợp lệ, THE Trình_tải_video SHALL tải video gốc về hệ thống và lưu dưới định dạng MP4
2. WHEN người dùng cung cấp một URL không thuộc miền douyin.com, THE Trình_tải_video SHALL trả về thông báo lỗi "URL không hợp lệ, vui lòng cung cấp link Douyin"
3. IF video tại Douyin_URL không tồn tại hoặc đã bị xóa, THEN THE Trình_tải_video SHALL trả về thông báo lỗi "Video không tồn tại hoặc đã bị xóa"
4. IF kết nối mạng bị gián đoạn trong quá trình tải, THEN THE Trình_tải_video SHALL thử lại tối đa 3 lần trước khi báo lỗi
5. WHEN video được tải thành công, THE Trình_tải_video SHALL trả về thông tin video bao gồm thời lượng, kích thước file và độ phân giải

### Requirement 2: Tách âm thanh từ video

**User Story:** As a người dùng, I want to tách phần giọng nói ra khỏi video đã tải, so that có thể nhận dạng và dịch nội dung lời nói.

#### Acceptance Criteria

1. WHEN Video_gốc được tải thành công, THE Trình_tách_âm SHALL tách track âm thanh ra khỏi video và lưu dưới định dạng WAV
2. THE Trình_tách_âm SHALL giữ nguyên chất lượng âm thanh gốc trong quá trình tách
3. IF Video_gốc không chứa track âm thanh, THEN THE Trình_tách_âm SHALL trả về thông báo lỗi "Video không có âm thanh"
4. WHEN track âm thanh được tách thành công, THE Trình_tách_âm SHALL loại bỏ nhạc nền và giữ lại phần giọng nói chính (vocal isolation)

### Requirement 3: Nhận dạng giọng nói thành văn bản

**User Story:** As a người dùng, I want to chuyển đổi giọng nói tiếng Trung trong video thành văn bản, so that có thể dịch nội dung sang tiếng Việt.

#### Acceptance Criteria

1. WHEN file âm thanh giọng nói được cung cấp, THE Trình_nhận_dạng SHALL chuyển đổi giọng nói tiếng Trung thành văn bản với độ chính xác tối thiểu 85%
2. THE Trình_nhận_dạng SHALL giữ nguyên thông tin timestamp (mốc thời gian) cho từng câu hoặc đoạn lời nói
3. IF file âm thanh không chứa giọng nói nhận dạng được, THEN THE Trình_nhận_dạng SHALL trả về thông báo lỗi "Không nhận dạng được giọng nói trong file âm thanh"
4. WHEN giọng nói chứa nhiều người nói, THE Trình_nhận_dạng SHALL phân biệt các đoạn lời của từng người nói riêng biệt

### Requirement 4: Dịch văn bản sang tiếng Việt

**User Story:** As a người dùng, I want to dịch nội dung văn bản tiếng Trung sang tiếng Việt, so that tôi có thể hiểu nội dung video.

#### Acceptance Criteria

1. WHEN văn bản tiếng Trung được cung cấp, THE Trình_dịch SHALL dịch toàn bộ văn bản sang tiếng Việt với ngữ nghĩa tự nhiên
2. THE Trình_dịch SHALL giữ nguyên cấu trúc câu và thông tin timestamp từ bước nhận dạng
3. IF văn bản chứa thuật ngữ chuyên ngành hoặc thành ngữ, THEN THE Trình_dịch SHALL dịch theo ngữ cảnh phù hợp thay vì dịch từng từ
4. WHEN văn bản đầu vào rỗng hoặc không chứa nội dung tiếng Trung, THE Trình_dịch SHALL trả về thông báo lỗi "Không có nội dung tiếng Trung để dịch"

### Requirement 5: Tổng hợp giọng đọc tiếng Việt

**User Story:** As a người dùng, I want to tạo giọng đọc tiếng Việt từ văn bản đã dịch, so that có thể nghe nội dung bằng tiếng Việt.

#### Acceptance Criteria

1. WHEN văn bản tiếng Việt đã dịch được cung cấp, THE Trình_tổng_hợp_giọng SHALL tạo file âm thanh giọng đọc tiếng Việt tự nhiên
2. THE Trình_tổng_hợp_giọng SHALL đồng bộ thời lượng giọng đọc tiếng Việt với timestamp của từng đoạn lời gốc
3. IF thời lượng giọng đọc tiếng Việt vượt quá thời lượng đoạn gốc, THEN THE Trình_tổng_hợp_giọng SHALL điều chỉnh tốc độ đọc để khớp với thời lượng gốc mà không làm biến dạng giọng
4. WHEN có nhiều người nói trong video gốc, THE Trình_tổng_hợp_giọng SHALL sử dụng giọng đọc khác nhau cho từng người nói

### Requirement 6: Ghép giọng đọc vào video

**User Story:** As a người dùng, I want to ghép giọng đọc tiếng Việt vào video gốc, so that tôi có thể xem video với lời thuyết minh tiếng Việt.

#### Acceptance Criteria

1. WHEN giọng đọc tiếng Việt được tạo thành công, THE Trình_ghép_video SHALL thay thế track âm thanh gốc bằng giọng đọc tiếng Việt trong video
2. THE Trình_ghép_video SHALL giữ nguyên chất lượng hình ảnh của video gốc trong quá trình ghép
3. THE Trình_ghép_video SHALL giữ lại nhạc nền gốc ở mức âm lượng thấp phía sau giọng đọc tiếng Việt
4. WHEN quá trình ghép hoàn tất, THE Trình_ghép_video SHALL xuất Video_đầu_ra ở định dạng MP4 với codec H.264
5. IF quá trình ghép gặp lỗi, THEN THE Trình_ghép_video SHALL lưu lại tiến trình và cho phép thử lại từ bước ghép mà không cần xử lý lại toàn bộ

### Requirement 7: Quản lý quy trình xử lý

**User Story:** As a người dùng, I want to theo dõi tiến trình xử lý video, so that tôi biết trạng thái hiện tại và thời gian chờ dự kiến.

#### Acceptance Criteria

1. WHEN một yêu cầu xử lý video được tạo, THE Hệ_thống SHALL hiển thị tiến trình xử lý theo từng bước (tải video, tách âm, nhận dạng, dịch, tổng hợp giọng, ghép video)
2. THE Hệ_thống SHALL cập nhật trạng thái tiến trình theo thời gian thực cho người dùng
3. WHILE quá trình xử lý đang diễn ra, THE Hệ_thống SHALL cho phép người dùng hủy yêu cầu và giải phóng tài nguyên
4. IF bất kỳ bước nào trong quy trình thất bại, THEN THE Hệ_thống SHALL thông báo rõ bước thất bại và cho phép thử lại từ bước đó
5. WHEN Video_đầu_ra được tạo thành công, THE Hệ_thống SHALL cung cấp link tải video cho người dùng trong vòng 24 giờ trước khi xóa

### Requirement 8: API Integration

**User Story:** As a nhà phát triển, I want to có REST API rõ ràng cho tool dịch video, so that có thể tích hợp hoặc mở rộng dễ dàng trong tương lai.

#### Acceptance Criteria

1. THE Hệ_thống SHALL cung cấp REST API endpoint để nhận yêu cầu dịch video với đầu vào là Douyin_URL
2. THE Hệ_thống SHALL trả về response theo chuẩn JSON bao gồm trạng thái xử lý, ID yêu cầu, và link tải kết quả
3. WHEN một yêu cầu mới được gửi, THE Hệ_thống SHALL trả về HTTP 202 (Accepted) kèm ID yêu cầu để theo dõi
4. THE Hệ_thống SHALL cung cấp endpoint kiểm tra trạng thái xử lý theo ID yêu cầu
5. IF người dùng gửi quá 5 yêu cầu đồng thời, THEN THE Hệ_thống SHALL trả về HTTP 429 (Too Many Requests) với thông tin thời gian chờ
