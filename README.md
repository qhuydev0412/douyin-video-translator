# Douyin Video Translator

Dịch video Douyin từ tiếng Trung sang tiếng Việt — tự động tải video, tách giọng nói, nhận dạng, dịch, và ghép giọng thuyết minh tiếng Việt vào video gốc.

## Tổng quan

Hệ thống nhận link Douyin qua REST API, xử lý video qua pipeline 7 bước:

1. **Tải video** — yt-dlp
2. **Tách âm thanh** — FFmpeg
3. **Tách giọng nói** — Demucs (Meta Research)
4. **Nhận dạng giọng nói** — OpenAI Whisper large-v3
5. **Dịch** — Google Cloud Translation API
6. **Tổng hợp giọng Việt** — edge-tts (Microsoft)
7. **Ghép video** — FFmpeg

## Kiến trúc

```
┌─────────┐       ┌──────────────┐       ┌───────────┐
│  Client │──────▶│  FastAPI API │──────▶│   Redis   │
└─────────┘       └──────────────┘       └─────┬─────┘
                                               │
                                   ┌───────────▼──────────┐
                                   │    Celery Worker     │
                                   │  (Translation        │
                                   │   Pipeline)          │
                                   └───────────┬──────────┘
                                               │
                                   ┌───────────▼──────────┐
                                   │  Local File Storage  │
                                   │  (storage/jobs/)     │
                                   └──────────────────────┘
```

## Cài đặt

### Docker (khuyên dùng)

```bash
# Clone project
git clone <repository-url>
cd douyin-video-translator

# Copy file cấu hình
cp .env.example .env

# Khởi chạy tất cả services
docker compose up -d

# Xem logs
docker compose logs -f app
```

### Cài đặt thủ công

**Yêu cầu:**
- Python 3.11+
- FFmpeg
- Redis

```bash
# Cài đặt dependencies
pip install -e .

# Cài thêm dev/test dependencies (tùy chọn)
pip install -e ".[dev,test]"

# Copy file cấu hình
cp .env.example .env

# Chạy Redis (cần cài sẵn hoặc dùng Docker)
docker run -d -p 6379:6379 redis:7-alpine

# Chạy FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Chạy Celery worker (terminal khác)
celery -A app.core.celery_app worker --loglevel=info

# Chạy Celery beat (terminal khác, cho cleanup task)
celery -A app.core.celery_app beat --loglevel=info
```

## API Documentation

Base URL: `http://localhost:8000/api/v1`

### 1. Tạo yêu cầu dịch video

```
POST /api/v1/translate
```

**Request body:**
```json
{
  "url": "https://www.douyin.com/video/1234567890"
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Đã tiếp nhận yêu cầu dịch video"
}
```

**Errors:**
- `400` — URL không hợp lệ
- `429` — Vượt quá giới hạn (tối đa 5 jobs đồng thời)

### 2. Kiểm tra trạng thái job

```
GET /api/v1/jobs/{job_id}
```

**Response (200 OK):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "current_step": "translating",
  "progress_percent": 60,
  "video_info": {
    "duration_seconds": 45.2,
    "file_size_bytes": 5242880,
    "resolution": "1080x1920",
    "title": "Video title"
  },
  "download_url": null,
  "error": null,
  "created_at": "2024-01-15T10:30:00Z",
  "expires_at": null
}
```

**Trạng thái:** `queued` → `processing` → `completed` | `failed` | `cancelled`

**Các bước pipeline:** `downloading` → `extracting_audio` → `isolating_vocals` → `recognizing_speech` → `translating` → `synthesizing_voice` → `composing_video`

### 3. Hủy job

```
DELETE /api/v1/jobs/{job_id}
```

**Response (200 OK):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "cancelled"
}
```

## Biến môi trường

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis cho lưu trạng thái job |
| `CELERY_BROKER_URL` | `redis://localhost:6379/1` | Redis broker cho Celery |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/1` | Redis backend cho kết quả Celery |
| `STORAGE_PATH` | `storage/jobs` | Thư mục lưu file tạm |
| `MAX_CONCURRENT_JOBS` | `5` | Số job đồng thời tối đa/IP |
| `FILE_EXPIRY_HOURS` | `24` | Giờ trước khi xóa file kết quả |
| `MAX_RETRY_ATTEMPTS` | `3` | Số lần thử lại khi lỗi mạng |
| `RETRY_BACKOFF_BASE` | `2` | Hệ số backoff (delay = base^attempt giây) |
| `BACKGROUND_VOLUME` | `0.2` | Âm lượng nhạc nền (0.0–1.0) |

## Chạy Tests

```bash
# Cài test dependencies
pip install -e ".[test]"

# Chạy tất cả unit tests
pytest tests/unit/

# Chạy property-based tests
pytest tests/property/

# Chạy với coverage
pytest --cov=app tests/
```

## Cấu trúc thư mục

```
douyin-video-translator/
├── app/
│   ├── api/            # REST API routes & dependencies
│   ├── core/           # Config, Celery, DI
│   ├── models/         # Data models (Pydantic, dataclasses)
│   ├── services/       # Business logic (pipeline steps)
│   ├── tasks/          # Celery task definitions
│   └── utils/          # Shared utilities
├── storage/jobs/       # Job working directories (auto-cleaned)
├── tests/
│   ├── unit/           # Unit tests
│   ├── property/       # Property-based tests (Hypothesis)
│   └── integration/    # Integration tests
├── docker-compose.yml  # Docker dev environment
├── Dockerfile          # Application container
├── pyproject.toml      # Project config & dependencies
└── .env.example        # Environment variable template
```
