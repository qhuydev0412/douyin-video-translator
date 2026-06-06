# Deploy Douyin Video Translator (Web UI)

## Quick Deploy

```bash
# Clone project lên server
git clone <repo-url> douyin-video-translator
cd douyin-video-translator

# Build và chạy (CPU mode)
docker compose up -d

# Xem logs
docker compose logs -f web
```

Mở browser: `http://your-server-ip:8080`

## GPU Mode (NVIDIA)

Nếu server có NVIDIA GPU (khuyên dùng cho Whisper):

```bash
# Cài nvidia-container-toolkit trước
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

docker compose --profile gpu up -d
```

## Google Cloud Translation API

Cần credentials cho bước dịch. 2 cách:

### Cách 1: Service Account JSON

```bash
mkdir credentials
# Copy file service-account.json vào credentials/gcloud.json
cp your-service-account.json credentials/gcloud.json
```

Uncomment volume mount trong `docker-compose.yml`:
```yaml
volumes:
  - ./credentials:/app/credentials:ro
```

### Cách 2: API Key (đơn giản hơn)

Sửa `app/services/translator.py` — thay `translate.Client()` bằng:
```python
translate.Client(target_language="vi")  # with API key in env
```

Và thêm env var:
```yaml
environment:
  - GOOGLE_API_KEY=your-api-key-here
```

## Cấu hình

| Env Var | Default | Mô tả |
|---------|---------|-------|
| `GOOGLE_APPLICATION_CREDENTIALS` | — | Path tới GCloud service account JSON |
| `WHISPER_MODEL` | base | Trong Dockerfile, thay đổi model pre-download |

## Whisper Model Choice

| Model | Size | RAM | Speed | Accuracy (Chinese) |
|-------|------|-----|-------|---------------------|
| base | 150MB | ~1GB | ⚡ Fast | Trung bình |
| small | 500MB | ~2GB | 🔶 OK | Khá |
| medium | 1.5GB | ~4GB | 🐢 Slow | Tốt |
| large-v3 | 3GB | ~8GB | 🐌 Very slow (CPU) | Rất tốt |

> 💡 Với CPU: dùng `base` hoặc `small`. Với GPU: dùng `large-v3`.

Để thay đổi model mặc định, sửa `Dockerfile.webui`:
```dockerfile
RUN python -c "import whisper; whisper.load_model('small')"
```

## Server Requirements

**Minimum (CPU, model base):**
- 4 vCPU, 4GB RAM, 20GB disk

**Recommended (GPU, model large-v3):**
- 4 vCPU, 16GB RAM, 50GB disk, NVIDIA GPU 8GB VRAM

## Rebuild

```bash
docker compose build --no-cache
docker compose up -d
```

## Stop

```bash
docker compose down
```
