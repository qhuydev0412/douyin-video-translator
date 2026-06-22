FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pin PyTorch CPU + torchaudio FIRST (prevents torchcodec being pulled)
RUN pip install --no-cache-dir \
    "torch==2.5.1" "torchaudio==2.5.1" \
    --index-url https://download.pytorch.org/whl/cpu

# Now install everything else in one shot (torchaudio already pinned above)
RUN pip install --no-cache-dir \
    openai-whisper \
    "demucs==4.0.1" \
    soundfile \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    python-dotenv \
    pydantic \
    pydantic-settings \
    httpx \
    deep-translator \
    openai \
    edge-tts \
    yt-dlp \
    "celery[redis]" \
    redis

COPY app/ ./app/
COPY web_ui.py .
COPY start.sh .
RUN chmod +x start.sh

RUN mkdir -p storage/jobs

# Pre-download models
RUN python -c "import whisper; whisper.load_model('base')"
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs')"

VOLUME /app/storage
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8080\")}/')" || exit 1

CMD ["./start.sh"]
