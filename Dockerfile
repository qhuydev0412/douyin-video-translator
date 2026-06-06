FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch CPU (separate step — large download, good to cache)
RUN pip install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install whisper and demucs (depends on torch)
RUN pip install --no-cache-dir openai-whisper demucs torchcodec

# Install remaining dependencies
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    pydantic \
    pydantic-settings \
    httpx \
    deep-translator \
    edge-tts \
    yt-dlp \
    "celery[redis]" \
    redis

# Copy application code
COPY app/ ./app/
COPY web_ui.py .

# Create directories
RUN mkdir -p storage/jobs

# Whisper model will be downloaded on first use
# To pre-download, uncomment: RUN python -c "import whisper; whisper.load_model('base')"

VOLUME /app/storage
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8080\")}/')" || exit 1

CMD ["python", "web_ui.py"]
