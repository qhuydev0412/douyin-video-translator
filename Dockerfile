FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all source first (needed for pip install)
COPY pyproject.toml .
COPY app/ ./app/
COPY web_ui.py .

# Install Python dependencies
RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir python-multipart deep-translator

# Create directories
RUN mkdir -p storage/jobs /root/.cache/whisper /root/.cache/torch/hub

# Pre-download Whisper base model (~150MB)
RUN python -c "import whisper; whisper.load_model('base')"

VOLUME /app/storage
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/')" || exit 1

CMD ["python", "web_ui.py"]
