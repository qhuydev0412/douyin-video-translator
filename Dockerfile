FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir python-multipart

# Copy application code
COPY app/ ./app/
COPY web_ui.py .

# Create directories
RUN mkdir -p storage/jobs /root/.cache/whisper /root/.cache/torch/hub

# Pre-download Whisper base model (~150MB)
# Change to 'small', 'medium', or 'large-v3' if needed
RUN python -c "import whisper; whisper.load_model('base')"

VOLUME /app/storage
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/')" || exit 1

# Default: Web UI mode (upload video)
# Override with: command: uvicorn app.main:app --host 0.0.0.0 --port 8000
CMD ["python", "web_ui.py"]
