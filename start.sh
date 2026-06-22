#!/bin/bash
set -e

echo "=========================================="
echo "  🎬 Douyin Video Translator - Starting"
echo "=========================================="

# Start Redis in background (embedded, for single-container deploy)
echo "▶ Starting Redis..."
redis-server --daemonize yes --port 6379 --maxmemory 128mb --maxmemory-policy allkeys-lru

# Wait for Redis to be ready
until redis-cli ping > /dev/null 2>&1; do
  sleep 0.2
done
echo "✅ Redis ready"

# Start Celery worker in background
echo "▶ Starting Celery worker..."
celery -A app.core.celery_app:celery_app worker \
  --loglevel=info \
  --concurrency=2 \
  --pool=prefork \
  --without-heartbeat \
  --without-mingle \
  --without-gossip &
CELERY_PID=$!

# Start Celery Beat in background (for periodic tasks)
echo "▶ Starting Celery Beat..."
celery -A app.core.celery_app:celery_app beat \
  --loglevel=info &
BEAT_PID=$!

# Give worker a moment to connect
sleep 2
echo "✅ Celery worker ready"

# Start FastAPI (foreground)
echo "▶ Starting FastAPI on port ${PORT:-8080}..."
exec python web_ui.py
