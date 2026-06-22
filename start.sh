#!/bin/sh
echo "=== Douyin Video Translator ==="
echo "REDIS_URL=${REDIS_URL}"
echo "CELERY_BROKER_URL=${CELERY_BROKER_URL}"
echo "Starting Celery worker..."
celery -A app.core.celery_app:celery_app worker --loglevel=info --concurrency=2 -B &
CELERY_PID=$!
sleep 2

# Check if celery is still running
if kill -0 $CELERY_PID 2>/dev/null; then
  echo "Celery worker started (PID=$CELERY_PID)"
else
  echo "ERROR: Celery worker crashed!"
fi

echo "Starting web server..."
python web_ui.py
