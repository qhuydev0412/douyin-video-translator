#!/bin/sh
celery -A app.core.celery_app worker --loglevel=info --concurrency=2 &
python web_ui.py
