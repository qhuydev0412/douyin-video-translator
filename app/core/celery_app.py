"""Celery application setup."""

from celery import Celery
from celery.schedules import crontab

from app.core.config import settings

celery_app = Celery(
    "douyin_video_translator",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.translation_task",
        "app.tasks.cleanup_task",
        "app.tasks.expiry_task",
        "app.tasks.resume_task",
    ],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Reliability
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Time limits (seconds)
    task_time_limit=1800,  # 30 minutes hard limit
    task_soft_time_limit=1500,  # 25 minutes soft limit
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    # Result settings
    result_expires=86400,  # 24 hours
    # Track task state
    task_track_started=True,
)

# Celery Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    "cleanup-expired-jobs-hourly": {
        "task": "cleanup_expired_jobs",
        "schedule": crontab(minute=0),  # Every hour at :00
    },
    "check-checkpoint-expiry": {
        "task": "check_checkpoint_expiry",
        "schedule": 30.0,  # Every 30 seconds
    },
}
