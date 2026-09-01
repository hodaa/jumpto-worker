"""Celery application configuration for the JumpTo worker."""

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "jumpto-worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks.transcription"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.job_timeout_seconds,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)
