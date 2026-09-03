"""Celery application configuration for the JumpTo worker."""

import ssl

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "jumpto-worker",
    broker=settings.redis_url,
    include=["app.tasks.transcription"],
)

_redis_url = settings.redis_url

if _redis_url.startswith("rediss://"):
    separator = "&" if "?" in _redis_url else "?"
    _redis_url = f"{_redis_url}{separator}ssl_cert_reqs=CERT_REQUIRED"

celery_app = Celery(
    "jumpto",
    broker=_redis_url,
    backend=_redis_url,
)

if _redis_url.startswith("rediss://"):
    celery_app.conf.broker_use_ssl = {
        "ssl_cert_reqs": ssl.CERT_REQUIRED,
    }

    celery_app.conf.result_backend_transport_options = {
        "ssl_cert_reqs": ssl.CERT_REQUIRED,
    }


celery_app.conf.broker_connection_retry_on_startup = True

celery_app.conf.result_backend = None

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=None,
    task_time_limit=settings.job_timeout_seconds,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

celery_app.conf.worker_concurrency = settings.celery_worker_concurrency
