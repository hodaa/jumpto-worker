"""Celery application configuration for the JumpTo worker."""

import ssl

from celery import Celery

from app.core.config import get_settings

settings = get_settings()

broker_url = settings.broker_url

# Redis over TLS: kombu reads the query param and needs broker_use_ssl too.
if broker_url.startswith("rediss://"):
    separator = "&" if "?" in broker_url else "?"
    broker_url = f"{broker_url}{separator}ssl_cert_reqs=CERT_REQUIRED"

celery_app = Celery("jumpto", broker=broker_url)

if broker_url.startswith("rediss://"):
    celery_app.conf.broker_use_ssl = {
        "ssl_cert_reqs": ssl.CERT_REQUIRED,
    }
    celery_app.conf.result_backend_transport_options = {
        "ssl_cert_reqs": ssl.CERT_REQUIRED,
    }
elif broker_url.startswith("amqps://"):
    # The amqp/pyamqp transport expects amqp-style ssl options (cert_reqs,
    # not redis's ssl_cert_reqs) and loads the system CA store when none is
    # given, so a plain cert_reqs verifies against trusted CAs.
    celery_app.conf.broker_use_ssl = {
        "cert_reqs": ssl.CERT_REQUIRED,
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
    # Let the application timeout run its failure reporting and cleanup before
    # Celery sends the hard-kill signal to the child process.
    task_soft_time_limit=settings.job_timeout_seconds + 5,
    task_time_limit=settings.job_timeout_seconds + settings.task_timeout_grace_seconds,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

celery_app.conf.worker_concurrency = settings.celery_worker_concurrency
