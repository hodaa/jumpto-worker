"""Sentry error tracking and structured logs for the JumpTo worker."""

import logging

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def init_sentry() -> None:
    """Initialise Sentry from settings; a no-op when no DSN is configured.

    Runs once at worker startup. Settings leave the DSN empty by default, so
    Sentry stays off unless ``SENTRY_DSN`` is set. With it set:

    - Console/structlog records at INFO+ are captured as queryable Sentry
      logs; structlog keyword fields (e.g. ``youtube_url``, ``job_id``) are
      promoted to top-level searchable attributes.
    - ERROR-level records additionally raise Sentry error events.
    - Celery task errors are captured by the Celery integration; performance
      tracing is gated behind ``SENTRY_TRACES_SAMPLE_RATE`` (default 0 = off).
    """
    settings = get_settings()
    if not settings.sentry_dsn:
        logger.debug("Sentry disabled: SENTRY_DSN not set")
        return

    import sentry_sdk
    from sentry_sdk.integrations.celery import CeleryIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.environment,
        traces_sample_rate=settings.sentry_traces_sample_rate,
        integrations=[
            CeleryIntegration(propagate_traces=True),
            LoggingIntegration(
                capture_sentry_logs=True,
                sentry_logs_level=logging.INFO,
                level=logging.INFO,
                event_level=logging.ERROR,
            ),
        ],
    )
    logger.info(
        "Sentry initialised",
        environment=settings.environment,
        traces_sample_rate=settings.sentry_traces_sample_rate,
    )
