"""Sentry error tracking for the JumpTo worker."""

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def init_sentry() -> None:
    """Initialise Sentry from settings; a no-op when no DSN is configured.

    Runs once at worker startup. Settings leave the DSN empty by default, so
    Sentry stays off unless ``SENTRY_DSN`` is set. Errors in Celery tasks are
    captured by the Celery integration; performance tracing is gated behind
    ``SENTRY_TRACES_SAMPLE_RATE`` (default 0 = off).
    """
    settings = get_settings()
    if not settings.sentry_dsn:
        logger.debug("Sentry disabled: SENTRY_DSN not set")
        return

    import sentry_sdk
    from sentry_sdk.integrations.celery import CeleryIntegration

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.environment,
        traces_sample_rate=settings.sentry_traces_sample_rate,
        integrations=[CeleryIntegration(propagate_traces=True)],
    )
    logger.info(
        "Sentry initialised",
        environment=settings.environment,
        traces_sample_rate=settings.sentry_traces_sample_rate,
    )
