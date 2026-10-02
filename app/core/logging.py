"""Structured logging configuration for the JumpTo worker.

Owns only *how* a record renders: processors, renderer and levels. What
identifies a job inside a record is a separate concern, owned by
``app.core.job_context``, so that adding an identifier is not a change to the
render format and vice versa.
"""

import logging
import socket

import structlog
from structlog.stdlib import LoggerFactory

from app.core.config import get_settings

# Resolved once: the worker's hostname is constant for the process, and
# ``socket.gethostname()`` can block on some container/network setups.
_HOSTNAME = socket.gethostname()


def add_hostname(_logger: object, _method: str, event_dict: dict) -> dict:
    """Add the worker's hostname to a log record.

    A record attribute rather than job context: the hostname is constant for the
    process and belongs on every record, including those emitted outside any job
    (worker startup, broker reconnects). Sentry's ``server_name`` only appears
    on error *events*, so without this the log record itself cannot say which
    worker produced it.
    """
    event_dict["hostname"] = _HOSTNAME
    return event_dict


def configure_logging() -> None:
    """Configure structlog for the worker."""
    settings = get_settings()

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        add_hostname,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.is_development:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # structlog's stdlib backend honours the stdlib logger level, which
    # defaults to WARNING — that would silently drop INFO job records (e.g.
    # which video is being processed) before the Sentry logs integration can
    # capture them. Raise the app loggers and add a root handler at INFO.
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("app").setLevel(logging.INFO)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a logger instance with the given name."""
    return structlog.get_logger(name)
