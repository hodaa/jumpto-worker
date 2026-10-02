"""Celery application configuration for the JumpTo worker."""

import ssl

from celery import Celery
from celery.signals import task_postrun, task_prerun

from app.core import job_context
from app.core.config import get_settings
from app.core.job_context import JobContext
from app.core.logging import configure_logging
from app.core.sentry import init_sentry
from app.core.timeouts import task_hard_time_limit_seconds, task_soft_time_limit_seconds

settings = get_settings()
configure_logging()
init_sentry()

broker_url = settings.broker_url

# Redis over TLS: kombu reads the query param and needs broker_use_ssl too.
if broker_url.startswith("rediss://"):
    separator = "&" if "?" in broker_url else "?"
    broker_url = f"{broker_url}{separator}ssl_cert_reqs=CERT_REQUIRED"

celery_app = Celery(settings.celery_app_name, broker=broker_url)

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
    # Let the pipeline timeout run its failure reporting and cleanup before
    # Celery sends the soft and then the hard-kill signal to the child process.
    # Ordering guarantees live in app/core/timeouts.py.
    task_soft_time_limit=task_soft_time_limit_seconds(settings),
    task_time_limit=task_hard_time_limit_seconds(settings),
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

celery_app.conf.worker_concurrency = settings.celery_worker_concurrency
celery_app.conf.worker_max_tasks_per_child = settings.celery_worker_max_tasks_per_child

# CloudAMQP meters `published + delivered` against a monthly quota, and Celery's
# default event traffic spends several messages per task on top of the task
# itself. Nothing in this service consumes Celery events, so the whole
# cluster-message machinery is off; keeping it on can exhaust the quota on a
# fraction of real traffic. Paired with --without-gossip/--without-mingle/
# --without-heartbeat in the worker command.
celery_app.conf.update(
    worker_send_task_events=False,
    event_queue_expires=60,
    # CloudAMQP runs low TCP keep-alive intervals, so AMQP heartbeats are redundant.
    broker_heartbeat=None,
    # This worker only consumes; it never publishes, so one pooled connection suffices.
    broker_pool_limit=1,
)


# Worker-wide publishing convention: every task is published with the job id as
# its first positional argument. Celery's task signals only see the published
# message, not the resolved call, so the job id has to be read from the wire
# arguments rather than from a task signature.
JOB_ID_ARG_INDEX = 0


def _published_job_id(task) -> str:
    """Return the job id a task was invoked with.

    An unrecognised message yields an empty string rather than a guess.
    """
    args = getattr(getattr(task, "request", None), "args", None) or ()
    return str(args[JOB_ID_ARG_INDEX]) if len(args) > JOB_ID_ARG_INDEX else ""


def _tag_job_id(job_id: str) -> None:
    """Mirror the job id onto the Sentry scope so task failures carry it.

    structlog context reaches records that pass through a logger call, but
    ``CeleryIntegration`` also raises events for failures that never reach one
    — an exception escaping the task body, or a hard time-limit kill. The tag
    makes those events filterable by ``job_id:`` in Sentry too. An unrecognised
    message has no job id, and an empty tag would only add noise.
    """
    if not job_id:
        return

    import sentry_sdk

    sentry_sdk.set_tag("job_id", job_id)


def _untag_job_id() -> None:
    """Drop the Sentry job-id tag; prefork children reuse the process scope.

    ``Scope.remove_tag`` is not re-exported at module level in sentry-sdk, and
    ``sentry_sdk.set_tag`` targets the isolation scope — so removal has to go
    through the same scope rather than a module-level helper that does not
    exist.
    """
    import sentry_sdk

    sentry_sdk.get_isolation_scope().remove_tag("job_id")


@task_prerun.connect
def bind_task_context(sender=None, task=None, **kwargs) -> None:
    """Give the whole task — including Celery's own failure report — a job identity."""
    job_context.clear()
    retries = getattr(getattr(task, "request", None), "retries", 0) or 0
    job_id = _published_job_id(task)
    job_context.bind(JobContext(job_id=job_id, retry_attempt=retries))
    _tag_job_id(job_id)


@task_postrun.connect
def unbind_task_context(sender=None, **kwargs) -> None:
    """Drop the job context so the next task on this reused process starts clean."""
    job_context.clear()
    _untag_job_id()
