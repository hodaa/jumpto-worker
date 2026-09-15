"""Celery task scheduling the transcription pipeline.

The pipeline orchestration lives in ``app.services`` (``pipeline.run_pipeline``)
and the process-wide async event loop in ``app.services.event_loop``; this
module is a thin scheduling shell that owns the Celery-facing behaviour:
retry semantics, the async-job wait budget, and clean webhook-driven exit.
"""

from collections.abc import Callable

from app.client import BackendClient
from app.core.logging import get_logger
from app.providers import TranscriptJobPending
from app.services.event_loop import run_async
from app.services.jobs import fail_job
from app.services.pipeline import EXTERNAL_FAILURE, run_pipeline
from app.tasks.celery_app import celery_app

logger = get_logger(__name__)

# Waiting budget for resumable async cloud jobs (e.g. Supadata AI generation).
# Each Celery attempt makes one provider call; the task retries on an
# exponential backoff until the job completes or the budget is exhausted.
_CLOUD_JOB_ATTEMPTS = 9
_CLOUD_JOB_RETRY_BASE_SECONDS = 2
_CLOUD_JOB_RETRY_MAX_SECONDS = 60

# Task-side user-safe failure message: shown when a resumable async job still
# has not completed within the in-worker retry budget.
_CLOUD_JOB_TIMEOUT_SAFE_MESSAGE = "Transcription timed out. Please try again later."


@celery_app.task(bind=True, max_retries=_CLOUD_JOB_ATTEMPTS, default_retry_delay=60)
def download_and_transcribe(
    self,
    job_id: str,
    resume_token: str = "",
    resume_provider: str = "",
    client_factory: Callable[..., BackendClient] | None = None,
) -> dict:

    try:
        run_async(
            run_pipeline(
                job_id,
                resume_token,
                resume_provider,
                client_factory=client_factory,
            )
        )
    except TranscriptJobPending as exc:
        if exc.webhook:
            logger.info(
                "Cloud provider completion webhook armed; ending task",
                job_id=job_id,
                resume_token=exc.resume_token,
                resume_provider=exc.provider,
            )
            return {"status": "submitted"}
        if not exc.resumable:
            run_async(fail_job(job_id, EXTERNAL_FAILURE, client_factory=client_factory))
            raise
        if self.request.retries >= _CLOUD_JOB_ATTEMPTS - 1:
            run_async(
                fail_job(job_id, _CLOUD_JOB_TIMEOUT_SAFE_MESSAGE, client_factory=client_factory)
            )
            raise
        raise self.retry(
            exc=exc,
            max_retries=_CLOUD_JOB_ATTEMPTS,
            countdown=_job_retry_countdown(self.request.retries),
            args=[job_id, exc.resume_token, exc.provider],
        ) from exc
    return {"status": "completed"}


def _job_retry_countdown(retries: int) -> int:
    """Exponential backoff in seconds for the async-job retry schedule."""
    return min(_CLOUD_JOB_RETRY_BASE_SECONDS * (2**retries), _CLOUD_JOB_RETRY_MAX_SECONDS)
