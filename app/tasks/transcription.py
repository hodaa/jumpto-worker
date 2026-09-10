import asyncio
import atexit
import os
from collections.abc import Callable

from app.client import BackendClient
from app.client.http import close_shared_http_clients
from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError, PermanentExternalServiceError
from app.core.logging import get_logger
from app.models import TranscriptSubmission, TranscriptWordData
from app.providers import TranscriptData, TranscriptJobPending
from app.providers.registry import candidates
from app.tasks.celery_app import celery_app
from app.utils.text import normalize_word

logger = get_logger(__name__)

_USER_SAFE_FAILURE = "Transcription failed. Please try again later."
_EXTERNAL_FAILURE = "Could not fetch the transcript for this video. Please try again later."
_TIMEOUT_SAFE_MESSAGE = "Transcription timed out. Please try again later."
_CLOUD_JOB_TIMEOUT_SAFE_MESSAGE = "Transcription timed out. Please try again later."

# Waiting budget for resumable async cloud jobs (e.g. Supadata AI generation).
# Each Celery attempt makes one provider call; the task retries on an
# exponential backoff until the job completes or the budget is exhausted.
_CLOUD_JOB_ATTEMPTS = 9
_CLOUD_JOB_RETRY_BASE_SECONDS = 2
_CLOUD_JOB_RETRY_MAX_SECONDS = 60

# A Celery prefork process executes tasks serially, so one persistent loop per
# process lets async clients and their connection pools survive task boundaries.
_EVENT_LOOP: asyncio.AbstractEventLoop | None = None
_EVENT_LOOP_PID: int | None = None


@celery_app.task(bind=True, max_retries=_CLOUD_JOB_ATTEMPTS, default_retry_delay=60)
def download_and_transcribe(
    self,
    job_id: str,
    resume_token: str = "",
    resume_provider: str = "",
    client_factory: Callable[..., BackendClient] | None = None,
) -> dict:

    try:
        _run_async(
            run_pipeline(
                job_id,
                resume_token,
                resume_provider,
                client_factory=client_factory,
            )
        )
    except TranscriptJobPending as exc:
        if not exc.resumable:
            _run_async(
                _fail_job(job_id, _EXTERNAL_FAILURE, client_factory=client_factory)
            )
            raise
        if self.request.retries >= _CLOUD_JOB_ATTEMPTS - 1:
            _run_async(
                _fail_job(job_id, _CLOUD_JOB_TIMEOUT_SAFE_MESSAGE, client_factory=client_factory)
            )
            raise
        raise self.retry(
            exc=exc,
            max_retries=_CLOUD_JOB_ATTEMPTS,
            countdown=_job_retry_countdown(self.request.retries),
            args=[job_id, exc.resume_token, exc.provider],
        ) from exc
    return {"status": "completed"}


def _run_async(coro):
    """Run a coroutine on the worker-process event loop."""
    global _EVENT_LOOP, _EVENT_LOOP_PID
    pid = os.getpid()
    if _EVENT_LOOP is None or _EVENT_LOOP.is_closed() or pid != _EVENT_LOOP_PID:
        _EVENT_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_EVENT_LOOP)
        _EVENT_LOOP_PID = pid
    return _EVENT_LOOP.run_until_complete(coro)


def _close_worker_event_loop() -> None:
    """Close pooled async clients and the persistent worker loop at exit."""
    global _EVENT_LOOP
    if _EVENT_LOOP is None or _EVENT_LOOP.is_closed():
        return
    _EVENT_LOOP.run_until_complete(close_shared_http_clients(_EVENT_LOOP))
    _EVENT_LOOP.close()
    _EVENT_LOOP = None


atexit.register(_close_worker_event_loop)


async def run_pipeline(
    job_id: str,
    resume_token: str = "",
    resume_provider: str = "",
    client_factory: Callable[..., BackendClient] | None = None,
) -> dict:
    """Run the transcription pipeline for a job against the backend API.

    ``resume_token``/``resume_provider`` carry a pending cloud async job across
    Celery retries: when present the job is already ``processing`` and the
    pipeline resumes by checking that job once (routed to the originating
    provider) instead of requeueing a new transcription.
    """
    settings = get_settings()
    factory = client_factory or _new_client
    client = factory(settings)

    try:
        job = await _load_job(client, job_id)
        submission = await asyncio.wait_for(
            _perform_transcription(job, resume_token, resume_provider),
            timeout=settings.job_timeout_seconds,
        )
        await _submit_result(client, job_id, submission)
    except TranscriptJobPending:
        logger.info(
            "Cloud provider job still processing; task will retry or fail",
            job_id=job_id,
            resume_token=resume_token,
            resume_provider=resume_provider,
        )
        raise
    except Exception as exc:
        error = _user_safe_message(exc)
        logger.exception("Transcription pipeline failed", job_id=job_id, error=error)
        await _mark_failed(client, job_id, error)
        raise
    finally:
        await _close_client(client)

    logger.info("Pipeline completed", job_id=job_id)
    return {"status": "completed", "video_id": job.video_id}


def _new_client(settings) -> BackendClient:
    """Build a backend client from worker settings."""
    return BackendClient(settings.backend_url, settings.internal_api_key)


async def _close_client(client) -> None:
    """Best-effort close a backend client."""
    close = getattr(client, "close", None)
    if close is not None:
        await close()


async def _load_job(client, job_id: str):
    """Fetch a job and advance it from ``pending`` to ``processing``."""
    job = await client.get_job(job_id)
    if job.status == "pending":
        await client.advance_job(job_id)
    return job


async def _submit_result(client, job_id: str, submission) -> None:
    """Submit the transcript and mark the job completed."""
    await client.store_transcript(job_id, submission)
    await client.complete_job(job_id)


async def _mark_failed(client, job_id: str, message: str) -> None:
    """Best-effort mark a job as failed through an existing client."""
    try:
        await client.fail_job(job_id, message)
    except Exception:
        logger.exception("Failed to mark job as failed", job_id=job_id)


async def _perform_transcription(
    job, resume_token: str = "", resume_provider: str = ""
) -> TranscriptSubmission:
    for candidate in candidates(get_settings()):
        result = await _try_provider(candidate, job, resume_token, resume_provider)
        if result is not None:
            logger.info(
                "Transcript provider used",
                provider=candidate.name,
                youtube_url=job.youtube_url,
            )
            return _build_result_submission(result, candidate.name)

    raise ExternalServiceError(
        "Could not fetch the transcript for this video",
        service="transcription",
    )


async def _try_provider(provider, job, resume_token: str = "", resume_provider: str = ""):
    """Fetch via a single provider strategy; a miss/error falls through.

    A still-processing async job (``TranscriptJobPending``) is re-raised so the
    task can decide whether to retry it or fail the job; everything else is
    treated as a regular miss (the caller moves to the fallback).
    """
    resume = (
        resume_token
        if (resume_token and provider.supports_resume and provider.name == resume_provider)
        else ""
    )
    try:
        return await provider.fetch(job.youtube_url, job.youtube_video_id, resume_token=resume)
    except TranscriptJobPending:
        raise
    except PermanentExternalServiceError:
        logger.exception(
            "Permanent cloud transcript provider failure",
            provider=provider.name,
            youtube_url=job.youtube_url,
        )
        raise
    except ExternalServiceError:
        logger.exception(
            "Transcript provider failed; falling back",
            provider=provider.name,
            youtube_url=job.youtube_url,
        )
        return None


def _build_result_submission(result, provider: str) -> TranscriptSubmission:
    """Build a submission payload directly from a provider strategy result."""
    return _build_submission(
        title=result.title or "Untitled video",
        duration_seconds=result.duration_seconds,
        transcript=result.transcript,
        provider=provider,
    )


def _build_submission(
    *,
    title: str,
    duration_seconds: int,
    transcript: TranscriptData,
    provider: str = "",
) -> TranscriptSubmission:
    """Build a transcript submission payload from media and transcript data."""
    words: list[TranscriptWordData] = []
    for word in transcript.words:
        normalized = normalize_word(word.word)
        if normalized:
            words.append(
                TranscriptWordData(
                    word_index=len(words),
                    word=normalized,
                    start_time=word.start_time,
                    end_time=word.end_time,
                )
            )
    return TranscriptSubmission(
        title=title,
        duration_seconds=duration_seconds,
        language=transcript.language,
        transcript_text=transcript.text,
        words=words,
        provider=provider,
    )


async def _fail_job(
    job_id: str,
    message: str,
    client_factory: Callable[..., BackendClient] | None = None,
) -> None:
    """Best-effort mark a job as failed in the backend."""
    settings = get_settings()
    factory = client_factory or _new_client
    client = factory(settings)
    try:
        await _mark_failed(client, job_id, message)
    finally:
        await _close_client(client)


def _job_retry_countdown(retries: int) -> int:
    """Exponential backoff in seconds for the async-job retry schedule."""
    return min(_CLOUD_JOB_RETRY_BASE_SECONDS * (2**retries), _CLOUD_JOB_RETRY_MAX_SECONDS)


def _user_safe_message(exc: Exception) -> str:
    """Map an exception to a user-safe failure message."""
    if isinstance(exc, ExternalServiceError):
        return _EXTERNAL_FAILURE
    if isinstance(exc, TimeoutError):
        return _TIMEOUT_SAFE_MESSAGE
    return _USER_SAFE_FAILURE
