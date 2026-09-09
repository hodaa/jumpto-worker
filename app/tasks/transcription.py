"""Transcription pipeline task for the JumpTo worker."""

import asyncio
import atexit
import os
import uuid
from types import SimpleNamespace

from app.client import BackendClient
from app.client.http import close_shared_http_clients
from app.core.config import _live_pipeline_enabled, get_settings
from app.core.exceptions import ExternalServiceError, PermanentExternalServiceError
from app.core.logging import get_logger
from app.models import TranscriptSubmission, TranscriptWordData
from app.providers import TranscriptData, TranscriptJobPending
from app.providers.cache import extract_youtube_video_id, get_transcript_cache
from app.providers.registry import build_provider_chain
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
_CACHE_LOCK_WAIT_INTERVAL_SECONDS = 0.5
_CACHE_LOCK_MAX_WAIT_SECONDS = 15

# A Celery prefork process executes tasks serially, so one persistent loop per
# process lets async clients and their connection pools survive task boundaries.
_EVENT_LOOP: asyncio.AbstractEventLoop | None = None
_EVENT_LOOP_PID: int | None = None


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


async def run_pipeline(job_id: str, resume_token: str = "", resume_provider: str = "") -> dict:
    """Run the transcription pipeline for a job against the backend API.

    ``resume_token``/``resume_provider`` carry a pending cloud async job across
    Celery retries: when present the job is already ``processing`` and the
    pipeline resumes by checking that job once (routed to the originating
    provider) instead of requeueing a new transcription.
    """
    settings = get_settings()
    client = BackendClient(settings.backend_url, settings.internal_api_key)

    try:
        job = await client.get_job(job_id)
        if not resume_token and job.status != "pending":
            logger.info("Skipping non-pending job", job_id=job_id, status=job.status)
            return {"status": job.status}
        if job.status == "pending":
            await client.advance_job(job_id)

        submission = await asyncio.wait_for(
            _perform_transcription(job, resume_token, resume_provider),
            timeout=settings.job_timeout_seconds,
        )
        await client.store_transcript(job_id, submission)
        await client.complete_job(job_id)
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
        try:
            await client.fail_job(job_id, error)
        except Exception:
            logger.exception("Failed to mark job as failed", job_id=job_id)
        raise
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            await close()

    logger.info("Pipeline completed", job_id=job_id)
    return {"status": "completed", "video_id": job.video_id}


async def _perform_transcription(
    job, resume_token: str = "", resume_provider: str = ""
) -> TranscriptSubmission:
    """Fetch a transcript through the configured strategy chain.

    A resume token starts at its owning provider and never replays providers
    that already ran. Successful results are cached across all strategies and
    a Redis lock prevents concurrent jobs for one video from doing duplicate
    provider work.
    """
    strategies = _provider_chain()
    start_index = 0
    if resume_token:
        if not resume_provider:
            raise ExternalServiceError(
                "Cannot resume transcription without an originating provider",
                service="transcription",
            )
        try:
            start_index = next(
                index for index, provider in enumerate(strategies) if provider.name == resume_provider
            )
        except StopIteration:
            raise ExternalServiceError(
                "The originating transcription provider is no longer configured",
                service="transcription",
            ) from None

    cache = None
    cache_video_id = ""
    cache_lock_owner = ""
    settings = get_settings()
    if not resume_token and _live_pipeline_enabled(settings):
        cache_video_id = job.youtube_video_id or extract_youtube_video_id(job.youtube_url)
        if cache_video_id:
            cache = get_transcript_cache()
            cached = await asyncio.to_thread(cache.get_with_provider, cache_video_id)
            if cached is not None:
                result, cached_provider = cached
                logger.info("Transcript cache hit", video_id=cache_video_id)
                return _build_result_submission(result, cached_provider or "cache")
            cache_lock_owner = await _acquire_cache_lock(cache, cache_video_id)
            # Re-check after waiting/acquiring in case another worker
            # populated the cache between the initial read and SET NX.
            cached = await asyncio.to_thread(cache.get_with_provider, cache_video_id)
            if cached is not None:
                if cache_lock_owner:
                    await asyncio.to_thread(
                        cache.release_lock, cache_video_id, cache_lock_owner
                    )
                result, cached_provider = cached
                logger.info("Transcript cache hit after lock", video_id=cache_video_id)
                return _build_result_submission(result, cached_provider or "cache")

    try:
        for provider in strategies[start_index:]:
            resume = resume_token if provider.name == resume_provider else ""
            result = await _try_cloud(provider, job, resume)
            if result is not None:
                logger.info(
                    "Transcript provider used",
                    provider=provider.name,
                    youtube_url=job.youtube_url,
                )
                if cache is not None and cache_video_id:
                    await asyncio.to_thread(
                        cache.set, cache_video_id, result, provider.name
                    )
                return _build_result_submission(result, provider.name)
        raise ExternalServiceError(
            "Could not fetch the transcript for this video",
            service="transcription",
        )
    finally:
        if cache is not None and cache_video_id and cache_lock_owner:
            await asyncio.to_thread(cache.release_lock, cache_video_id, cache_lock_owner)


async def _acquire_cache_lock(cache, video_id: str) -> str:
    """Wait briefly for another worker to populate a cache miss."""
    owner = uuid.uuid4().hex
    deadline = asyncio.get_running_loop().time() + _CACHE_LOCK_MAX_WAIT_SECONDS
    while True:
        acquired, owner = await asyncio.to_thread(cache.acquire_lock, video_id, owner)
        if acquired:
            return owner
        if asyncio.get_running_loop().time() >= deadline:
            logger.warning("Transcript cache lock wait timed out; proceeding", video_id=video_id)
            return ""
        await asyncio.sleep(_CACHE_LOCK_WAIT_INTERVAL_SECONDS)
        cached = await asyncio.to_thread(cache.get_with_provider, video_id)
        if cached is not None:
            # The caller will perform the final cache read after acquiring a
            # lock. Returning an empty owner allows it to continue without
            # holding a lock that it does not own.
            return ""




def _provider_chain() -> list:
    """Build the configured transcript strategy chain in execution order.

    The provider named by ``DEFAULT_VIDEO_PROVIDER`` leads the chain; every
    other registered strategy follows in standard priority order. Cloud
    strategies are skipped while live external calls are disabled; the local
    ``yt-dlp`` strategy is always the terminal fallback.
    """
    strategies = build_provider_chain(get_settings())
    if not _live_pipeline_enabled(get_settings()):
        return [strategy for strategy in strategies if not strategy.uses_cloud]
    return strategies


async def _try_cloud(provider, job, resume_token: str = ""):
    """Fetch via a provider strategy; any miss/error moves to the next one.

    A still-processing async job (``TranscriptJobPending``) is re-raised so the
    task can decide whether to retry it or fail the job; everything else is
    treated as a regular miss (next provider tried).
    """
    try:
        return await provider.fetch(
            job.youtube_url, job.youtube_video_id, resume_token=resume_token
        )
    except TranscriptJobPending:
        raise
    except PermanentExternalServiceError:
        logger.exception(
            "Permanent cloud transcript provider failure",
            provider=provider.name,
            youtube_url=job.youtube_url,
        )
        raise
    except Exception:
        logger.exception(
            "Cloud transcript provider failed; trying next provider",
            provider=provider.name,
            youtube_url=job.youtube_url,
        )
        return None


def _build_result_submission(result, provider: str) -> TranscriptSubmission:
    """Build a submission payload directly from a provider strategy result."""
    media = SimpleNamespace(
        title=result.title or "Untitled video",
        duration_seconds=result.duration_seconds,
    )
    return _build_submission(media, result.transcript, provider=provider)


def _build_submission(
    media, transcript: TranscriptData, provider: str = ""
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
        title=media.title,
        duration_seconds=media.duration_seconds,
        language=transcript.language,
        transcript_text=transcript.text,
        words=words,
        provider=provider,
    )


@celery_app.task(bind=True, max_retries=_CLOUD_JOB_ATTEMPTS, default_retry_delay=60)
def download_and_transcribe(
    self, job_id: str, resume_token: str = "", resume_provider: str = ""
) -> dict:
    """Run the transcription pipeline from the Celery worker.

    While a resumable cloud async job (e.g. Supadata) is processing, the task
    returns early (instead of blocking in a poll loop) and re-enqueues itself
    with an exponential backoff, carrying the job's resume token routed to its
    originating provider. A non-resumable pending job (e.g. TranscriptFetch's
    one-call rule) is failed immediately. When the waiting budget runs out the
    job is marked failed and the task gives up.
    """
    try:
        _run_async(run_pipeline(job_id, resume_token, resume_provider))
    except TranscriptJobPending as exc:
        if not exc.resumable:
            _run_async(_fail_job(job_id, _EXTERNAL_FAILURE))
            raise
        if self.request.retries >= _CLOUD_JOB_ATTEMPTS - 1:
            _run_async(_fail_job(job_id, _CLOUD_JOB_TIMEOUT_SAFE_MESSAGE))
            raise
        raise self.retry(
            exc=exc,
            max_retries=_CLOUD_JOB_ATTEMPTS,
            countdown=_job_retry_countdown(self.request.retries),
            args=[job_id, exc.resume_token, exc.provider],
        ) from exc
    return {"status": "completed"}


async def _fail_job(job_id: str, message: str) -> None:
    """Best-effort mark a job as failed in the backend."""
    settings = get_settings()
    client = BackendClient(settings.backend_url, settings.internal_api_key)
    try:
        await client.fail_job(job_id, message)
    except Exception:
        logger.exception("Failed to mark job as failed", job_id=job_id)
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            await close()


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
