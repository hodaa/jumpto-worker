"""Transcription pipeline task for the JumpTo worker."""

import asyncio
from types import SimpleNamespace

from app.client import BackendClient
from app.core.config import _live_pipeline_enabled, get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.models import TranscriptSubmission, TranscriptWordData
from app.providers import TranscriptData, TranscriptJobPending
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

    Each strategy is tried in order — the provider named by
    ``DEFAULT_VIDEO_PROVIDER`` first (when configured), then the standard
    priority order (TranscriptFetch -> Supadata -> VidWords -> yt-dlp). One
    with no transcript or a transient/permanent error is skipped. A
    still-processing async job raises ``TranscriptJobPending`` (bubbles to the
    retry-aware task). If every strategy misses, the pipeline raises so the
    job is failed rather than completed empty.
    """
    for provider in _provider_chain():
        resume = resume_token if (resume_token and provider.name == resume_provider) else ""
        result = await _try_cloud(provider, job, resume)
        if result is not None:
            logger.info(
                "Transcript provider used",
                provider=provider.name,
                youtube_url=job.youtube_url,
            )
            return _build_result_submission(result, provider.name)
    raise ExternalServiceError(
        "Could not fetch the transcript for this video",
        service="transcription",
    )


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
    _words = [
        (normalize_word(word.word), word.start_time, word.end_time) for word in transcript.words
    ]
    words = [
        TranscriptWordData(
            word_index=index,
            word=normalized,
            start_time=start_time,
            end_time=end_time,
        )
        for index, (normalized, start_time, end_time) in enumerate(w for w in _words if w[0])
    ]
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
        asyncio.run(run_pipeline(job_id, resume_token, resume_provider))
    except TranscriptJobPending as exc:
        if not exc.resumable:
            asyncio.run(_fail_job(job_id, _EXTERNAL_FAILURE))
            raise
        if self.request.retries >= _CLOUD_JOB_ATTEMPTS - 1:
            asyncio.run(_fail_job(job_id, _CLOUD_JOB_TIMEOUT_SAFE_MESSAGE))
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
