"""Transcription pipeline task for the JumpTo worker."""

import asyncio
from types import SimpleNamespace

from app.client import BackendClient
from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.models import TranscriptSubmission, TranscriptWordData
from app.providers import (
    SupadataTranscriptProvider,
    TranscriptData,
    TranscriptFetchTranscriptProvider,
    TranscriptJobPending,
    VidWordsTranscriptProvider,
    YouTubeCaptionTranscriptProvider,
    get_media_info_with_raw,
    get_transcript_provider,
)
from app.tasks.celery_app import celery_app
from app.utils.text import normalize_word

logger = get_logger(__name__)

_RETRY_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2
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
    """Fetch a transcript, trying configured cloud providers first.

    Order: TranscriptFetch -> Supadata -> VidWords -> yt-dlp captions ->
    Assembly. Each cloud provider is skipped (next one tried) when it has no
    transcript or hits a transient/permanent error. A still-processing async
    job raises ``TranscriptJobPending`` (bubbles to the retry-aware task),
    carrying a resume token the originating provider follows on its next
    attempt; a completed or failed job resolves within this attempt.
    """
    for provider in _cloud_providers():
        resume = resume_token if (resume_token and provider.name == resume_provider) else ""
        result = await _try_cloud(provider, job, resume)
        if result is not None:
            logger.info(
                "Transcript provider used",
                provider=provider.name,
                youtube_url=job.youtube_url,
            )
            return _build_cloud_submission(result, provider.name)
    media, info = await asyncio.to_thread(
        get_media_info_with_raw, job.youtube_video_id, job.youtube_url
    )
    transcript = await _fetch_transcript_with_retry(job.youtube_url, info)
    return _build_submission(media, transcript, provider="yt-dlp")


def _cloud_providers() -> list:
    """Build the configured cloud transcript providers in priority order."""
    if not _live_pipeline_enabled():
        return []
    settings = get_settings()
    providers: list = []
    transcriptfetch_key = getattr(settings, "transcriptfetch_api_key", "")
    if transcriptfetch_key:
        providers.append(
            TranscriptFetchTranscriptProvider(
                api_key=transcriptfetch_key,
                lang=getattr(settings, "transcriptfetch_lang", "en") or "en",
                mode=getattr(settings, "transcriptfetch_mode", "auto") or "auto",
            )
        )
    supadata_key = getattr(settings, "supadata_api_key", "")
    if supadata_key:
        providers.append(
            SupadataTranscriptProvider(
                api_key=supadata_key,
                lang=getattr(settings, "supadata_lang", "en") or "en",
                mode=getattr(settings, "supadata_mode", "auto") or "auto",
            )
        )
    vidwords_key = getattr(settings, "vidwords_api_key", "")
    if vidwords_key:
        providers.append(
            VidWordsTranscriptProvider(
                api_key=vidwords_key,
                base_url=getattr(settings, "vidwords_api_url", "https://vidwords.com"),
                lang=getattr(settings, "vidwords_lang", "en") or "en",
            )
        )
    return providers


async def _try_cloud(provider, job, resume_token: str = ""):
    """Fetch via a cloud provider; any miss/error moves to the next provider.

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


def _build_cloud_submission(result, provider: str) -> TranscriptSubmission:
    """Build a submission payload directly from a cloud provider result."""
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


async def _fetch_transcript_with_retry(
    youtube_url: str,
    info: dict | None = None,
) -> TranscriptData:
    """Fetch a transcript, retrying transient external failures."""
    if _live_pipeline_enabled():
        try:
            return await YouTubeCaptionTranscriptProvider().fetch(youtube_url, info=info)
        except Exception:
            logger.exception(
                "Captions fast-path failed; falling back to audio transcription",
                youtube_url=youtube_url,
            )
    provider = get_transcript_provider()
    last_error: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return await provider.fetch(youtube_url)
        except ExternalServiceError as exc:
            last_error = exc
            logger.warning("Transcript fetch attempt failed", attempt=attempt + 1)
            if attempt + 1 < _RETRY_ATTEMPTS:
                await asyncio.sleep(_RETRY_DELAY_SECONDS * (attempt + 1))
    if last_error:
        raise last_error
    return await provider.fetch(youtube_url)  # pragma: no cover


def _live_pipeline_enabled() -> bool:
    """Return whether the live external transcription pipeline is active."""
    settings = get_settings()
    return settings.jumpto_live_external_calls and settings.jumpto_transcript_mode.lower() != "fake"


def _user_safe_message(exc: Exception) -> str:
    """Map an exception to a user-safe failure message."""
    if isinstance(exc, ExternalServiceError):
        return _EXTERNAL_FAILURE
    if isinstance(exc, TimeoutError):
        return _TIMEOUT_SAFE_MESSAGE
    return _USER_SAFE_FAILURE
