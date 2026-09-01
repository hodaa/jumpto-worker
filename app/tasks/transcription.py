"""Transcription pipeline task for the JumpTo worker."""

import asyncio

from app.client import BackendClient
from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.models import TranscriptSubmission, TranscriptWordData
from app.providers import (
    TranscriptData,
    YouTubeCaptionTranscriptProvider,
    get_media_info,
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

PROGRESS_ADVANCE = 10
PROGRESS_MEDIA = 40
PROGRESS_TRANSCRIPT = 70
PROGRESS_STORE = 90


async def run_pipeline(job_id: str) -> dict:
    """Run the transcription pipeline for a job against the backend API."""
    settings = get_settings()
    client = BackendClient(settings.backend_url, settings.internal_api_key)

    async def report(progress: int) -> None:
        """Report progress to the backend, ignoring reporting failures."""
        try:
            await client.report_progress(job_id, progress)
        except Exception:
            logger.exception("Failed to report progress", job_id=job_id, progress=progress)

    try:
        job = await client.get_job(job_id)
        if job.status != "pending":
            logger.info("Skipping non-pending job", job_id=job_id, status=job.status)
            return {"status": job.status}

        await client.advance_job(job_id)
        submission = await asyncio.wait_for(
            _perform_transcription(job, report),
            timeout=settings.job_timeout_seconds,
        )
        await report(PROGRESS_STORE)
        await client.store_transcript(job_id, submission)
        await client.complete_job(job_id)
    except Exception as exc:
        error = _user_safe_message(exc)
        logger.exception("Transcription pipeline failed", job_id=job_id, error=error)
        try:
            await client.fail_job(job_id, error)
        except Exception:
            logger.exception("Failed to mark job as failed", job_id=job_id)
        raise

    logger.info("Pipeline completed", job_id=job_id)
    return {"status": "completed", "video_id": job.video_id}


async def _perform_transcription(job, report) -> TranscriptSubmission:
    """Fetch media metadata and transcript, then build a submission payload."""
    await report(PROGRESS_MEDIA)
    media = await asyncio.to_thread(get_media_info, job.youtube_video_id, job.youtube_url)
    transcript = await _fetch_transcript_with_retry(job.youtube_url)
    await report(PROGRESS_TRANSCRIPT)
    return _build_submission(media, transcript)


def _build_submission(media, transcript: TranscriptData) -> TranscriptSubmission:
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
    )


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def download_and_transcribe(self, job_id: str) -> dict:
    """Run the transcription pipeline from the Celery worker."""
    asyncio.run(run_pipeline(job_id))
    return {"status": "completed"}


async def _fetch_transcript_with_retry(youtube_url: str) -> TranscriptData:
    """Fetch a transcript, retrying transient external failures."""
    if _live_pipeline_enabled():
        try:
            return await YouTubeCaptionTranscriptProvider().fetch(youtube_url)
        except ExternalServiceError:
            logger.info("Captions fast-path unavailable; falling back to audio transcription")
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
