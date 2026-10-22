"""Shared audio-job plumbing for the worker-local transcript strategies.

Both local strategies (youtube, drive) end a job the same way: hand an audio
file to the configured transcription leaf, honor the same retry budget, and
re-route a nested cloud job's pending signal to the strategy that owns the
job. This module holds that shared surface — the :class:`AudioJob` request,
the attempt loops, and the pending remap — so neither strategy duplicates it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import ExternalServiceError, PermanentExternalServiceError
from app.core.logging import get_logger
from app.providers.audio import AudioFileLease
from app.providers.models import TranscriptData, TranscriptJobPending
from app.sources.base import CaptionlessSource, PreparedAudio
from app.speech_to_text.base import SpeechToTextProvider

logger = get_logger(__name__)

_RETRY_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2


@dataclass(frozen=True)
class AudioJob:
    """One audio transcription request: what the leaf is called with, and its context.

    Bundling the leaf parameters keeps them from being threaded through the
    retry loop as loose arguments, and lets the leaf call be a plain typed call
    instead of a ``**kwargs`` splat — so a renamed leaf parameter fails here
    loudly rather than being silently dropped. ``video_id`` keys the shared
    audio file; ``language`` lets an asynchronous transcriber pick the language
    from the video title; ``resume_token`` polls a pending cloud job instead of
    submitting a new one; ``webhook_url`` is the completion callback URL,
    forwarded to Assembly on the initial submit only; ``audio_path`` is a
    caller-owned audio file (captionless sources) that replaces the lease's own
    download and is reused across transcription retries.
    """

    youtube_url: str
    video_id: str = ""
    resume_token: str = ""
    webhook_url: str = ""
    language: str = ""
    audio_path: str = ""


async def run_download_attempts(
    source: CaptionlessSource,
    url: str,
    directory: Path,
    settings: Settings | None = None,
) -> PreparedAudio:
    """Run the download retry budget for one captionless acquisition.

    Only a *download* failure is retried here; a failed transcription is
    retried inside the attempts loop against the same extracted audio.
    """
    last_error: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return await asyncio.to_thread(source.prepare_audio, url, directory, settings)
        except PermanentExternalServiceError:
            raise
        except ExternalServiceError as exc:
            last_error = exc
            logger.warning(
                "Source media acquisition attempt failed",
                attempt=attempt + 1,
                source=source.name,
                error=str(exc),
            )
            if attempt + 1 < _RETRY_ATTEMPTS:
                await asyncio.sleep(_RETRY_DELAY_SECONDS * (attempt + 1))
    assert last_error is not None
    raise last_error


async def run_transcription_attempts(
    provider: SpeechToTextProvider,
    job: AudioJob,
    lease: AudioFileLease | None = None,
    *,
    strategy: str,
) -> TranscriptData:
    """Run the retry budget for one audio job, holding the file across attempts.

    With a ``lease`` (the youtube strategy) the audio is downloaded once and
    released when the job settles; captionless jobs instead pass a
    caller-owned ``job.audio_path`` and no lease, so neither download nor
    release ever runs here. Every parameter is passed to the leaf explicitly —
    both leaves default each to ``""`` and treat that as "not provided" — so
    the call is checked against their signatures rather than assembled by name
    at runtime.

    A pending cloud job bubbles up as ``TranscriptJobPending`` re-routed to the
    owning strategy's name, so the task resumes it (or ends cleanly when a
    completion webhook was armed); it leaves a leased file in place for
    whatever resumes it.
    """
    last_error: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            if job.audio_path:
                audio_path = job.audio_path
            elif lease is not None:
                audio_path = await lease.acquire(job.youtube_url, resume=bool(job.resume_token))
            else:
                audio_path = ""
            transcript = await provider.fetch(
                job.youtube_url,
                resume_token=job.resume_token,
                webhook_url=job.webhook_url,
                language=job.language,
                audio_path=audio_path,
            )
        except TranscriptJobPending as exc:
            raise strategy_pending(exc, strategy) from exc
        except ExternalServiceError as exc:
            last_error = exc
            logger.warning("Transcript fetch attempt failed", attempt=attempt + 1)
            if attempt + 1 < _RETRY_ATTEMPTS:
                await asyncio.sleep(_RETRY_DELAY_SECONDS * (attempt + 1))
            continue
        if lease is not None:
            await lease.release()
        return transcript
    if lease is not None:
        await lease.release()
    assert last_error is not None
    raise last_error


def strategy_pending(exc: TranscriptJobPending, strategy: str) -> TranscriptJobPending:
    """Re-route a nested provider's pending job to the strategy-level name.

    The audio leaf (Assembly) is nested inside a local strategy, so a pending
    cloud job must bubble with the strategy's own name — the route that
    ``try_provider`` matches when picking the resume path on retry. Without this
    remap the resume token is dropped and the pipeline re-downloads the same
    audio file on every retry.
    """
    if exc.provider == strategy:
        return exc
    return TranscriptJobPending(
        message=str(exc),
        provider=strategy,
        resume_token=exc.resume_token,
        resumable=exc.resumable,
        webhook=exc.webhook,
        details=exc.details,
    )
