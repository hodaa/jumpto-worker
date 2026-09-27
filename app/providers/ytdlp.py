"""Local transcript strategy: yt-dlp captions, then audio transcription.

Unlike the cloud providers, this strategy runs on the worker itself: yt-dlp
downloads the caption track (fast path) or the audio stream, which the
configured audio-transcription leaf (Deepgram by default, or Assembly)
transcribes. It is the terminal fallback — it either produces a transcript or
raises, so a job that reaches it with no result is failed, not completed empty.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from app.core.config import Settings, _live_pipeline_enabled, get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.providers.audio import AudioFileLease
from app.providers.base import TranscriptProviderStrategy, TranscriptService, VideoTranscriptResult
from app.providers.media import MediaInfo, get_media_info_with_raw
from app.providers.models import TranscriptData, TranscriptJobPending
from app.providers.registry import TranscriptProviderSpec, provider_spec, register_provider
from app.providers.speech_to_text import get_speech_to_text_provider
from app.providers.transcript import YouTubeCaptionTranscriptService
from app.storage.cache import TranscriptCache, extract_youtube_video_id, get_transcript_cache
from app.utils.text import detect_language_from_title

logger = get_logger(__name__)

_ORDER = 5


def register_self() -> None:
    """Register this strategy's spec. Idempotent; safe on every import."""
    if provider_spec(YtDlpTranscriptProvider.name) is not None:
        return
    register_provider(
        TranscriptProviderSpec(
            name=YtDlpTranscriptProvider.name,
            build=lambda settings: YtDlpTranscriptProvider(settings=settings),
            order=_ORDER,
            description="Local yt-dlp captions with audio transcription fallback (free, first provider).",
            uses_cloud=False,
            cache_config_fields=("speech_to_text_provider",),
        )
    )


_RETRY_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2


@dataclass(frozen=True)
class _AudioJob:
    """One audio transcription request: what the leaf is called with, and its context.

    Bundling the four leaf parameters keeps them from being threaded through the
    retry loop as loose arguments, and lets the leaf call be a plain typed call
    instead of a ``**kwargs`` splat — so a renamed leaf parameter fails here
    loudly rather than being silently dropped. ``video_id`` keys the shared audio
    file; ``language`` lets an asynchronous transcriber pick the language from
    the video title; ``resume_token`` polls a pending cloud job instead of
    submitting a new one; ``webhook_url`` is the completion callback URL,
    forwarded to Assembly on the initial submit only.
    """

    youtube_url: str
    video_id: str = ""
    resume_token: str = ""
    webhook_url: str = ""
    language: str = ""


class YtDlpTranscriptProvider(TranscriptProviderStrategy):
    """Downloads the transcript directly (yt-dlp captions, then audio transcription)."""

    name = "yt-dlp"
    supports_resume = True
    supports_webhook = True

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        captions_service: YouTubeCaptionTranscriptService | None = None,
        speech_to_text_provider: Callable[..., TranscriptService | None] | None = None,
        cache: TranscriptCache | None = None,
    ) -> None:
        """Build the strategy, optionally wiring collaborators explicitly.

        ``settings`` is injected by the registry spec builder; when ``None`` the
        global settings resolve lazily per call so a plain
        ``YtDlpTranscriptProvider()`` keeps working. ``captions_service``,
        ``speech_to_text_provider``, and ``cache`` abstract the leaf
        collaborators (caption fast path, audio transcription, and the
        transcript cache); when omitted they fall back to the concrete service,
        the audio-transcription factory, and the shared cache built by
        :func:`~app.storage.cache.get_transcript_cache`, preserving current
        behavior.
        """
        self._settings = settings
        self._captions_service = captions_service
        self._speech_to_text_provider = speech_to_text_provider or get_speech_to_text_provider
        self._cache = cache

    def _resolve_settings(self) -> Settings:
        return self._settings or get_settings()

    def _resolve_captions_service(self) -> YouTubeCaptionTranscriptService:
        # The injected settings are threaded in so the caption download builds
        # its yt-dlp options (cookie file, socket timeout) from the caller's
        # config instead of re-reading the process-wide singleton.
        return self._captions_service or YouTubeCaptionTranscriptService(
            settings=self._resolve_settings()
        )

    async def fetch(
        self,
        youtube_url: str,
        youtube_video_id: str = "",
        resume_token: str = "",
        *,
        webhook_url: str = "",
    ) -> VideoTranscriptResult:
        """Fetch media metadata and the best available transcript.

        The caption fast path is tried first; captionless or failing videos
        fall back to audio transcription. ``resume_token`` resumes a previously
        queued Assembly job so a retry re-polls it instead of re-downloading
        the audio. ``webhook_url`` is a public callback URL for the pending
        Assembly job (built by the pipeline); it arms a completion webhook
        instead of in-worker polling.

        When the live pipeline is on, finished transcripts are cached on disk
        keyed by the video id, so a later job for the same video skips yt-dlp
        entirely. The id comes from the job when present or is parsed from the
        URL otherwise; unparseable ids simply disable caching.

        A resume of a pending Assembly job reads the title/duration from the
        cache (written when the job was first submitted) instead of re-running
        the yt-dlp metadata extract — the only work left is a single Assembly
        poll. A cache miss falls back to a fresh metadata fetch.

        The video title is used to derive a language tag (Arabic-script titles
        transcribe as Arabic, else English) that is forwarded to the audio
        leaf, which Deepgram consumes for the transcription request.
        """
        video_id = youtube_video_id or extract_youtube_video_id(youtube_url)
        cached = await self._read_cache(video_id)
        if cached is not None:
            return cached

        media = None
        if resume_token:
            media = await self._read_media(video_id)

        if media is not None:
            info = None  # resume skips the caption fast path entirely
        else:
            media, info = await asyncio.to_thread(
                get_media_info_with_raw,
                youtube_video_id,
                youtube_url,
                settings=self._resolve_settings(),
            )

        language = detect_language_from_title(media.title)
        transcript = await self._fetch_transcript_with_retry(
            youtube_url,
            info,
            resume_token=resume_token,
            webhook_url=webhook_url,
            language=language,
            video_id=video_id,
        )
        result = _build_video_result(media, transcript)

        await self._write_cache(video_id, result)
        await self._write_media(video_id, media)
        return result

    def _usable_cache(self, video_id: str):
        """Return the transcript cache, or ``None`` when it must not be used.

        Caching requires a known video id and an enabled live pipeline.
        """
        if not video_id or not _live_pipeline_enabled(self._resolve_settings()):
            return None
        return self._cache or get_transcript_cache()

    async def _read_cache(self, video_id: str) -> VideoTranscriptResult | None:
        """Look up a cached transcript for a video id, if caching is usable."""
        cache = self._usable_cache(video_id)
        if cache is None:
            return None
        cached = await asyncio.to_thread(cache.get, video_id)
        if cached is not None:
            logger.info("Transcript cache hit", video_id=video_id)
        return cached

    async def _write_cache(self, video_id: str, result: VideoTranscriptResult) -> None:
        """Store a finished transcript in the cache, if caching is usable."""
        cache = self._usable_cache(video_id)
        if cache is None:
            return
        await asyncio.to_thread(cache.set, video_id, result)

    async def _read_media(self, video_id: str) -> MediaInfo | None:
        """Read cached title/duration so a resume skips the metadata extract."""
        cache = self._usable_cache(video_id)
        if cache is None:
            return None
        media = await asyncio.to_thread(cache.get_media, video_id)
        if media is not None:
            logger.info("Media metadata cache hit", video_id=video_id)
        return media

    async def _write_media(self, video_id: str, media: MediaInfo) -> None:
        """Cache title/duration so a later resume can skip the re-extract."""
        cache = self._usable_cache(video_id)
        if cache is None:
            return
        await asyncio.to_thread(cache.set_media, video_id, media.title, media.duration_seconds)

    async def _fetch_transcript_with_retry(
        self,
        youtube_url: str,
        info: dict | None = None,
        resume_token: str = "",
        webhook_url: str = "",
        language: str = "",
        video_id: str = "",
    ) -> TranscriptData:
        """Fetch a transcript, prioritizing captions and falling back to audio.

        A ``resume_token`` skips the caption fast path and directly resumes the
        pending Assembly job. Otherwise captions are tried first; when they are
        missing or fail, the remaining parameters are handed to the audio path as
        an :class:`_AudioJob`. ``None`` from the audio provider means audio
        transcription is not configured, so captionless videos fail cleanly
        instead of fabricating data.
        """
        job = _AudioJob(
            youtube_url,
            video_id=video_id,
            resume_token=resume_token,
            webhook_url=webhook_url,
            language=language,
        )
        if job.resume_token:
            return await self._fetch_audio_with_retry(job)

        captions = await self._try_captions(youtube_url, info)
        if captions is not None:
            return captions
        return await self._fetch_audio_with_retry(job)

    async def _try_captions(
        self, youtube_url: str, info: dict | None = None
    ) -> TranscriptData | None:
        """Return the caption fast path result, or ``None`` on a miss.

        Captions are skipped entirely when the live pipeline is off. Missing
        caption tracks are a normal outcome (the caller falls back to audio);
        genuine caption failures are logged and also fall through.
        """
        if not _live_pipeline_enabled(self._resolve_settings()):
            return None
        try:
            captions = await self._resolve_captions_service().fetch(youtube_url, info=info)
        except ExternalServiceError:
            logger.warning(
                "Captions fast-path failed; falling back to audio transcription",
                youtube_url=youtube_url,
                exc_info=True,
            )
            return None
        if captions is None:
            logger.info(
                "No captions available; falling back to audio transcription",
                youtube_url=youtube_url,
            )
            return None
        return captions

    def _require_speech_to_text_leaf(self, resume_token: str) -> TranscriptService:
        """The configured audio leaf, or a clean failure when none is configured.

        Fails the job rather than fabricating data: a captionless video with no
        audio transcription available has no transcript to produce.
        """
        provider = self._speech_to_text_provider(self._settings)
        if provider is not None:
            return provider
        message = (
            "Audio transcription is not configured; cannot resume the job"
            if resume_token
            else "No usable captions and audio transcription is not configured"
        )
        raise ExternalServiceError(message, service="yt-dlp")

    async def _fetch_audio_with_retry(self, job: _AudioJob) -> TranscriptData:
        """Transcribe audio via the configured leaf, retrying transient failures.

        Resolves the leaf, holds the job's audio file for the duration (see
        :class:`~app.providers.audio.AudioFileLease`), and runs the retry budget.
        The audio is downloaded once and reused across attempts; only the
        download is retried, never the transcription. A pending Assembly job
        bubbles up as ``TranscriptJobPending`` re-routed to the ``yt-dlp``
        strategy name so the task resumes it (or ends cleanly when a completion
        webhook was armed).
        """
        provider = self._require_speech_to_text_leaf(job.resume_token)
        settings = self._resolve_settings()
        lease = AudioFileLease(
            job.video_id,
            settings,
            keyed=bool(job.video_id) and _live_pipeline_enabled(settings),
        )
        try:
            return await _run_transcription_attempts(provider, job, lease)
        finally:
            await lease.aclose()


async def _run_transcription_attempts(
    provider: TranscriptService,
    job: _AudioJob,
    lease: AudioFileLease,
) -> TranscriptData:
    """Run the retry budget for one audio job, holding the file across attempts.

    The audio is acquired inside the retried region, so a failed *download* is
    retried while a failed *transcription* re-submits the same file. Every
    parameter is passed to the leaf explicitly — both leaves default each to
    ``""`` and treat that as "not provided" — so the call is checked against
    their signatures rather than assembled by name at runtime.

    The shared audio file is released once the job settles, whether it succeeded
    or the budget ran out (the transcript cache now serves future jobs for this
    video). A pending cloud job returns early and leaves the file in place for
    whatever resumes it.
    """
    last_error: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            audio_path = await lease.acquire(job.youtube_url, resume=bool(job.resume_token))
            transcript = await provider.fetch(
                job.youtube_url,
                resume_token=job.resume_token,
                webhook_url=job.webhook_url,
                language=job.language,
                audio_path=audio_path,
            )
        except TranscriptJobPending as exc:
            raise _strategy_pending(exc) from exc
        except ExternalServiceError as exc:
            last_error = exc
            logger.warning("Transcript fetch attempt failed", attempt=attempt + 1)
            if attempt + 1 < _RETRY_ATTEMPTS:
                await asyncio.sleep(_RETRY_DELAY_SECONDS * (attempt + 1))
            continue
        await lease.release()
        return transcript
    await lease.release()
    assert last_error is not None
    raise last_error


def _build_video_result(media: MediaInfo, transcript: TranscriptData) -> VideoTranscriptResult:
    """Assemble a strategy result from media metadata and a transcript."""
    return VideoTranscriptResult(
        title=media.title,
        author="",
        duration_seconds=media.duration_seconds,
        is_generated=False,
        transcript=transcript,
    )


def _strategy_pending(exc: TranscriptJobPending) -> TranscriptJobPending:
    """Re-route a nested provider's pending job to the strategy-level name.

    Assembly transcription is nested inside the yt-dlp strategy, so a pending
    Assembly job must bubble with ``provider="yt-dlp"`` — the route that
    ``_try_provider`` matches when picking the resume path on retry. Without this
    remap the resume token is dropped and the pipeline re-downloads the same
    audio file on every retry.
    """
    if exc.provider == "yt-dlp":
        return exc
    return TranscriptJobPending(
        message=str(exc),
        provider="yt-dlp",
        resume_token=exc.resume_token,
        resumable=exc.resumable,
        webhook=exc.webhook,
        details=exc.details,
    )


register_self()
