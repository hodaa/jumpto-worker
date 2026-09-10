"""Local transcript strategy: yt-dlp captions, then Assembly.ai audio.

Unlike the cloud providers, this strategy runs on the worker itself: yt-dlp
downloads the caption track (fast path) or the audio stream, which Assembly.ai
transcribes. It is the terminal fallback — it either produces a transcript or
raises, so a job that reaches it with no result is failed, not completed empty.
"""

import asyncio
from collections.abc import Callable

from app.core.config import Settings, _live_pipeline_enabled, get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.providers.assembly import get_transcript_provider
from app.providers.base import TranscriptProviderStrategy, TranscriptService, VideoTranscriptResult
from app.providers.media import MediaInfo, get_media_info_with_raw
from app.providers.models import TranscriptData, TranscriptJobPending
from app.providers.registry import TranscriptProviderSpec, provider_spec, register_provider
from app.providers.transcript import YouTubeCaptionTranscriptService
from app.storage.cache import TranscriptCache, extract_youtube_video_id, get_transcript_cache

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
            description="Local yt-dlp captions with Assembly.ai audio fallback (free, first provider).",
            uses_cloud=False,
        )
    )


_RETRY_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2


class YtDlpTranscriptProvider(TranscriptProviderStrategy):
    """Downloads the transcript directly (yt-dlp captions, then Assembly audio)."""

    name = "yt-dlp"
    supports_resume = True

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        captions_service: YouTubeCaptionTranscriptService | None = None,
        assembly_provider: Callable[..., TranscriptService | None] | None = None,
        cache: TranscriptCache | None = None,
    ) -> None:
        """Build the strategy, optionally wiring collaborators explicitly.

        ``settings`` is injected by the registry spec builder; when ``None`` the
        global settings resolve lazily per call so a plain
        ``YtDlpTranscriptProvider()`` keeps working. ``captions_service``,
        ``assembly_provider``, and ``cache`` abstract the leaf collaborators
        (caption fast path, Assembly audio, and the transcript cache); when
        omitted they fall back to the concrete service, the Assembly factory,
        and the shared cache, preserving current behavior.
        """
        self._settings = settings
        self._captions_service = captions_service
        self._assembly_provider = assembly_provider or get_transcript_provider
        self._cache = cache

    def _resolve_settings(self) -> Settings:
        return self._settings or get_settings()

    def _resolve_captions_service(self) -> YouTubeCaptionTranscriptService:
        return self._captions_service or YouTubeCaptionTranscriptService()

    async def fetch(
        self,
        youtube_url: str,
        youtube_video_id: str = "",
        resume_token: str = "",
    ) -> VideoTranscriptResult:
        """Fetch media metadata and the best available transcript.

        The caption fast path is tried first; captionless or failing videos
        fall back to audio transcription via Assembly.ai. ``resume_token``
        resumes a previously queued Assembly job so a retry re-polls it instead
        of re-downloading the audio.

        When the live pipeline is on, finished transcripts are cached in Redis
        keyed by the video id, so a later job for the same video skips yt-dlp
        entirely. The id comes from the job when present or is parsed from the
        URL otherwise; unparseable ids simply disable caching.
        """
        video_id = youtube_video_id or extract_youtube_video_id(youtube_url)
        cached = await self._read_cache(video_id)
        if cached is not None:
            return cached

        media, info = await asyncio.to_thread(
            get_media_info_with_raw, youtube_video_id, youtube_url
        )
        if resume_token:
            transcript = await self._fetch_transcript_with_retry(
                youtube_url, info, resume_token=resume_token
            )
        else:
            transcript = await self._fetch_transcript_with_retry(youtube_url, info)
        result = _build_video_result(media, transcript)

        await self._write_cache(video_id, result)
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

    async def _fetch_transcript_with_retry(
        self,
        youtube_url: str,
        info: dict | None = None,
        resume_token: str = "",
    ) -> TranscriptData:
        """Fetch a transcript, prioritizing captions and falling back to audio.

        A ``resume_token`` skips the caption fast path and directly resumes the
        pending Assembly job. Otherwise captions are tried first; when they are
        missing or fail, audio transcription via Assembly.ai runs with retry.
        ``None`` from the assembly provider means audio transcription is not
        configured, so captionless videos fail cleanly instead of fabricating
        data.
        """
        if resume_token:
            return await self._fetch_audio_with_retry(youtube_url, resume_token=resume_token)

        captions = await self._try_captions(youtube_url, info)
        if captions is not None:
            return captions
        return await self._fetch_audio_with_retry(youtube_url)

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

    async def _fetch_audio_with_retry(
        self,
        youtube_url: str,
        resume_token: str = "",
    ) -> TranscriptData:
        """Transcribe audio via Assembly.ai, retrying transient failures.

        A pending Assembly job bubbles up as ``TranscriptJobPending`` re-routed
        to the ``yt-dlp`` strategy name so the task resumes it. Uses the
        provider only when audio transcription is configured; otherwise the job
        fails cleanly.
        """
        provider = self._assembly_provider(self._settings)
        if provider is None:
            message = (
                "Audio transcription is not configured; cannot resume the job"
                if resume_token
                else "No usable captions and audio transcription is not configured"
            )
            raise ExternalServiceError(message, service="yt-dlp")

        last_error: Exception | None = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                kwargs: dict = {}
                if resume_token:
                    kwargs["resume_token"] = resume_token
                return await provider.fetch(youtube_url, **kwargs)
            except TranscriptJobPending as exc:
                raise _strategy_pending(exc) from exc
            except ExternalServiceError as exc:
                last_error = exc
                logger.warning("Transcript fetch attempt failed", attempt=attempt + 1)
                if attempt + 1 < _RETRY_ATTEMPTS:
                    await asyncio.sleep(_RETRY_DELAY_SECONDS * (attempt + 1))
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
        details=exc.details,
    )


register_self()
