"""Local transcript strategy: yt-dlp captions, then Assembly.ai audio.

Unlike the cloud providers, this strategy runs on the worker itself: yt-dlp
downloads the caption track (fast path) or the audio stream, which Assembly.ai
transcribes. It is the terminal fallback — it either produces a transcript or
raises, so a job that reaches it with no result is failed, not completed empty.
"""

import asyncio

from app.core.config import _live_pipeline_enabled, get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.providers.base import TranscriptProviderStrategy, VideoTranscriptResult
from app.providers.cache import extract_youtube_video_id, get_transcript_cache
from app.providers.media import get_media_info_with_raw
from app.providers.transcript import (
    TranscriptData,
    YouTubeCaptionTranscriptProvider,
    get_transcript_provider,
)

logger = get_logger(__name__)

_RETRY_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2


class YtDlpTranscriptStrategy(TranscriptProviderStrategy):
    """Downloads the transcript directly (yt-dlp captions, then Assembly audio)."""

    name = "yt-dlp"
    uses_cloud = False

    async def fetch(
        self,
        youtube_url: str,
        youtube_video_id: str = "",
        resume_token: str = "",
    ) -> VideoTranscriptResult:
        """Fetch media metadata and the best available transcript.

        The caption fast path is tried first when live calls are enabled;
        captionless or failing videos fall back to audio transcription via
        Assembly.ai. ``resume_token`` is accepted for interface uniformity but
        never used (this strategy is synchronous).

        When the live pipeline is on, finished transcripts are cached in Redis
        keyed by the video id, so a later job for the same video skips yt-dlp
        entirely. The id comes from the job when present or is parsed from the
        URL otherwise; unparseable ids simply disable caching.
        """
        video_id = youtube_video_id or extract_youtube_video_id(youtube_url)
        cache = get_transcript_cache() if _live_pipeline_enabled(get_settings()) else None
        if cache is not None and video_id:
            cached = await asyncio.to_thread(cache.get, video_id)
            if cached is not None:
                logger.info("Transcript cache hit", video_id=video_id)
                return cached
        media, info = await asyncio.to_thread(
            get_media_info_with_raw, youtube_video_id, youtube_url
        )
        transcript = await _fetch_transcript_with_retry(youtube_url, info)
        result = VideoTranscriptResult(
            title=media.title,
            author="",
            duration_seconds=media.duration_seconds,
            is_generated=False,
            transcript=transcript,
        )
        if cache is not None and video_id:
            await asyncio.to_thread(cache.set, video_id, result)
        return result


async def _fetch_transcript_with_retry(
    youtube_url: str,
    info: dict | None = None,
) -> TranscriptData:
    """Fetch a transcript, retrying transient external failures."""
    if _live_pipeline_enabled(get_settings()):
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
            return await provider.fetch(youtube_url, info=info)
        except ExternalServiceError as exc:
            last_error = exc
            logger.warning("Transcript fetch attempt failed", attempt=attempt + 1)
            if attempt + 1 < _RETRY_ATTEMPTS:
                await asyncio.sleep(_RETRY_DELAY_SECONDS * (attempt + 1))
    if last_error:
        raise last_error
    return await provider.fetch(youtube_url, info=info)  # pragma: no cover
