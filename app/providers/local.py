"""Shared plumbing for the worker-local transcript strategies.

Every local strategy runs on the worker itself and carries the same
collaborator seam — injected settings, the audio-transcription leaf factory,
and the worker's disk cache — plus the same cache helpers: a finished
transcript keyed by video id, and the title/duration a resume uses to skip
metadata extraction. This module holds that surface so the concrete strategies
(youtube, drive) only declare what makes them different.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from app.core.config import Settings, _live_pipeline_enabled, get_settings
from app.core.logging import get_logger
from app.providers.base import TranscriptProviderStrategy, VideoTranscriptResult
from app.providers.media import MediaInfo
from app.providers.models import TranscriptData
from app.speech_to_text.base import SpeechToTextProvider
from app.storage.cache import TranscriptCache, get_transcript_cache

logger = get_logger(__name__)


def build_video_result(media: MediaInfo, transcript: TranscriptData) -> VideoTranscriptResult:
    """Assemble a strategy result from media metadata and a transcript."""
    return VideoTranscriptResult(
        title=media.title,
        author="",
        duration_seconds=media.duration_seconds,
        is_generated=False,
        transcript=transcript,
    )


class LocalTranscriptProvider(TranscriptProviderStrategy):
    """Base for the strategies that run on the worker itself.

    ``settings`` is injected by the registry spec builder; when ``None`` the
    global settings resolve lazily per call. ``speech_to_text_provider`` and
    ``cache`` abstract the audio-transcription leaf and the transcript cache;
    when omitted they fall back to the concrete factory — imported lazily so
    :mod:`app.providers` and :mod:`app.speech_to_text` stay independently
    importable — and to the shared cache built by
    :func:`~app.storage.cache.get_transcript_cache`.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        speech_to_text_provider: Callable[..., SpeechToTextProvider | None] | None = None,
        cache: TranscriptCache | None = None,
    ) -> None:
        if speech_to_text_provider is None:
            from app.speech_to_text import get_speech_to_text_provider

            speech_to_text_provider = get_speech_to_text_provider
        self._settings = settings
        self._speech_to_text_provider = speech_to_text_provider
        self._cache = cache

    def _resolve_settings(self) -> Settings:
        return self._settings or get_settings()

    def _usable_cache(self, video_id: str) -> TranscriptCache | None:
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
