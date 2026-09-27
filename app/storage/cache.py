"""Worker-side transcript cache.

Reprocessing the same YouTube video re-runs yt-dlp end-to-end (metadata
extraction + caption/audio download).  The cache stores the finished
:class:`VideoTranscriptResult` keyed by YouTube video id on the local disk
via :mod:`diskcache` — no separate service needed — so a second job for the
same video is served without any network call.

When ``TRANSCRIPT_CACHE_ENABLED=false`` the factory returns
:class:`NoOpTranscriptCache`, which satisfies the same interface without touching
the filesystem.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from functools import lru_cache

from diskcache import Cache as _DiskCache

from app.core.config import get_settings
from app.core.logging import get_logger
from app.providers.base import VideoTranscriptResult
from app.providers.media import MediaInfo
from app.providers.models import TranscriptData, TranscriptWordData

logger = get_logger(__name__)

_NAMESPACE = "jumpto:transcript"

# YouTube video id extraction from common URL shapes (watch, youtu.be, shorts,
# embed). The id is 11 chars of [A-Za-z0-9_-].
_VIDEO_ID_PATTERNS = (
    re.compile(r"[?&]v=([A-Za-z0-9_-]{6,})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{6,})"),
    re.compile(r"/(?:shorts|embed|live)/([A-Za-z0-9_-]{6,})"),
)


def extract_youtube_video_id(youtube_url: str) -> str:
    """Best-effort YouTube video id from a URL, or '' when it cannot be parsed."""
    if not youtube_url:
        return ""
    for pattern in _VIDEO_ID_PATTERNS:
        match = pattern.search(youtube_url)
        if match:
            return match.group(1)
    return ""


def _cache_namespace(settings) -> str:
    """Build a versioned cache namespace from each spec's declared cache fields.

    Every provider spec declares which settings knobs affect the transcript it
    returns (language, mode, etc.) via ``cache_config_fields``. The namespace
    iterates specs in their canonical priority order and reads those attributes,
    so adding a new provider's cache-affecting settings never requires editing
    the cache module itself.
    """
    from app.providers.registry import ordered_specs

    values = []
    for spec in ordered_specs():
        for name in spec.cache_config_fields:
            values.append(getattr(settings, name, "") or "")
    suffix = ":".join(re.sub(r"[^A-Za-z0-9_.-]", "_", str(v)) for v in values)
    return f"{_NAMESPACE}:v3:{suffix}"


def _serialize_result(result: VideoTranscriptResult) -> str:
    """Flatten a result to compact JSON (word/float tuples survive losslessly)."""
    return json.dumps(
        {
            "title": result.title,
            "author": result.author,
            "duration_seconds": result.duration_seconds,
            "is_generated": result.is_generated,
            "transcript": {
                "language": result.transcript.language,
                "text": result.transcript.text,
                "words": [[w.word, w.start_time, w.end_time] for w in result.transcript.words],
            },
        },
        separators=(",", ":"),
    )


def _deserialize_result(payload: str) -> VideoTranscriptResult | None:
    """Rebuild a result from JSON, or None on a corrupt entry."""
    try:
        data = json.loads(payload)
        raw_words = data["transcript"]["words"]
        transcript = TranscriptData(
            language=data["transcript"]["language"],
            text=data["transcript"]["text"],
            words=[
                TranscriptWordData(word=word, start_time=start, end_time=end)
                for word, start, end in raw_words
            ],
        )
        return VideoTranscriptResult(
            title=data["title"],
            author=data.get("author", ""),
            duration_seconds=data.get("duration_seconds", 0),
            is_generated=data.get("is_generated", False),
            transcript=transcript,
        )
    except (KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError) as exc:
        logger.warning("Discarding corrupt transcript cache entry", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# TranscriptCache — abstract interface (Strategy / Interface Segregation)
# ---------------------------------------------------------------------------


class TranscriptCache(ABC):
    """Abstract worker-side transcript cache.

    Concrete implementations (:class:`DiskBackedTranscriptCache`,
    :class:`NoOpTranscriptCache`) are selected by :func:`get_transcript_cache`
    based on configuration.

    The interface is exactly what the one client — the yt-dlp composite — calls:
    :meth:`get`, :meth:`set`, and the optional metadata pair
    (:meth:`get_media` / :meth:`set_media`, which default to a quiet no-op so a
    cache that cannot store metadata still satisfies the interface).
    """

    ttl_seconds: int
    namespace: str

    @abstractmethod
    def get(self, video_id: str) -> VideoTranscriptResult | None: ...

    @abstractmethod
    def set(self, video_id: str, result: VideoTranscriptResult) -> None: ...

    def get_media(self, video_id: str) -> MediaInfo | None:
        """Read cached media metadata (title/duration), or ``None`` on a miss.

        Metadata is cached ahead of Assembly submission so a resume of a
        pending job can skip the full yt-dlp metadata re-extract and go
        straight to the Assembly poll. Default no-op: caches that don't
        support metadata return a quiet miss.
        """
        return None

    def set_media(self, video_id: str, title: str, duration_seconds: int) -> None:
        """Store media metadata so a later resume can skip the re-extract.

        Default no-op; overridden by disk-backed caches.
        """
        return


# ---------------------------------------------------------------------------
# NoOpTranscriptCache — satisfied interface that never touches I/O
# ---------------------------------------------------------------------------


class NoOpTranscriptCache(TranscriptCache):
    """Cache disabled or unavailable; every operation is a silent no-op."""

    def __init__(
        self,
        ttl_seconds: int = 0,
        namespace: str = _NAMESPACE,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.namespace = namespace

    def get(self, video_id: str) -> VideoTranscriptResult | None:
        return None

    def set(self, video_id: str, result: VideoTranscriptResult) -> None:
        return


# ---------------------------------------------------------------------------
# DiskBackedTranscriptCache — SQLite + filesystem via diskcache
# ---------------------------------------------------------------------------


class DiskBackedTranscriptCache(TranscriptCache):
    """Disk-backed transcript cache with a TTL and fail-open semantics.

    Uses :class:`diskcache.Cache` (SQLite + filesystem) so cached transcripts
    survive process restarts, are shared across prefork workers, and need no
    external service.
    """

    def __init__(
        self,
        directory: str,
        ttl_seconds: int,
        namespace: str = _NAMESPACE,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.namespace = namespace
        # Short timeout so a locked database never stalls a transcription job.
        self._backend = _DiskCache(directory, timeout=0.1)

    # -- key helpers --------------------------------------------------------

    def _key(self, video_id: str) -> str:
        return f"{self.namespace}:{video_id}"

    def _media_key(self, video_id: str) -> str:
        return f"{self.namespace}:media:{video_id}"

    # -- public API ---------------------------------------------------------

    def get(self, video_id: str) -> VideoTranscriptResult | None:
        """Read a cached transcript, or ``None`` on a miss or a corrupt entry."""
        if not video_id:
            return None
        key = self._key(video_id)
        try:
            payload = self._backend.get(key)
        except Exception as exc:  # noqa: BLE001 – fail-open
            logger.warning("Transcript cache read failed", key=key, error=str(exc))
            return None
        if payload is None:
            return None
        return _deserialize_result(payload)

    def set(self, video_id: str, result: VideoTranscriptResult) -> None:
        """Store a finished transcript in the cache, best-effort."""
        if not video_id:
            return
        key = self._key(video_id)
        try:
            self._backend.set(key, _serialize_result(result), expire=self.ttl_seconds)
        except Exception as exc:  # noqa: BLE001 – fail-open
            logger.warning("Transcript cache write failed", key=key, error=str(exc))

    def get_media(self, video_id: str) -> MediaInfo | None:
        """Read cached media metadata (title/duration), or ``None`` on a miss.

        A corrupt or expired entry is treated as a miss (fail-open), so a
        resume falls back to the full metadata fetch rather than failing.
        """
        if not video_id:
            return None
        key = self._media_key(video_id)
        try:
            payload = self._backend.get(key)
        except Exception as exc:  # noqa: BLE001 – fail-open
            logger.warning("Media metadata cache read failed", key=key, error=str(exc))
            return None
        if payload is None:
            return None
        try:
            data = json.loads(payload)
            return MediaInfo(
                title=str(data["title"]),
                duration_seconds=int(data["duration_seconds"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Discarding corrupt media metadata cache entry", key=key, error=str(exc))
            return None

    def set_media(self, video_id: str, title: str, duration_seconds: int) -> None:
        """Store media metadata so a later resume can skip the re-extract."""
        if not video_id:
            return
        key = self._media_key(video_id)
        try:
            payload = json.dumps(
                {"title": title, "duration_seconds": int(duration_seconds)},
                separators=(",", ":"),
            )
            self._backend.set(key, payload, expire=self.ttl_seconds)
        except Exception as exc:  # noqa: BLE001 – fail-open
            logger.warning("Media metadata cache write failed", key=key, error=str(exc))


# ---------------------------------------------------------------------------
# Factory — single enforcement point for the feature-flag seam
# ---------------------------------------------------------------------------


@lru_cache
def get_transcript_cache() -> TranscriptCache:
    """Return the shared cache instance built from the current settings.

    When ``TRANSCRIPT_CACHE_ENABLED=false`` a :class:`NoOpTranscriptCache` is
    returned so that callers never branch on availability.
    """
    settings = get_settings()
    if not settings.transcript_cache_enabled:
        return NoOpTranscriptCache(
            ttl_seconds=settings.transcript_cache_ttl_seconds,
            namespace=_cache_namespace(settings),
        )
    return DiskBackedTranscriptCache(
        directory=settings.transcript_cache_directory,
        ttl_seconds=settings.transcript_cache_ttl_seconds,
        namespace=_cache_namespace(settings),
    )
