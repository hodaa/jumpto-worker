"""Worker-side transcript cache.

Reprocessing the same YouTube video re-runs yt-dlp end-to-end (metadata
extraction + caption/audio download). The cache stores the finished
:class:`VideoTranscriptResult` keyed by YouTube video id in Redis — which is
already running as the Celery broker — so a second job for the same video is
served without any network call.

Failures degrade gracefully: a Redis outage, a corrupt entry, or an empty
video id all behave as a cache miss, and writes are best-effort. The frontend's
in-memory cache only helps one browser session; this one is shared and durable.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

try:
    import redis as _redis
    from redis.exceptions import RedisError as _RedisError
except ImportError:  # pragma: no cover - redis is a hard dependency
    _redis = None
    _RedisError = Exception

from app.core.config import get_settings
from app.core.logging import get_logger
from app.providers.base import VideoTranscriptResult
from app.providers.transcript import TranscriptData, TranscriptWordData

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


def _cache_key(video_id: str) -> str:
    return f"{_NAMESPACE}:{video_id}"


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
    """Rebuild a result from stored JSON, or None for a corrupt entry."""
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


class TranscriptCache:
    """Redis-backed transcript cache with a TTL and fail-open semantics."""

    def __init__(self, redis_url: str, ttl_seconds: int, enabled: bool = True) -> None:
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        if enabled and _redis is not None:
            # Short timeouts so a dead broker never stalls a transcription job.
            self._client = _redis.from_url(
                redis_url, decode_responses=True, socket_connect_timeout=1, socket_timeout=1
            )
        else:
            self._client = None

    def get(self, video_id: str) -> VideoTranscriptResult | None:
        if not self.enabled or self._client is None or not video_id:
            return None
        key = _cache_key(video_id)
        try:
            payload = self._client.get(key)
        except _RedisError as exc:
            logger.warning("Transcript cache read failed", key=key, error=str(exc))
            return None
        if not payload:
            return None
        return _deserialize_result(payload)

    def set(self, video_id: str, result: VideoTranscriptResult) -> None:
        if not self.enabled or self._client is None or not video_id:
            return
        key = _cache_key(video_id)
        try:
            self._client.set(key, _serialize_result(result), ex=self.ttl_seconds)
        except _RedisError as exc:
            logger.warning("Transcript cache write failed", key=key, error=str(exc))


@lru_cache
def get_transcript_cache() -> TranscriptCache:
    """Return the shared cache instance built from the current settings."""
    settings = get_settings()
    return TranscriptCache(
        redis_url=settings.redis_url,
        ttl_seconds=settings.transcript_cache_ttl_seconds,
        enabled=settings.transcript_cache_enabled,
    )
