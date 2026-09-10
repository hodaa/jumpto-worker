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
import uuid
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
from app.providers.models import TranscriptData, TranscriptWordData

logger = get_logger(__name__)

_NAMESPACE = "jumpto:transcript"
_DEFAULT_LOCK_TTL_SECONDS = 900

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
    """Build a versioned cache namespace from transcript-affecting settings."""
    values = (
        getattr(settings, "transcriptfetch_lang", "en"),
        getattr(settings, "transcriptfetch_mode", "auto"),
        getattr(settings, "supadata_lang", "en"),
        getattr(settings, "supadata_mode", "auto"),
        getattr(settings, "vidwords_lang", "en"),
    )
    suffix = ":".join(re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or "")) for value in values)
    return f"{_NAMESPACE}:v3:{suffix}"


def _serialize_result(result: VideoTranscriptResult, provider: str = "") -> str:
    """Flatten a result to compact JSON (word/float tuples survive losslessly)."""
    return json.dumps(
        {
            "title": result.title,
            "author": result.author,
            "duration_seconds": result.duration_seconds,
            "is_generated": result.is_generated,
            "provider": provider,
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

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int,
        enabled: bool = True,
        lock_ttl_seconds: int = _DEFAULT_LOCK_TTL_SECONDS,
        namespace: str = _NAMESPACE,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.lock_ttl_seconds = lock_ttl_seconds
        self.namespace = namespace
        self.enabled = enabled
        if enabled and _redis is not None:
            # Short timeouts so a dead broker never stalls a transcription job.
            self._client = _redis.from_url(
                redis_url, decode_responses=True, socket_connect_timeout=1, socket_timeout=1
            )
        else:
            self._client = None

    def _key(self, video_id: str) -> str:
        return f"{self.namespace}:{video_id}"

    def _lock_key(self, video_id: str) -> str:
        return f"{self.namespace}:lock:{video_id}"

    def get(self, video_id: str) -> VideoTranscriptResult | None:
        """Read a cached transcript, ignoring any stored provenance."""
        entry = self.get_with_provider(video_id)
        return entry[0] if entry is not None else None

    def get_with_provider(self, video_id: str) -> tuple[VideoTranscriptResult, str] | None:
        """Read a cached transcript and the provider that produced it."""
        if not self.enabled or self._client is None or not video_id:
            return None
        key = self._key(video_id)
        try:
            payload = self._client.get(key)
        except _RedisError as exc:
            logger.warning("Transcript cache read failed", key=key, error=str(exc))
            return None
        if not payload:
            return None
        result = _deserialize_result(payload)
        if result is None:
            return None
        try:
            provider = str(json.loads(payload).get("provider") or "")
        except (TypeError, ValueError, json.JSONDecodeError):
            provider = ""
        return result, provider

    def set(self, video_id: str, result: VideoTranscriptResult, provider: str = "") -> None:
        if not self.enabled or self._client is None or not video_id:
            return
        key = self._key(video_id)
        try:
            self._client.set(key, _serialize_result(result, provider), ex=self.ttl_seconds)
        except _RedisError as exc:
            logger.warning("Transcript cache write failed", key=key, error=str(exc))

    def acquire_lock(self, video_id: str, owner: str | None = None) -> tuple[bool, str]:
        """Acquire a short-lived distributed lock for a video cache miss.

        Redis failures fail open: duplicate work is preferable to blocking every
        transcription when the optional cache is unavailable.
        """
        owner = owner or uuid.uuid4().hex
        if not self.enabled or self._client is None or not video_id:
            return True, owner
        key = self._lock_key(video_id)
        try:
            acquired = self._client.set(
                key, owner, ex=self.lock_ttl_seconds, nx=True
            )
            return bool(acquired), owner
        except (_RedisError, TypeError) as exc:
            logger.warning("Transcript cache lock failed open", key=key, error=str(exc))
            return True, owner

    def release_lock(self, video_id: str, owner: str) -> None:
        """Release a lock only when it is still owned by this worker."""
        if not self.enabled or self._client is None or not video_id or not owner:
            return
        key = self._lock_key(video_id)
        try:
            # Compare-and-delete prevents an expired lock from being deleted
            # after another worker has acquired it.
            self._client.eval(
                "if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('del', KEYS[1]) else return 0 end",
                1,
                key,
                owner,
            )
        except (_RedisError, AttributeError) as exc:
            logger.warning("Transcript cache lock release failed", key=key, error=str(exc))


@lru_cache
def get_transcript_cache() -> TranscriptCache:
    """Return the shared cache instance built from the current settings."""
    settings = get_settings()
    return TranscriptCache(
        redis_url=settings.redis_url,
        ttl_seconds=settings.transcript_cache_ttl_seconds,
        enabled=settings.transcript_cache_enabled,
        lock_ttl_seconds=max(
            getattr(settings, "transcript_cache_lock_ttl_seconds", 900),
            getattr(settings, "job_timeout_seconds", 600) + 60,
        ),
        namespace=_cache_namespace(settings),
    )
