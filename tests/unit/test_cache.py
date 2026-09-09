"""Unit tests for the worker-side Redis transcript cache."""

import pytest
from redis.exceptions import RedisError

import app.providers.local as local_module
from app.providers.base import VideoTranscriptResult
from app.providers.transcript import TranscriptData, TranscriptWordData
from app.storage.cache import (
    TranscriptCache,
    _deserialize_result,
    _serialize_result,
    extract_youtube_video_id,
)

VIDEO_ID = "abc123xyz99"
WATCH_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


class _FakeRedis:
    """Minimal redis client stub backed by an in-memory dict."""

    def __init__(self, store=None) -> None:
        self.store = dict(store or {})
        self.set_calls: list[tuple[str, str, int | None]] = []

    def get(self, key: str):
        return self.store.get(key)

    def set(self, key: str, value: str, ex=None, nx: bool = False) -> bool | None:
        if nx and key in self.store:
            return None
        self.store[key] = value
        self.set_calls.append((key, value, ex))
        return True

    def eval(self, _script: str, _numkeys: int, key: str, owner: str) -> int:
        if self.store.get(key) == owner:
            del self.store[key]
            return 1
        return 0


def _result() -> VideoTranscriptResult:
    return VideoTranscriptResult(
        title="Example title",
        author="Channel",
        duration_seconds=300,
        is_generated=False,
        transcript=TranscriptData(
            language="en",
            text="hello world",
            words=[
                TranscriptWordData(word="hello", start_time=0.0, end_time=1.5),
                TranscriptWordData(word="world", start_time=1.5, end_time=2.0),
            ],
        ),
    )


def _make_cache(monkeypatch, *, enabled: bool = True, ttl: int = 3600, store=None):
    fake = _FakeRedis(store)
    monkeypatch.setattr("app.storage.cache._redis.from_url", lambda *args, **kwargs: fake)
    cache = TranscriptCache("redis://unused:6379/0", ttl, enabled=enabled)
    return cache, fake


class TestExtractYouTubeVideoId:
    def test_watch_url(self) -> None:
        assert extract_youtube_video_id(WATCH_URL) == VIDEO_ID

    def test_short_url(self) -> None:
        assert extract_youtube_video_id(f"https://youtu.be/{VIDEO_ID}") == VIDEO_ID

    def test_shorts_url(self) -> None:
        assert extract_youtube_video_id(f"https://www.youtube.com/shorts/{VIDEO_ID}") == VIDEO_ID

    def test_embed_url(self) -> None:
        assert extract_youtube_video_id(f"https://www.youtube.com/embed/{VIDEO_ID}") == VIDEO_ID

    def test_empty_or_bad_url(self) -> None:
        assert extract_youtube_video_id("") == ""
        assert extract_youtube_video_id("https://www.youtube.com/playlist?list=xyz") == ""


class TestSerialization:
    def test_roundtrip_preserves_result(self) -> None:
        restored = _deserialize_result(_serialize_result(_result()))

        assert restored is not None
        assert restored.title == "Example title"
        assert restored.author == "Channel"
        assert restored.duration_seconds == 300
        assert restored.is_generated is False
        assert restored.transcript.language == "en"
        assert restored.transcript.text == "hello world"
        assert [(w.word, w.start_time, w.end_time) for w in restored.transcript.words] == [
            ("hello", 0.0, 1.5),
            ("world", 1.5, 2.0),
        ]

    def test_deserialize_corrupt_payload_returns_none(self) -> None:
        assert _deserialize_result("{not json") is None
        assert _deserialize_result('{"title": 1}') is None
        assert _deserialize_result("[]") is None


class TestTranscriptCache:
    def test_set_then_get_roundtrip(self, monkeypatch) -> None:
        cache, fake = _make_cache(monkeypatch)

        cache.set(VIDEO_ID, _result())

        assert fake.set_calls[0][0] == f"jumpto:transcript:{VIDEO_ID}"
        assert fake.set_calls[0][2] == 3600  # TTL propagated
        cached = cache.get(VIDEO_ID)
        assert cached is not None and cached.transcript.text == "hello world"

    def test_miss_returns_none(self, monkeypatch) -> None:
        cache, _ = _make_cache(monkeypatch)

        assert cache.get(VIDEO_ID) is None

    def test_disabled_cache_never_reads_or_writes(self, monkeypatch) -> None:
        cache, fake = _make_cache(monkeypatch, enabled=False)

        assert cache.get(VIDEO_ID) is None
        cache.set(VIDEO_ID, _result())
        assert fake.store == {}

    def test_redis_read_error_degrades_to_miss(self, monkeypatch) -> None:
        def boom(*args, **kwargs):
            raise RedisError("connection refused")

        cache, _ = _make_cache(monkeypatch)
        monkeypatch.setattr(cache, "_client", type("Cl", (), {"get": boom})())

        assert cache.get(VIDEO_ID) is None

    def test_redis_write_error_is_best_effort(self, monkeypatch) -> None:
        def boom(*args, **kwargs):
            raise RedisError("connection refused")

        cache, _ = _make_cache(monkeypatch)
        monkeypatch.setattr(cache, "_client", type("Cl", (), {"set": boom})())

        cache.set(VIDEO_ID, _result())  # must not raise

    def test_lock_is_single_flight_and_owner_safe(self, monkeypatch) -> None:
        cache, fake = _make_cache(monkeypatch)

        acquired, owner = cache.acquire_lock(VIDEO_ID, "owner-a")
        assert acquired is True
        acquired_again, other_owner = cache.acquire_lock(VIDEO_ID, "owner-b")
        assert acquired_again is False
        assert other_owner == "owner-b"

        cache.release_lock(VIDEO_ID, "owner-b")
        assert f"jumpto:transcript:lock:{VIDEO_ID}" in fake.store
        cache.release_lock(VIDEO_ID, owner)
        assert f"jumpto:transcript:lock:{VIDEO_ID}" not in fake.store


class TestStrategyCaching:
    """The yt-dlp strategy must consult and populate the cache only when live."""

    def _install_cache(self, monkeypatch, *, enabled: bool = True, store=None):
        cache, fake = _make_cache(monkeypatch, enabled=enabled, store=store)
        monkeypatch.setattr(local_module, "get_transcript_cache", lambda: cache)
        return cache, fake

    @pytest.mark.asyncio
    async def test_cache_hit_skips_yt_dlp(self, monkeypatch) -> None:
        key = f"jumpto:transcript:{VIDEO_ID}"
        cache, _ = self._install_cache(monkeypatch, store={key: _serialize_result(_result())})
        monkeypatch.setattr(local_module, "_live_pipeline_enabled", lambda settings: True)

        media_called = {"n": 0}

        def media_info(video_id, url):
            media_called["n"] += 1
            raise AssertionError("media fetch must not run on cache hit")

        monkeypatch.setattr(local_module, "get_media_info_with_raw", media_info)
        transcript_called = {"n": 0}

        async def fetch_transcript(url, info=None):
            transcript_called["n"] += 1
            raise AssertionError("transcript fetch must not run on cache hit")

        monkeypatch.setattr(local_module, "_fetch_transcript_with_retry", fetch_transcript)

        result = await local_module.YtDlpTranscriptStrategy().fetch(WATCH_URL, VIDEO_ID)

        assert result.transcript.text == "hello world"
        assert media_called["n"] == 0
        assert transcript_called["n"] == 0

    @pytest.mark.asyncio
    async def test_cache_miss_fetches_and_populates(self, monkeypatch, tmp_path) -> None:
        cache, fake = self._install_cache(monkeypatch)
        monkeypatch.setattr(local_module, "_live_pipeline_enabled", lambda settings: True)

        media = type("Media", (), {"title": "T", "duration_seconds": 120})()
        monkeypatch.setattr(
            local_module, "get_media_info_with_raw", lambda video_id, url: (media, {})
        )

        async def fetch_transcript(url, info=None):
            return TranscriptData(language="en", text="cached me", words=[])

        monkeypatch.setattr(local_module, "_fetch_transcript_with_retry", fetch_transcript)

        result = await local_module.YtDlpTranscriptStrategy().fetch(WATCH_URL)

        assert result.transcript.text == "cached me"
        assert f"jumpto:transcript:{VIDEO_ID}" in fake.store  # parsed from URL

    @pytest.mark.asyncio
    async def test_non_live_pipeline_does_not_use_cache(self, monkeypatch) -> None:
        cache, fake = self._install_cache(monkeypatch)
        monkeypatch.setattr(local_module, "_live_pipeline_enabled", lambda settings: False)

        media = type("Media", (), {"title": "T", "duration_seconds": 120})()
        monkeypatch.setattr(
            local_module, "get_media_info_with_raw", lambda video_id, url: (media, None)
        )

        async def fetch_transcript(url, info=None):
            return TranscriptData(language="en", text="fake", words=[])

        monkeypatch.setattr(local_module, "_fetch_transcript_with_retry", fetch_transcript)

        await local_module.YtDlpTranscriptStrategy().fetch(WATCH_URL, VIDEO_ID)

        assert fake.store == {}
