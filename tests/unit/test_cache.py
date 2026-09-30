"""Unit tests for the worker-side disk-backed transcript cache."""

import json
from types import SimpleNamespace

import pytest

import app.providers.ytdlp as ytdlp_module
import app.storage.cache as cache_module
from app.providers.base import VideoTranscriptResult
from app.providers.models import TranscriptData, TranscriptWordData
from app.storage.cache import (
    _NAMESPACE,
    DiskBackedTranscriptCache,
    NoOpTranscriptCache,
    _cache_namespace,
    _deserialize_result,
    _serialize_result,
    extract_youtube_video_id,
    get_transcript_cache,
)

VIDEO_ID = "abc123xyz99"
WATCH_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


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


def _disk_cache(tmp_path, *, ttl: int = 3600) -> DiskBackedTranscriptCache:
    return DiskBackedTranscriptCache(
        directory=str(tmp_path / "cache"),
        ttl_seconds=ttl,
    )


# ---------------------------------------------------------------------------
# Video id extraction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


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

    def test_deserialize_ignores_legacy_provider_field(self) -> None:
        """Entries written before the provider field was removed must still load."""
        legacy = json.dumps(
            {
                "title": "Old title",
                "author": "Channel",
                "duration_seconds": 12,
                "is_generated": True,
                "provider": "yt-dlp",
                "transcript": {"language": "en", "text": "legacy", "words": []},
            }
        )

        restored = _deserialize_result(legacy)

        assert restored is not None
        assert restored.title == "Old title"
        assert restored.transcript.text == "legacy"


# ---------------------------------------------------------------------------
# DiskBackedTranscriptCache
# ---------------------------------------------------------------------------


class TestDiskBackedTranscriptCache:
    def test_set_then_get_roundtrip(self, tmp_path) -> None:
        cache = _disk_cache(tmp_path)
        cache.set(VIDEO_ID, _result())

        cached = cache.get(VIDEO_ID)
        assert cached is not None and cached.transcript.text == "hello world"

    def test_namespace_is_prefixed(self, tmp_path) -> None:
        cache = DiskBackedTranscriptCache(
            directory=str(tmp_path / "cache"),
            ttl_seconds=3600,
            namespace="ns:v2:en",
        )
        cache.set(VIDEO_ID, _result())

        # The on-disk key includes the namespace.
        key = f"ns:v2:en:{VIDEO_ID}"
        assert cache._backend.get(key) is not None

    def test_miss_returns_none(self, tmp_path) -> None:
        cache = _disk_cache(tmp_path)
        assert cache.get(VIDEO_ID) is None

    def test_empty_video_id_returns_none(self, tmp_path) -> None:
        cache = _disk_cache(tmp_path)
        assert cache.get("") is None
        cache.set("", _result())  # must not raise

    def test_corrupt_payload_returns_none(self, tmp_path) -> None:
        cache = _disk_cache(tmp_path)
        cache._backend.set(cache._key(VIDEO_ID), "not-valid-json")
        assert cache.get(VIDEO_ID) is None

    def test_disk_write_error_is_best_effort(self, tmp_path) -> None:
        cache = _disk_cache(tmp_path)
        # Force an error by replacing the backend with an object that raises on set.
        cache._backend = type(
            "Fake",
            (),
            {
                "set": staticmethod(lambda *a, **kw: (_ for _ in ()).throw(Exception("boom"))),
                "get": staticmethod(lambda *a, **kw: None),
            },
        )()
        cache.set(VIDEO_ID, _result())  # must not raise

    def test_disk_read_error_degrades_to_miss(self, tmp_path) -> None:
        cache = _disk_cache(tmp_path)
        cache._backend = type(
            "Fake",
            (),
            {"get": staticmethod(lambda *a, **kw: (_ for _ in ()).throw(Exception("boom")))},
        )()
        assert cache.get(VIDEO_ID) is None

    def test_media_set_then_get_roundtrip(self, tmp_path) -> None:
        cache = _disk_cache(tmp_path)
        cache.set_media(VIDEO_ID, "My Video", 120)
        media = cache.get_media(VIDEO_ID)
        assert media is not None
        assert media.title == "My Video"
        assert media.duration_seconds == 120

    def test_media_miss_returns_none(self, tmp_path) -> None:
        cache = _disk_cache(tmp_path)
        assert cache.get_media(VIDEO_ID) is None

    def test_media_empty_video_id_returns_none(self, tmp_path) -> None:
        cache = _disk_cache(tmp_path)
        assert cache.get_media("") is None
        cache.set_media("", "X", 1)  # must not raise

    def test_media_corrupt_entry_returns_none(self, tmp_path) -> None:
        cache = _disk_cache(tmp_path)
        cache._backend.set(cache._media_key(VIDEO_ID), "not-valid-json")
        assert cache.get_media(VIDEO_ID) is None

    def test_media_key_is_separate_from_transcript_key(self, tmp_path) -> None:
        cache = _disk_cache(tmp_path)
        cache.set(VIDEO_ID, _result())
        cache.set_media(VIDEO_ID, "Meta", 50)
        # Both should be readable independently.
        assert cache.get(VIDEO_ID) is not None
        assert cache.get_media(VIDEO_ID) is not None


# ---------------------------------------------------------------------------
# Fail-open semantics — a broken cache backend must never fail a job
# ---------------------------------------------------------------------------


class _ExplodingBackend:
    """A diskcache stand-in that fails every operation."""

    def get(self, key):
        raise OSError("cache volume is gone")

    def set(self, key, payload, expire=None):
        raise OSError("cache volume is read-only")


class TestFailOpenSemantics:
    """The cache is an optimisation: a broken backend degrades to a miss.

    If any of these raised, a full or unmounted cache volume would fail
    transcription jobs outright instead of just costing a re-transcription.
    """

    def test_media_read_fails_open_when_backend_errors(self, tmp_path) -> None:
        cache = _disk_cache(tmp_path)
        cache._backend = _ExplodingBackend()

        assert cache.get_media(VIDEO_ID) is None

    def test_media_write_failure_is_swallowed(self, tmp_path) -> None:
        cache = _disk_cache(tmp_path)
        cache._backend = _ExplodingBackend()

        cache.set_media(VIDEO_ID, "T", 10)  # must not raise


# ---------------------------------------------------------------------------
# Factory — the feature-flag seam
# ---------------------------------------------------------------------------


class TestGetTranscriptCacheFactory:
    """``get_transcript_cache`` is the one place the cache flag is honoured."""

    @staticmethod
    def _settings(tmp_path, *, enabled: bool):
        return SimpleNamespace(
            transcript_cache_enabled=enabled,
            transcript_cache_directory=str(tmp_path / "cache"),
            transcript_cache_ttl_seconds=3600,
        )

    @pytest.fixture(autouse=True)
    def _clear_lru(self):
        get_transcript_cache.cache_clear()
        yield
        get_transcript_cache.cache_clear()

    def test_disabled_returns_noop_cache(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            cache_module, "get_settings", lambda: self._settings(tmp_path, enabled=False)
        )

        cache = get_transcript_cache()

        assert isinstance(cache, NoOpTranscriptCache)
        assert cache.get(VIDEO_ID) is None

    def test_enabled_returns_disk_backed_cache(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            cache_module, "get_settings", lambda: self._settings(tmp_path, enabled=True)
        )

        cache = get_transcript_cache()

        assert isinstance(cache, DiskBackedTranscriptCache)
        assert cache.ttl_seconds == 3600

    def test_namespace_is_versioned_and_settings_derived(self, monkeypatch, tmp_path) -> None:
        """Changing a provider's cache-affecting setting must change the key space.

        The namespace is built from each spec's declared ``cache_config_fields``
        rather than a hard-coded list, so this also proves the registry-driven
        part of the contract.
        """
        monkeypatch.setattr(
            cache_module, "get_settings", lambda: self._settings(tmp_path, enabled=False)
        )
        first = _cache_namespace(self._settings(tmp_path, enabled=False))

        monkeypatch.setattr(
            cache_module, "get_settings", lambda: self._settings(tmp_path, enabled=True)
        )
        second = _cache_namespace(self._settings(tmp_path, enabled=True))

        assert first == second
        assert first.startswith(f"{_NAMESPACE}:v3:")
        assert get_transcript_cache().namespace == first

    def test_namespace_changes_when_cache_affecting_setting_changes(self, tmp_path) -> None:
        base = self._settings(tmp_path, enabled=True)
        other = SimpleNamespace(**{**vars(base), "speech_to_text_provider": "deepgram"})

        assert _cache_namespace(base) != _cache_namespace(other)


# ---------------------------------------------------------------------------
# NoOpTranscriptCache
# ---------------------------------------------------------------------------


class TestNoOpTranscriptCache:
    def test_get_always_misses(self) -> None:
        cache = NoOpTranscriptCache()
        assert cache.get(VIDEO_ID) is None

    def test_get_media_always_misses(self) -> None:
        cache = NoOpTranscriptCache()
        assert cache.get_media(VIDEO_ID) is None

    def test_set_does_not_raise(self) -> None:
        cache = NoOpTranscriptCache()
        cache.set(VIDEO_ID, _result())

    def test_set_media_does_not_raise(self) -> None:
        cache = NoOpTranscriptCache()
        cache.set_media(VIDEO_ID, "X", 1)


# ---------------------------------------------------------------------------
# Strategy integration — provider uses injected cache; live-gate respected.
# ---------------------------------------------------------------------------


class TestStrategyCaching:
    """The yt-dlp strategy must consult and populate the cache only when live."""

    @staticmethod
    def _provider(cache, *, live_calls: bool) -> ytdlp_module.YtDlpTranscriptProvider:
        return ytdlp_module.YtDlpTranscriptProvider(
            settings=SimpleNamespace(live_external_calls=live_calls),
            cache=cache,
        )

    @pytest.mark.asyncio
    async def test_cache_hit_skips_yt_dlp(self, tmp_path, monkeypatch) -> None:
        cache = _disk_cache(tmp_path)
        cache.set(VIDEO_ID, _result())

        media_called = {"n": 0}

        def media_info(video_id, url, settings=None):
            media_called["n"] += 1
            raise AssertionError("media fetch must not run on cache hit")

        monkeypatch.setattr(ytdlp_module, "get_media_info_with_raw", media_info)
        transcript_called = {"n": 0}

        async def fetch_transcript(
            url, info=None, resume_token="", webhook_url="", language="", video_id=""
        ):
            transcript_called["n"] += 1
            raise AssertionError("transcript fetch must not run on cache hit")

        provider = self._provider(cache, live_calls=True)
        monkeypatch.setattr(provider, "_fetch_transcript_with_retry", fetch_transcript)

        result = await provider.fetch(WATCH_URL, VIDEO_ID)

        assert result.transcript.text == "hello world"
        assert media_called["n"] == 0
        assert transcript_called["n"] == 0

    @pytest.mark.asyncio
    async def test_cache_miss_fetches_and_populates(self, tmp_path, monkeypatch) -> None:
        cache = _disk_cache(tmp_path)

        media = type("Media", (), {"title": "T", "duration_seconds": 120})()
        monkeypatch.setattr(
            ytdlp_module,
            "get_media_info_with_raw",
            lambda video_id, url, settings=None: (media, {}),
        )

        async def fetch_transcript(
            url, info=None, resume_token="", webhook_url="", language="", video_id=""
        ):
            return TranscriptData(language="en", text="cached me", words=[])

        provider = self._provider(cache, live_calls=True)
        monkeypatch.setattr(provider, "_fetch_transcript_with_retry", fetch_transcript)

        result = await provider.fetch(WATCH_URL)

        assert result.transcript.text == "cached me"
        assert cache.get(VIDEO_ID) is not None  # parsed from URL and stored

    @pytest.mark.asyncio
    async def test_non_live_pipeline_does_not_use_cache(self, tmp_path, monkeypatch) -> None:
        cache = _disk_cache(tmp_path)

        media = type("Media", (), {"title": "T", "duration_seconds": 120})()
        monkeypatch.setattr(
            ytdlp_module,
            "get_media_info_with_raw",
            lambda video_id, url, settings=None: (media, None),
        )

        async def fetch_transcript(
            url, info=None, resume_token="", webhook_url="", language="", video_id=""
        ):
            return TranscriptData(language="en", text="fake", words=[])

        provider = self._provider(cache, live_calls=False)
        monkeypatch.setattr(provider, "_fetch_transcript_with_retry", fetch_transcript)

        await provider.fetch(WATCH_URL, VIDEO_ID)

        assert cache.get(VIDEO_ID) is None  # cache never populated

    @pytest.mark.asyncio
    async def test_resume_skips_metadata_fetch_when_media_cached(
        self, tmp_path, monkeypatch
    ) -> None:
        """On resume with a metadata cache hit, get_media_info_with_raw must not run."""
        cache = _disk_cache(tmp_path)
        cache.set_media(VIDEO_ID, "Cached Title", 99)

        media_called = {"n": 0}

        def media_info(video_id, url, settings=None):
            media_called["n"] += 1
            raise AssertionError("get_media_info_with_raw must not run on resume")

        monkeypatch.setattr(ytdlp_module, "get_media_info_with_raw", media_info)

        async def fetch_transcript(
            url, info=None, resume_token="", webhook_url="", language="", video_id=""
        ):
            assert resume_token == "asm-123"
            assert info is None  # info not needed — captions are skipped on resume
            return TranscriptData(language="en", text="resumed transcript", words=[])

        provider = self._provider(cache, live_calls=True)
        monkeypatch.setattr(provider, "_fetch_transcript_with_retry", fetch_transcript)

        result = await provider.fetch(WATCH_URL, VIDEO_ID, resume_token="asm-123")

        assert result.title == "Cached Title"
        assert result.duration_seconds == 99
        assert result.transcript.text == "resumed transcript"
        assert media_called["n"] == 0

    @pytest.mark.asyncio
    async def test_resume_falls_back_to_metadata_fetch_on_cache_miss(
        self, tmp_path, monkeypatch
    ) -> None:
        """On resume with a metadata cache miss, get_media_info_with_raw runs."""
        cache = _disk_cache(tmp_path)

        media = type("Media", (), {"title": "Fresh Title", "duration_seconds": 60})()
        monkeypatch.setattr(
            ytdlp_module,
            "get_media_info_with_raw",
            lambda video_id, url, settings=None: (media, {}),
        )

        async def fetch_transcript(
            url, info=None, resume_token="", webhook_url="", language="", video_id=""
        ):
            assert resume_token == "asm-456"
            return TranscriptData(language="en", text="fresh transcript", words=[])

        provider = self._provider(cache, live_calls=True)
        monkeypatch.setattr(provider, "_fetch_transcript_with_retry", fetch_transcript)

        result = await provider.fetch(WATCH_URL, VIDEO_ID, resume_token="asm-456")

        assert result.title == "Fresh Title"
        assert result.transcript.text == "fresh transcript"
        # Metadata should now be cached for a future resume.
        cached_media = cache.get_media(VIDEO_ID)
        assert cached_media is not None
        assert cached_media.title == "Fresh Title"
