"""Unit tests for the captionless-source (Drive) transcript strategy."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import ExternalServiceError, PermanentExternalServiceError
from app.providers import audio_jobs as audio_jobs_module
from app.providers import drive as drive_module
from app.providers.media import MediaInfo
from app.providers.models import TranscriptData, TranscriptJobPending
from app.sources.base import PreparedAudio, VideoRef

_DRIVE_ID = "1DOoJhC6NHpLSu2nAvaWy3HukVimEyyhE"
_DRIVE_URL = f"https://drive.google.com/file/d/{_DRIVE_ID}/view"


def _settings(*, live_calls: bool = True, stt_language: str = "") -> SimpleNamespace:
    """Build a minimal settings object for the strategy."""
    return SimpleNamespace(
        live_external_calls=live_calls,
        drive_stt_language=stt_language,
    )


def _transcript() -> TranscriptData:
    """Build a fake transcript the leaf can return."""
    return TranscriptData(language="en", text="Hello, world!", words=[])


class _FakeTranscriptCache:
    """Cache double returning canned media and recording transcript writes."""

    def __init__(self, media: MediaInfo | None = None) -> None:
        self.media = media
        self.results: list[tuple[str, object]] = []
        self.media_writes: list[tuple[str, str, int]] = []

    def get(self, video_id: str):
        return None

    def set(self, video_id: str, result) -> None:
        self.results.append((video_id, result))

    def get_media(self, video_id: str) -> MediaInfo | None:
        return self.media

    def set_media(self, video_id: str, title: str, duration_seconds: int) -> None:
        self.media = MediaInfo(title, duration_seconds)
        self.media_writes.append((video_id, title, duration_seconds))


class TestDriveTranscriptProvider:
    """Captionless sources never touch yt-dlp or metadata endpoints."""

    @staticmethod
    def _drive_source(prepare_audio, *, native_captions: bool = False):
        """A Drive-shaped source double whose identify/parse are trivial."""
        return SimpleNamespace(
            name="drive",
            native_captions=native_captions,
            identify=lambda url: VideoRef(source="drive", source_id=_DRIVE_ID, canonical_url=url),
            prepare_audio=prepare_audio,
        )

    @staticmethod
    @staticmethod
    def _provider(speech=None, cache=None, *, live_calls=True, stt_language=""):
        return drive_module.DriveTranscriptProvider(
            settings=_settings(live_calls=live_calls, stt_language=stt_language),
            speech_to_text_provider=speech,
            cache=cache or _FakeTranscriptCache(),
        )

    @staticmethod
    def _prepared(directory, *, title="Lec 2", duration=3600) -> PreparedAudio:
        """A prepared workspace whose audio file lives in ``directory``."""
        audio = Path(directory) / "audio.m4a"
        audio.write_bytes(b"audio")
        return PreparedAudio(
            title=title,
            duration_seconds=duration,
            audio_path=audio,
            directory=Path(directory),
        )

    @pytest.mark.asyncio
    async def test_skips_metadata_for_a_drive_url(self, monkeypatch) -> None:
        calls: list[tuple[str, object]] = []
        prepared: list[PreparedAudio] = []

        def prepare_audio(url, directory, settings=None):
            calls.append((url, settings))
            workspace = self._prepared(directory)
            prepared.append(workspace)
            return workspace

        source = self._drive_source(prepare_audio)
        monkeypatch.setattr(drive_module, "resolve_source", lambda url: source)
        leaf = AsyncMock()
        leaf.fetch.return_value = _transcript()
        provider = self._provider(speech=lambda settings: leaf)

        result = await provider.fetch(_DRIVE_URL, _DRIVE_ID)

        assert result.title == "Lec 2"
        assert result.duration_seconds == 3600
        assert calls == [(_DRIVE_URL, provider._resolve_settings())]
        leaf.fetch.assert_awaited_once_with(
            _DRIVE_URL,
            resume_token="",
            webhook_url="",
            language="en",
            audio_path=str(prepared[0].audio_path),
        )

    @pytest.mark.asyncio
    async def test_arabic_title_selects_the_arabic_language(self, monkeypatch, tmp_path) -> None:
        source = self._drive_source(
            lambda url, directory, settings=None: self._prepared(directory, title="محاضرة 2")
        )
        monkeypatch.setattr(drive_module, "resolve_source", lambda url: source)
        leaf = AsyncMock()
        leaf.fetch.return_value = _transcript()
        provider = self._provider(speech=lambda settings: leaf)

        await provider.fetch(_DRIVE_URL, _DRIVE_ID)

        assert leaf.fetch.await_args.kwargs["language"] == "ar"

    @pytest.mark.asyncio
    async def test_configured_language_overrides_the_title_heuristic(
        self, monkeypatch, tmp_path
    ) -> None:
        source = self._drive_source(
            lambda url, directory, settings=None: self._prepared(directory, title="Lec 2")
        )
        monkeypatch.setattr(drive_module, "resolve_source", lambda url: source)
        leaf = AsyncMock()
        leaf.fetch.return_value = _transcript()
        provider = self._provider(speech=lambda settings: leaf, stt_language="ar")

        await provider.fetch(_DRIVE_URL, _DRIVE_ID)

        assert leaf.fetch.await_args.kwargs["language"] == "ar"

    @pytest.mark.asyncio
    async def test_workspace_is_removed_once_the_job_settles(self, monkeypatch, tmp_path) -> None:
        prepared: list[PreparedAudio] = []

        def prepare_audio(url, directory, settings=None):
            workspace = self._prepared(directory)
            prepared.append(workspace)
            return workspace

        source = self._drive_source(prepare_audio)
        monkeypatch.setattr(drive_module, "resolve_source", lambda url: source)
        leaf = AsyncMock()
        leaf.fetch.return_value = _transcript()
        provider = self._provider(speech=lambda settings: leaf)

        await provider.fetch(_DRIVE_URL, _DRIVE_ID)

        assert not prepared[0].audio_path.exists()

    @pytest.mark.asyncio
    async def test_workspace_is_removed_when_transcription_fails(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setattr(audio_jobs_module, "_RETRY_ATTEMPTS", 2)
        monkeypatch.setattr(audio_jobs_module, "_RETRY_DELAY_SECONDS", 0)
        prepared: list[PreparedAudio] = []

        def prepare_audio(url, directory, settings=None):
            workspace = self._prepared(directory)
            prepared.append(workspace)
            return workspace

        source = self._drive_source(prepare_audio)
        monkeypatch.setattr(drive_module, "resolve_source", lambda url: source)
        leaf = AsyncMock()
        leaf.fetch.side_effect = lambda *args, **kwargs: (_ for _ in ()).throw(
            ExternalServiceError("boom", service="deepgram")
        )
        provider = self._provider(speech=lambda settings: leaf)

        with pytest.raises(ExternalServiceError):
            await provider.fetch(_DRIVE_URL, _DRIVE_ID)

        assert leaf.fetch.await_count == 2
        assert not prepared[0].audio_path.exists()

    @pytest.mark.asyncio
    async def test_download_failures_are_retried_within_budget(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(audio_jobs_module, "_RETRY_ATTEMPTS", 3)
        monkeypatch.setattr(audio_jobs_module, "_RETRY_DELAY_SECONDS", 0)
        attempts: list[int] = []

        def prepare_audio(url, directory, settings=None):
            attempts.append(len(attempts) + 1)
            if len(attempts) < 3:
                raise ExternalServiceError("Drive hiccup", service="drive")
            return self._prepared(directory)

        source = self._drive_source(prepare_audio)
        monkeypatch.setattr(drive_module, "resolve_source", lambda url: source)
        leaf = AsyncMock()
        leaf.fetch.return_value = _transcript()
        provider = self._provider(speech=lambda settings: leaf)

        result = await provider.fetch(_DRIVE_URL, _DRIVE_ID)

        assert result.title == "Lec 2"
        assert attempts == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_permanent_download_failure_is_not_retried(self, monkeypatch, tmp_path) -> None:
        attempts: list[int] = []

        def prepare_audio(url, directory, settings=None):
            attempts.append(1)
            raise PermanentExternalServiceError("not public", service="drive")

        source = self._drive_source(prepare_audio)
        monkeypatch.setattr(drive_module, "resolve_source", lambda url: source)
        provider = self._provider(speech=lambda settings: AsyncMock())

        with pytest.raises(PermanentExternalServiceError):
            await provider.fetch(_DRIVE_URL, _DRIVE_ID)

        assert len(attempts) == 1

    @pytest.mark.asyncio
    async def test_resume_polls_from_cached_media_without_refetching(
        self, monkeypatch, tmp_path
    ) -> None:
        cache = _FakeTranscriptCache(media=MediaInfo("Cached Lecture", 120))
        monkeypatch.setattr(
            drive_module,
            "resolve_source",
            lambda url: self._drive_source(
                lambda *args, **kwargs: pytest.fail("resume must not re-fetch media")
            ),
        )
        leaf = AsyncMock()
        leaf.fetch.return_value = _transcript()
        provider = self._provider(speech=lambda settings: leaf, cache=cache)

        result = await provider.fetch(_DRIVE_URL, _DRIVE_ID, "asm-1")

        assert result.title == "Cached Lecture"
        assert result.duration_seconds == 120
        leaf.fetch.assert_awaited_once_with(
            _DRIVE_URL,
            resume_token="asm-1",
            webhook_url="",
            language="en",
            audio_path="",
        )

    @pytest.mark.asyncio
    async def test_pending_job_bubbles_up_re_routed_to_the_strategy(
        self, monkeypatch, tmp_path
    ) -> None:
        source = self._drive_source(lambda url, directory, settings=None: self._prepared(directory))
        monkeypatch.setattr(drive_module, "resolve_source", lambda url: source)

        def pending(url, resume_token="", webhook_url="", language="", audio_path=""):
            raise TranscriptJobPending(
                message="still processing",
                provider="assemblyai",
                resume_token="asm-1",
                resumable=True,
            )

        leaf = AsyncMock()
        leaf.fetch.side_effect = pending
        provider = self._provider(speech=lambda settings: leaf)

        with pytest.raises(TranscriptJobPending) as excinfo:
            await provider.fetch(_DRIVE_URL, _DRIVE_ID)

        assert excinfo.value.provider == "drive"
        assert excinfo.value.resume_token == "asm-1"

    @pytest.mark.asyncio
    async def test_missing_audio_leaf_fails_before_fetching_media(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setattr(
            drive_module,
            "resolve_source",
            lambda url: self._drive_source(
                lambda *args, **kwargs: pytest.fail("must not fetch without a leaf")
            ),
        )
        provider = self._provider(speech=lambda settings: None)

        with pytest.raises(ExternalServiceError) as excinfo:
            await provider.fetch(_DRIVE_URL, _DRIVE_ID)

        assert "not configured" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_rejects_caption_bearing_sources(self, monkeypatch) -> None:
        source = self._drive_source(
            lambda *args, **kwargs: pytest.fail("must not fetch a captioned source"),
            native_captions=True,
        )
        monkeypatch.setattr(drive_module, "resolve_source", lambda url: source)
        leaf = AsyncMock()
        provider = self._provider(speech=lambda settings: leaf)

        with pytest.raises(PermanentExternalServiceError):
            await provider.fetch("https://www.youtube.com/watch?v=abcde12345", "abcde12345")

        leaf.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_finished_transcript_is_cached_under_the_drive_id(
        self, monkeypatch, tmp_path
    ) -> None:
        source = self._drive_source(lambda url, directory, settings=None: self._prepared(directory))
        monkeypatch.setattr(drive_module, "resolve_source", lambda url: source)
        leaf = AsyncMock()
        leaf.fetch.return_value = _transcript()
        cache = _FakeTranscriptCache()
        provider = self._provider(speech=lambda settings: leaf, cache=cache)

        result = await provider.fetch(_DRIVE_URL, _DRIVE_ID)

        assert cache.results == [(_DRIVE_ID, result)]
        assert cache.media_writes == [(_DRIVE_ID, "Lec 2", 3600)]

    @pytest.mark.asyncio
    async def test_resume_skips_media_acquisition(self, monkeypatch) -> None:
        cache = _FakeTranscriptCache(media=MediaInfo("Cached Lecture", 120))
        monkeypatch.setattr(
            drive_module,
            "resolve_source",
            lambda url: self._drive_source(
                lambda *args, **kwargs: pytest.fail("resume must not re-fetch media")
            ),
        )
        leaf = AsyncMock()
        leaf.fetch.return_value = _transcript()
        provider = self._provider(speech=lambda settings: leaf, cache=cache)

        await provider.fetch(_DRIVE_URL, _DRIVE_ID, "asm-1")

        leaf.fetch.assert_awaited_once()
