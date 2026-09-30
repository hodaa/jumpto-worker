"""Audio download persistence semantics (single source: app/providers/audio.py)."""

import time
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.exceptions import ExternalServiceError
from app.providers import audio


def _settings(tmp_path: Path) -> Settings:
    """Real settings, because the download builds yt-dlp options from them."""
    return Settings(_env_file=None, audio_cache_directory=str(tmp_path / "audio-cache"))


def _fake_run_download(filename: str) -> None:
    """Return a run_download fake that materializes a file at options.outtmpl."""

    def run(options: dict, youtube_url: str) -> None:
        Path(options["outtmpl"]).write_bytes(b"audio-data")

    return run


class TestAudioCache:
    def test_cache_hit_skips_download(self, monkeypatch, tmp_path) -> None:
        settings = _settings(tmp_path)
        cached = audio.audio_cache_path("abcde12345", settings)
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"already-there")

        monkeypatch.setattr(
            audio, "run_download", lambda options, url: pytest.fail("must not download")
        )

        path = audio.download_audio(
            "https://youtu.be/abcde12345", video_id="abcde12345", settings=settings
        )

        assert path == str(cached)

    def test_download_persists_at_cache_path(self, monkeypatch, tmp_path) -> None:
        settings = _settings(tmp_path)
        monkeypatch.setattr(audio, "run_download", _fake_run_download("audio.webm"))

        path = audio.download_audio(
            "https://youtu.be/abcde12345", video_id="abcde12345", settings=settings
        )

        assert path == str(audio.audio_cache_path("abcde12345", settings))
        assert Path(path).is_file()

    def test_download_without_video_id_is_a_temp_file(self, monkeypatch, tmp_path) -> None:
        settings = _settings(tmp_path)
        monkeypatch.setattr(audio, "run_download", _fake_run_download("audio.webm"))

        path = audio.download_audio("https://youtu.be/abcde12345", settings=settings)

        cache_root = Path(settings.audio_cache_directory)
        path_obj = Path(path)
        assert path_obj.is_file()
        assert cache_root not in path_obj.parents
        assert path_obj.suffix == ".webm"

    def test_remove_audio_cache_deletes_file(self, tmp_path) -> None:
        settings = _settings(tmp_path)
        cached = audio.audio_cache_path("abcde12345", settings)
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"data")

        audio.remove_audio_cache("abcde12345", settings)

        assert not cached.exists()

    def test_remove_audio_cache_noop_without_video_id(self, tmp_path) -> None:
        settings = _settings(tmp_path)
        audio.remove_audio_cache("", settings)

    def test_stale_sweep_reclaims_stragglers(self, monkeypatch, tmp_path) -> None:
        settings = _settings(tmp_path)
        monkeypatch.setattr(audio, "_AUDIO_STALE_TTL_SECONDS", -1)
        monkeypatch.setattr(audio, "run_download", _fake_run_download("audio.webm"))

        stale = audio.audio_cache_path("stale-video", settings)
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"old")
        old_mtime = time.time() - 7200
        import os

        os.utime(stale, (old_mtime, old_mtime))

        audio.download_audio("https://youtu.be/other", video_id="other", settings=settings)

        assert not stale.exists()


class TestDownloadFailurePaths:
    """A failed download must fail the job *and* clean up the temp file.

    Every failure below writes to a temp file first, so without the cleanup each
    one would leak a file on disk for every failed job. The temp file is
    unlinked before the download starts, so the cleanup is asserted on the
    ``remove_file`` call rather than on the file's absence — by the time the
    error surfaces the path may not exist either way.
    """

    @staticmethod
    def _patched(monkeypatch, body, *, removed: list):
        """Drive ``download_audio`` with a recording run_download and remove_file."""
        monkeypatch.setattr(audio, "remove_file", lambda path: removed.append(path))
        seen: list[str] = []

        def run(options: dict, youtube_url: str) -> None:
            seen.append(options["outtmpl"])
            body(options)

        monkeypatch.setattr(audio, "run_download", run)
        return seen

    def test_empty_download_is_rejected_and_temp_removed(self, monkeypatch, tmp_path) -> None:
        """yt-dlp exiting 0 without writing a file must not look like success.

        Transcribing a zero-byte file would yield a bogus empty transcript, so an
        empty or missing destination is a hard failure.
        """
        removed: list[str] = []
        seen = self._patched(monkeypatch, lambda o: None, removed=removed)

        with pytest.raises(ExternalServiceError, match="produced no file"):
            audio.download_audio("https://youtu.be/abcde12345", settings=_settings(tmp_path))

        assert len(seen) == 1
        assert removed == seen

    def test_underlying_service_error_propagates_and_temp_removed(
        self, monkeypatch, tmp_path
    ) -> None:
        """A typed leaf error keeps its own identity rather than being re-wrapped."""
        removed: list[str] = []

        def boom(options):
            raise ExternalServiceError("bot check", service="assembly")

        seen = self._patched(monkeypatch, boom, removed=removed)

        with pytest.raises(ExternalServiceError, match="bot check") as excinfo:
            audio.download_audio("https://youtu.be/abcde12345", settings=_settings(tmp_path))

        assert excinfo.value.details["service"] == "assembly"
        assert removed == seen

    def test_unexpected_download_error_is_wrapped_and_temp_removed(
        self, monkeypatch, tmp_path
    ) -> None:
        """An unexpected failure becomes a typed service error, not a raw traceback."""
        removed: list[str] = []

        def boom(options):
            raise RuntimeError("yt-dlp exploded")

        seen = self._patched(monkeypatch, boom, removed=removed)

        with pytest.raises(ExternalServiceError, match="Could not download audio") as excinfo:
            audio.download_audio("https://youtu.be/abcde12345", settings=_settings(tmp_path))

        assert excinfo.value.details["service"] == "yt-dlp"
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert removed == seen

    def test_cleanup_is_skipped_when_the_download_succeeds(self, monkeypatch, tmp_path) -> None:
        """The success path must not delete the file it just produced."""
        settings = _settings(tmp_path)
        removed: list[str] = []
        self._patched(
            monkeypatch,
            lambda o: Path(o["outtmpl"]).write_bytes(b"audio-data"),
            removed=removed,
        )

        # With a video_id the download is moved into the persistent cache, so
        # this covers the rename-into-cache branch too -- a stray cleanup there
        # would delete the file before it is moved into place.
        path = audio.download_audio(
            "https://youtu.be/abcde12345", video_id="abcde12345", settings=settings
        )

        assert Path(path) == Path(settings.audio_cache_directory) / "abcde12345.webm"
        assert Path(path).is_file()
        assert removed == []


class TestAudioCacheUnavailable:
    """An unusable cache directory degrades to a temp file, never fails."""

    def test_unwritable_cache_dir_falls_back_to_temp_file(self, monkeypatch, tmp_path) -> None:
        settings = _settings(tmp_path)

        def explode(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(Path, "mkdir", explode)
        monkeypatch.setattr(audio, "run_download", _fake_run_download("audio.webm"))

        path = audio.download_audio(
            "https://youtu.be/abcde12345", video_id="abcde12345", settings=settings
        )

        assert Path(path).is_file()
        assert Path(path).parent != Path(settings.audio_cache_directory)
