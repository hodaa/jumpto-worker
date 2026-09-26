"""Audio download persistence semantics (single source: app/providers/audio.py)."""

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.providers import audio


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(audio_cache_directory=str(tmp_path / "audio-cache"))


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
