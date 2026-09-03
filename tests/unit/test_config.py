"""Unit tests for worker configuration settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings


class TestCeleryWorkerConcurrencySetting:
    """Tests for the Celery worker concurrency setting."""

    def test_default_is_eight(self) -> None:
        assert Settings(_env_file=None).celery_worker_concurrency == 8

    def test_reads_configured_value_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("CELERY_WORKER_CONCURRENCY", "3")
        assert Settings(_env_file=None).celery_worker_concurrency == 3

    def test_rejects_non_positive_values(self) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, celery_worker_concurrency=0)


class TestResolvedYtDlpCookieFile:
    """Tests for the yt-dlp cookie file resolver."""

    def test_returns_none_when_no_file_configured(self, monkeypatch) -> None:
        monkeypatch.delenv("YTDLP_COOKIE_FILE", raising=False)
        settings = Settings(_env_file=None, ytdlp_cookie_file="")
        assert settings.resolved_ytdlp_cookie_file is None

    def test_returns_configured_path_when_it_exists(self, monkeypatch) -> None:
        monkeypatch.setenv("YTDLP_COOKIE_FILE", "/some/cookies.txt")
        monkeypatch.setattr(Path, "is_file", lambda self: str(self) == "/some/cookies.txt")
        settings = Settings(_env_file=None, ytdlp_cookie_file="")
        assert settings.resolved_ytdlp_cookie_file == "/some/cookies.txt"

    def test_falls_back_to_mounted_docker_path(self, monkeypatch) -> None:
        monkeypatch.delenv("YTDLP_COOKIE_FILE", raising=False)
        settings = Settings(_env_file=None, ytdlp_cookie_file="")
        mounted = settings.mounted_ytdlp_cookie_file
        monkeypatch.setattr(Path, "is_file", lambda self: str(self) == mounted)
        assert settings.resolved_ytdlp_cookie_file == mounted

    def test_ignores_mounted_path_when_it_does_not_exist(self, monkeypatch) -> None:
        monkeypatch.delenv("YTDLP_COOKIE_FILE", raising=False)
        settings = Settings(_env_file=None, ytdlp_cookie_file="")
        monkeypatch.setattr(Path, "is_file", lambda self: False)
        assert settings.resolved_ytdlp_cookie_file is None
