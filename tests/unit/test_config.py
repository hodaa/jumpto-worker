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


class TestTranscriptFetchApiKey:
    """Tests for TranscriptFetch key env-var name variants."""

    def test_reads_correctly_spelled_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("TRANSCRIPTFETCH_API_KEY", "secret-1")
        assert Settings(_env_file=None).transcriptfetch_api_key == "secret-1"

    def test_reads_misspelled_env_var_from_dotenv(self, monkeypatch) -> None:
        # "transscriptfetch" mirrors the key name the user has in .env.
        monkeypatch.setenv("TRANSSCRIPTFETCH_API_KEY", "secret-2")
        assert Settings(_env_file=None).transcriptfetch_api_key == "secret-2"


class TestDefaultVideoProvider:
    """Tests for the default transcript provider selection setting."""

    def test_defaults_to_empty(self) -> None:
        assert Settings(_env_file=None).default_video_provider == ""

    def test_reads_configured_value_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("DEFAULT_VIDEO_PROVIDER", "yt-dlp")
        assert Settings(_env_file=None).default_video_provider == "yt-dlp"


class TestQueueProvider:
    """Tests for the Celery broker transport selection."""

    def test_defaults_to_redis(self) -> None:
        assert Settings(_env_file=None).queue_provider == "redis"

    def test_default_redis_broker_url(self) -> None:
        assert Settings(_env_file=None).broker_url == "redis://localhost:6379/0"

    def test_broker_url_uses_redis_url_for_redis_provider(self) -> None:
        settings = Settings(_env_file=None, redis_url="redis://redis:6379/0")
        assert settings.broker_url == "redis://redis:6379/0"

    def test_rabbitmq_broker_url_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("QUEUE_PROVIDER", "rabbitmq")
        monkeypatch.setenv("RABBITMQ_URL", "amqp://user:pass@rabbit:5672//")
        settings = Settings(_env_file=None)
        assert settings.queue_provider == "rabbitmq"
        assert settings.broker_url == "amqp://user:pass@rabbit:5672//"

    def test_default_rabbitmq_broker_url(self, monkeypatch) -> None:
        monkeypatch.setenv("QUEUE_PROVIDER", "rabbitmq")
        assert Settings(_env_file=None).broker_url == "amqp://guest:guest@localhost:5672//"

    def test_accepts_mixed_case_provider(self, monkeypatch) -> None:
        monkeypatch.setenv("QUEUE_PROVIDER", "RabbitMQ")
        assert Settings(_env_file=None).queue_provider == "rabbitmq"

    def test_rejects_unknown_provider(self) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, queue_provider="kafka")


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


class TestCookieRefreshMarkerPath:
    """Tests for the cookie-refresh marker path setting."""

    def test_defaults_to_worker_state_marker(self) -> None:
        assert (
            Settings(_env_file=None).cookie_refresh_marker_path
            == "/var/lib/jumpto/state/refresh-requested"
        )

    def test_reads_configured_value_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("COOKIE_REFRESH_MARKER_PATH", "/etc/jumpto/state/refresh-requested")
        assert (
            Settings(_env_file=None).cookie_refresh_marker_path
            == "/etc/jumpto/state/refresh-requested"
        )

    def test_empty_value_disables_marker(self, monkeypatch) -> None:
        monkeypatch.setenv("COOKIE_REFRESH_MARKER_PATH", "")
        assert Settings(_env_file=None).cookie_refresh_marker_path == ""
