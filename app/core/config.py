"""Application configuration for the JumpTo worker service."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _coerce_bool(value: str | bool | int) -> bool:
    """Coerce empty-string / falsy env values to a valid bool."""
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("", "0", "false", "no"):
            return False
        if value in ("1", "true", "yes"):
            return True
    return bool(value)


class Settings(BaseSettings):
    """Worker settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("queue_provider", mode="before")
    @classmethod
    def _normalize_queue_provider(cls, value: object) -> object:
        """Lowercase/strip QUEUE_PROVIDER so "Redis"/"RABBITMQ" both work."""
        return value.strip().lower() if isinstance(value, str) else value

    # Backend communication
    backend_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of the JumpTo backend internal API",
    )
    internal_api_key: str = Field(
        default="",
        description="Shared API key for authenticating to the backend internal API",
    )

    # Blob storage (Vercel Blob) for produced video downloads. The SDK reads
    # BLOB_READ_WRITE_TOKEN from the process environment; Compose passes the
    # worker .env through env_file, and this setting additionally mirrors the
    # value into the environment for runs where only pydantic loads the file.
    blob_read_write_token: str = Field(
        default="",
        description=(
            "Vercel Blob read-write token used to upload produced downloads "
            "(BLOB_READ_WRITE_TOKEN env var)"
        ),
    )

    # Celery / broker
    queue_provider: Literal["redis", "rabbitmq"] = Field(
        default="redis",
        description="Celery broker transport: redis or rabbitmq (QUEUE_PROVIDER env var)",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL used as the Celery broker (when QUEUE_PROVIDER=redis)",
    )
    rabbitmq_url: str = Field(
        default="amqp://guest:guest@localhost:5672//",
        description="RabbitMQ connection URL used as the Celery broker (when QUEUE_PROVIDER=rabbitmq)",
    )
    celery_app_name: str = Field(
        default="jumpto",
        description="Celery application/task namespace name; must match producer task names",
    )
    celery_worker_concurrency: int = Field(
        default=8,
        ge=1,
        description="How many worker processes Celery should spawn for this service",
    )
    celery_worker_max_tasks_per_child: int = Field(
        default=100,
        ge=1,
        description=(
            "How many tasks a prefork child may execute before it is replaced; "
            "recycles RSS growth from heavy leaves (yt-dlp/Deno/ffmpeg) "
            "(CELERY_WORKER_MAX_TASKS_PER_CHILD)"
        ),
    )

    # Job / transcription timeout
    job_timeout_seconds: int = Field(
        default=600,
        ge=1,
        description="Max seconds a transcription job may run before it is failed",
    )
    task_timeout_grace_seconds: int = Field(
        default=30,
        ge=1,
        description="Extra hard-time-limit margin for cleanup after a job timeout",
    )

    # Assembly AI
    assembly_api_key: str = Field(
        default="",
        description="Assembly.ai API key for transcription",
    )
    assembly_webhook_base_url: str = Field(
        default="",
        description=(
            "Public base URL the backend exposes for Assembly.ai completion "
            "callbacks. When set, the worker submits Assembly transcripts with "
            "job context appended and ends the task instead of poller-retrying; "
            "empty keeps the legacy in-worker retry (ASSEMBLY_WEBHOOK_BASE_URL)"
        ),
    )

    # Audio transcription fallback selection (local strategies)
    speech_to_text_provider: str = Field(
        default="deepgram",
        description=(
            "Audio transcript provider used when captions are missing "
            "(SPEECH_TO_TEXT_PROVIDER env var): deepgram (default) or assembly"
        ),
    )
    drive_stt_language: str = Field(
        default="",
        description=(
            "Speech-to-text language forced for Google Drive sources "
            "(DRIVE_STT_LANGUAGE; empty keeps the title heuristic)"
        ),
    )

    # Google Drive media downloads
    google_api_key: str = Field(
        default="",
        description=(
            "Google API key for public Drive media downloads; when empty the "
            "browser download endpoint is used (GOOGLE_API_KEY env var)"
        ),
    )
    google_drive_api_url: str = Field(
        default="https://www.googleapis.com/drive/v3",
        description="Google Drive API base URL (GOOGLE_DRIVE_API_URL env var)",
    )

    # Lecture notes (transcript cleanup)
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for lecture-notes cleanup (OPENAI_API_KEY env var)",
    )
    openai_api_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL (OPENAI_API_URL env var)",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model used for transcript cleanup (OPENAI_MODEL)",
    )
    notes_chunk_seconds: int = Field(
        default=600,
        ge=60,
        description="Transcript window cleaned per AI call (NOTES_CHUNK_SECONDS)",
    )
    notes_timeout_seconds: int = Field(
        default=1800,
        ge=60,
        description="Maximum seconds a lecture-notes task may run (NOTES_TIMEOUT_SECONDS)",
    )

    # Deepgram
    deepgram_api_key: str = Field(
        default="",
        description="Deepgram API key for audio transcription (DEEPGRAM_API_KEY env var)",
    )
    deepgram_api_url: str = Field(
        default="https://api.deepgram.com/v1",
        description="Deepgram API base URL (DEEPGRAM_API_URL env var)",
    )
    deepgram_model: str = Field(
        default="nova-3",
        description="Deepgram batch transcription model (DEEPGRAM_MODEL env var)",
    )

    # Environment
    environment: str = Field(
        default="development",
        description="Application environment (development/production)",
    )

    # Sentry error tracking
    sentry_dsn: str = Field(
        default="",
        description="Sentry DSN; empty disables error reporting (SENTRY_DSN env var)",
    )
    sentry_traces_sample_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sentry performance traces sample rate 0..1 (SENTRY_TRACES_SAMPLE_RATE)",
    )

    # External calls
    live_external_calls: Annotated[bool, BeforeValidator(_coerce_bool)] = Field(
        default=False,
        description="Enable live external API calls (yt-dlp, Assembly.ai)",
    )

    # yt-dlp cookie file (shared across providers)
    ytdlp_cookie_file: str = Field(
        default="",
        description="Path to a Netscape cookie file for yt-dlp (YTDLP_COOKIE_FILE env var)",
    )

    # Default cookie file location when running in the Docker worker, where the
    # host cookies are mounted at a well-known path (see docker-compose.yml).
    mounted_ytdlp_cookie_file: str = "/etc/jumpto/cookies.txt"

    # When yt-dlp hits YouTube's "Sign in to confirm you're not a bot", the
    # worker touches this marker file so a host-side cron can re-export cookies
    # from the authenticated Chromium container and replace the cookie file.
    # Leave empty to disable the auto-refresh trigger (marker never written).
    cookie_refresh_marker_path: str = Field(
        default="/var/lib/jumpto/state/refresh-requested",
        description="Path worker touches to request a host-side cookie refresh (COOKIE_REFRESH_MARKER_PATH)",
    )

    # Optional bgutil PO-token provider base URL (YTDLP_BGUTIL_URL env var).
    # When set, build_ydlp_options wires yt-dlp's "youtubepot-bgutilhttp"
    # extractor_args to this server. When left empty, no extractor_args are set
    # and the bgutil plugin falls back to its own built-in 127.0.0.1:4416.
    ytdlp_bgutil_url: str = Field(
        default="",
        description="Base URL of the bgutil-ytdlp-pot-provider server (e.g. http://bgutil-pot:4416)",
    )

    # Socket timeout (seconds) for yt-dlp network requests. Bounds how long a
    # hung YouTube response can pin a worker thread before the retry backoff
    # kicks in; without it a stalled connection could idle a to_thread slot
    # for minutes.
    ytdlp_socket_timeout: float = Field(
        default=30,
        ge=1,
        description="Network socket timeout (seconds) for yt-dlp requests (YTDLP_SOCKET_TIMEOUT)",
    )

    # Worker-side transcript cache (DiskCache / SQLite). Reprocessing the same
    # YouTube video is served from a local on-disk cache keyed by video id
    # instead of re-running yt-dlp.
    transcript_cache_enabled: Annotated[bool, BeforeValidator(_coerce_bool)] = Field(
        default=True,
        description="Enable the disk-backed worker transcript cache (TRANSCRIPT_CACHE_ENABLED)",
    )
    transcript_cache_directory: str = Field(
        default="/var/tmp/jumpto-transcript-cache",
        description="Directory for the disk-backed transcript cache (TRANSCRIPT_CACHE_DIRECTORY)",
    )
    transcript_cache_ttl_seconds: int = Field(
        default=86400,
        ge=0,
        description="How long a cached transcript stays valid (TRANSCRIPT_CACHE_TTL_SECONDS)",
    )
    audio_cache_directory: str = Field(
        default="/var/tmp/jumpto-audio-cache",
        description=(
            "Directory for persisted per-video audio files reused across "
            "transcription attempts and jobs (AUDIO_CACHE_DIRECTORY)"
        ),
    )

    @property
    def resolved_ytdlp_cookie_file(self) -> str | None:
        """Return cookie file path if set and exists, else None."""
        path = self.ytdlp_cookie_file or os.environ.get("YTDLP_COOKIE_FILE", "")
        if path and Path(path).is_file():
            return path
        if path:
            from app.core.logging import get_logger

            get_logger(__name__).warning("YTDLP_COOKIE_FILE set but file not found", path=path)
        mounted = self.mounted_ytdlp_cookie_file
        if mounted and Path(mounted).is_file():
            return mounted
        return None

    @property
    def broker_url(self) -> str:
        """Return the Celery broker URL for the configured queue provider."""
        if self.queue_provider == "rabbitmq":
            return self.rabbitmq_url
        return self.redis_url

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def _live_pipeline_enabled(settings: Settings | None = None) -> bool:
    """Return whether live external transcription calls are active."""
    settings = settings or get_settings()
    return bool(getattr(settings, "live_external_calls", False))
