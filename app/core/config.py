"""Application configuration for the JumpTo worker service."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, BeforeValidator, Field, field_validator
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
    celery_worker_concurrency: int = Field(
        default=8,
        ge=1,
        description="How many worker processes Celery should spawn for this service",
    )

    # Job / transcription timeout
    job_timeout_seconds: int = Field(
        default=600,
        ge=1,
        description="Max seconds a transcription job may run before it is failed",
    )

    # Assembly AI
    assembly_api_key: str = Field(
        default="",
        description="Assembly.ai API key for transcription",
    )

    # VidWords (YouTube transcripts API)
    vidwords_api_key: str = Field(
        default="",
        description="VidWords API token for fetch transcripts/title via their API",
    )
    vidwords_lang: str = Field(
        default="en",
        description="Preferred caption language for VidWords (VIDWORDS_LANG env var)",
    )
    vidwords_api_url: str = Field(
        default="https://vidwords.com",
        description="VidWords API base URL (VIDWORDS_API_URL env var)",
    )

    # Supadata (YouTube transcripts/metadata API)
    supadata_api_key: str = Field(
        default="",
        description="Supadata API key (SUPADATA_API_KEY env var)",
    )
    supadata_lang: str = Field(
        default="en",
        description="Preferred transcript language for Supadata (SUPADATA_LANG env var)",
    )
    supadata_mode: str = Field(
        default="auto",
        description="Supadata transcript mode: native, generate or auto (SUPADATA_MODE env var)",
    )

    # TranscriptFetch (YouTube transcripts/metadata API)
    transcriptfetch_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "transcriptfetch_api_key",
            "TRANSCRIPTFETCH_API_KEY",
            # The user's .env has an extra "s" ("transscriptfetch"); accept the
            # typo so the key is picked up regardless of spelling.
            "TRANSSCRIPTFETCH_API_KEY",
        ),
        description="TranscriptFetch API key (TRANSCRIPTFETCH_API_KEY env var)",
    )
    transcriptfetch_lang: str = Field(
        default="en",
        description="Preferred caption language for TranscriptFetch (TRANSCRIPTFETCH_LANG env var)",
    )
    transcriptfetch_mode: str = Field(
        default="auto",
        description="TranscriptFetch mode: captions, audio or auto (TRANSCRIPTFETCH_MODE env var)",
    )

    # Environment
    environment: str = Field(
        default="development",
        description="Application environment (development/production)",
    )

    # Default transcript provider strategy (the single cloud one, led first)
    default_video_provider: str = Field(
        default="",
        description=(
            "The single cloud transcript provider used first; empty uses yt-dlp only. "
            "yt-dlp is always tried second as the free fallback. "
            "e.g. DEFAULT_VIDEO_PROVIDER=vidwords"
        ),
    )

    # External calls
    jumpto_live_external_calls: Annotated[bool, BeforeValidator(_coerce_bool)] = Field(
        default=False,
        description="Enable live external API calls (yt-dlp, Assembly.ai)",
    )
    # Transcript mode
    jumpto_transcript_mode: str = Field(
        default="real",
        description="Transcript mode: real or fake",
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

    # Optional HTTP(S)/SOCKS proxy for yt-dlp, e.g. a residential gateway, to
    # avoid YouTube bot-blocks on datacenter IPs (YTDLP_PROXY env var).
    ytdlp_proxy: str = Field(
        default="",
        description="Proxy URL for yt-dlp (e.g. http://user:pass@gateway:port)",
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

    # Worker-side transcript cache (Redis). Reprocessing the same YouTube video
    # is served from cache keyed by video id instead of re-running yt-dlp.
    transcript_cache_enabled: Annotated[bool, BeforeValidator(_coerce_bool)] = Field(
        default=True,
        description="Enable the Redis-backed worker transcript cache (TRANSCRIPT_CACHE_ENABLED)",
    )
    transcript_cache_ttl_seconds: int = Field(
        default=86400,
        ge=0,
        description="How long a cached transcript stays valid (TRANSCRIPT_CACHE_TTL_SECONDS)",
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
    live = bool(getattr(settings, "jumpto_live_external_calls", False))
    mode = str(getattr(settings, "jumpto_transcript_mode", "")).lower()
    return live and mode != "fake"
