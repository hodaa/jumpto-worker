"""Application configuration for the JumpTo worker service."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, Field
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

    # Backend communication
    backend_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of the JumpTo backend internal API",
    )
    internal_api_key: str = Field(
        default="",
        description="Shared API key for authenticating to the backend internal API",
    )

    # Celery / Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL used as the Celery broker",
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

    # Environment
    environment: str = Field(
        default="development",
        description="Application environment (development/production)",
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

    # Optional HTTP(S)/SOCKS proxy for yt-dlp, e.g. a residential gateway, to
    # avoid YouTube bot-blocks on datacenter IPs (YTDLP_PROXY env var).
    ytdlp_proxy: str = Field(
        default="",
        description="Proxy URL for yt-dlp (e.g. http://user:pass@gateway:port)",
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
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
