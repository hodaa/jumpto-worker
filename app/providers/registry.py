"""Transcript provider registry and factory.

The registry declares the available transcript strategies (name -> spec) and
the factory builds a concrete strategy instance from settings. Callers build
the single provider named by ``DEFAULT_VIDEO_PROVIDER`` (see
``app.tasks.transcription._configured_provider``) with the always-available
local yt-dlp strategy as its fallback — there is no chain to iterate.

Registered strategies:
    transcriptfetch, supadata, vidwords (cloud, need credentials) and
    yt-dlp (free, always available).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.core.config import Settings
from app.core.logging import get_logger
from app.providers.base import TranscriptProviderStrategy
from app.providers.local import YtDlpTranscriptStrategy
from app.providers.supadata import SupadataTranscriptProvider
from app.providers.transcriptfetch import TranscriptFetchTranscriptProvider
from app.providers.vidwords import VidWordsTranscriptProvider

logger = get_logger(__name__)


@dataclass(frozen=True)
class TranscriptProviderSpec:
    """A registered provider: its id, a settings factory, and its priority."""

    name: str
    build: Callable[[Settings], TranscriptProviderStrategy | None]
    order: int
    description: str = ""


_REGISTRY: dict[str, TranscriptProviderSpec] = {}


def register_provider(spec: TranscriptProviderSpec) -> None:
    """Register a provider spec under ``spec.name``."""
    _REGISTRY[spec.name] = spec


def provider_spec(name: str) -> TranscriptProviderSpec | None:
    """Return the registered spec for ``name`` or ``None``."""
    return _REGISTRY.get(name)


def ordered_specs() -> list[TranscriptProviderSpec]:
    """Return registered provider specs in standard priority order."""
    return sorted(_REGISTRY.values(), key=lambda spec: spec.order)


def build_provider(name: str, settings: Settings) -> TranscriptProviderStrategy | None:
    """Factory: build a single strategy for ``name`` from ``settings``.

    Returns ``None`` when the provider is not configured (missing credentials);
    raises :class:`ValueError` for unknown provider ids.
    """
    spec = provider_spec(name)
    if spec is None:
        raise ValueError(f"Unknown transcript provider: {name}")
    return spec.build(settings)


def _build_transcriptfetch(settings: Settings) -> TranscriptProviderStrategy | None:
    api_key = getattr(settings, "transcriptfetch_api_key", "")
    if not api_key:
        return None
    return TranscriptFetchTranscriptProvider(
        api_key=api_key,
        lang=getattr(settings, "transcriptfetch_lang", "en") or "en",
        mode=getattr(settings, "transcriptfetch_mode", "auto") or "auto",
    )


def _build_supadata(settings: Settings) -> TranscriptProviderStrategy | None:
    api_key = getattr(settings, "supadata_api_key", "")
    if not api_key:
        return None
    return SupadataTranscriptProvider(
        api_key=api_key,
        lang=getattr(settings, "supadata_lang", "en") or "en",
        mode=getattr(settings, "supadata_mode", "auto") or "auto",
    )


def _build_vidwords(settings: Settings) -> TranscriptProviderStrategy | None:
    api_key = getattr(settings, "vidwords_api_key", "")
    if not api_key:
        return None
    return VidWordsTranscriptProvider(
        api_key=api_key,
        base_url=getattr(settings, "vidwords_api_url", "https://vidwords.com"),
        lang=getattr(settings, "vidwords_lang", "en") or "en",
    )


def _build_ytdlp(settings: Settings) -> TranscriptProviderStrategy | None:
    return YtDlpTranscriptStrategy()


register_provider(
    TranscriptProviderSpec(
        name="yt-dlp",
        build=_build_ytdlp,
        order=5,
        description="Local yt-dlp captions with Assembly.ai audio fallback (free, first provider).",
    )
)
register_provider(
    TranscriptProviderSpec(
        name="transcriptfetch",
        build=_build_transcriptfetch,
        order=10,
        description="TranscriptFetch YouTube transcripts API.",
    )
)
register_provider(
    TranscriptProviderSpec(
        name="supadata",
        build=_build_supadata,
        order=20,
        description="Supadata YouTube transcripts/metadata API (async AI jobs).",
    )
)
register_provider(
    TranscriptProviderSpec(
        name="vidwords",
        build=_build_vidwords,
        order=30,
        description="VidWords YouTube transcripts API.",
    )
)
