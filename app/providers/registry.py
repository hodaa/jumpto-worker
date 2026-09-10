from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.core.config import Settings, _live_pipeline_enabled
from app.core.logging import get_logger
from app.providers.base import TranscriptProviderStrategy

logger = get_logger(__name__)


@dataclass(frozen=True)
class TranscriptProviderSpec:
    """A registered provider: its id, a settings factory, and registry metadata.

    ``uses_cloud`` records whether the strategy needs live external API calls —
    a deployment property kept here rather than on the strategy interface so
    strategies stay pure transcript behavior.
    """

    name: str
    build: Callable[[Settings], TranscriptProviderStrategy | None]
    order: int
    description: str = ""
    uses_cloud: bool = True


_REGISTRY: dict[str, TranscriptProviderSpec] = {}
_PROVIDERS_LOADED = False


def _ensure_registered() -> None:
    """Import every provider module so its module-level registration runs."""
    global _PROVIDERS_LOADED
    if _PROVIDERS_LOADED:
        return
    _PROVIDERS_LOADED = True


def register_provider(spec: TranscriptProviderSpec) -> None:
    """Register a provider spec under ``spec.name``."""
    _REGISTRY[spec.name] = spec


def provider_spec(name: str) -> TranscriptProviderSpec | None:
    """Return the registered spec for ``name`` or ``None``."""
    _ensure_registered()
    return _REGISTRY.get(name)


def ordered_specs() -> list[TranscriptProviderSpec]:
    """Return registered provider specs in standard priority order."""
    _ensure_registered()
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


def _resolve_chain(settings: Settings) -> list[str]:
    """Return the ordered list of provider names to try.

    Uses the explicit ``provider_chain`` list when set; otherwise falls back
    to ``default_video_provider`` (if set) followed by ``yt-dlp`` as the
    always-available last resort.
    """
    chain = [
        name.strip().lower()
        for name in (getattr(settings, "provider_chain", None) or [])
        if name.strip()
    ]
    if chain:
        return chain

    default = (getattr(settings, "default_video_provider", "") or "").strip().lower()
    return ([default] if default else []) + ["yt-dlp"]


def _build_candidate(
    name: str, settings: Settings
) -> TranscriptProviderStrategy | None:
    """Build a provider, returning None when it's unavailable.

    Skips unknown names, unconfigured providers, and cloud providers
    when live external calls are disabled.
    """
    spec = provider_spec(name)
    if spec is None:
        logger.warning("Unknown transcript provider in chain; skipping", provider=name)
        return None
    if spec.uses_cloud and not _live_pipeline_enabled(settings):
        logger.info("Skipping cloud provider while live calls disabled", provider=name)
        return None
    provider = spec.build(settings)
    if provider is None:
        logger.warning(
            "Transcript provider in chain not configured; skipping", provider=name
        )
        return None
    return provider


def candidates(settings: Settings) -> list[TranscriptProviderStrategy]:
    """Build and return the ordered list of transcript providers to try.

    Resolves the provider chain from ``settings``, builds each candidate,
    and skips unknown/unconfigured/cloud providers when live calls are off.
    Deduplicates, keeping the first occurrence of each provider name.
    """
    chain = _resolve_chain(settings)
    result: list[TranscriptProviderStrategy] = []
    seen: set[str] = set()
    for name in chain:
        if name in seen:
            continue
        seen.add(name)
        provider = _build_candidate(name, settings)
        if provider is not None:
            result.append(provider)
    return result
