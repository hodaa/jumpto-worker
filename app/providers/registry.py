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

    ``cache_config_fields`` lists the settings attribute names whose values
    change what transcript a provider returns (language, mode, ...). The worker
    transcript cache uses them to version its namespace, so a provider declares
    its own cache-affecting knobs instead of the cache module hardcoding them.
    """

    name: str
    build: Callable[[Settings], TranscriptProviderStrategy | None]
    order: int
    description: str = ""
    uses_cloud: bool = True
    cache_config_fields: tuple[str, ...] = ()


_REGISTRY: dict[str, TranscriptProviderSpec] = {}
_PROVIDERS_LOADED = False


def _ensure_registered() -> None:
    """Import every provider module so its module-level registration runs.

    This is the closed-registry extension point: adding a provider means
    writing a module with a spec and adding one import line here. ``_REGISTRY``
    is never mutated by the registry itself; each provider module calls
    ``register_provider()`` at import time.
    """
    global _PROVIDERS_LOADED
    if _PROVIDERS_LOADED:
        return
    _PROVIDERS_LOADED = True
    from app.providers import (
        supadata,  # noqa: F401
        transcriptfetch,  # noqa: F401
        vidwords,  # noqa: F401
        ytdlp,  # noqa: F401
    )


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


def resolve_provider(settings: Settings) -> TranscriptProviderStrategy:
    """Build and return the single configured transcript provider.

    Exactly one provider runs per job (``VIDEO_PROVIDER``, default ``yt-dlp``).
    There is no fallback chain: an unknown name, an unconfigured provider, or a
    cloud provider while live calls are disabled raises a :class:`ValueError`
    so the job fails loudly instead of silently switching strategies.
    """
    name = (getattr(settings, "video_provider", "") or "yt-dlp").strip().lower()
    spec = provider_spec(name)
    if spec is None:
        raise ValueError(f"Unknown transcript provider: {name}")
    if spec.uses_cloud and not _live_pipeline_enabled(settings):
        raise ValueError(
            f"Transcript provider {name!r} needs live external calls, which are disabled"
        )
    provider = spec.build(settings)
    if provider is None:
        raise ValueError(f"Transcript provider {name!r} is not configured")
    return provider
