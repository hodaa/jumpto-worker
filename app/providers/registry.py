from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.core.config import Settings, _live_pipeline_enabled
from app.core.logging import get_logger
from app.providers.base import TranscriptProviderStrategy
from app.sources import resolve_source

logger = get_logger(__name__)

# The strategy every caption-bearing (YouTube) job runs. The worker has no
# other transcript provider: sources resolve to their own strategy.
DEFAULT_PROVIDER = "youtube"


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
_ALIASES: dict[str, str] = {}
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
        drive,  # noqa: F401
        youtube,  # noqa: F401
    )


def register_provider(spec: TranscriptProviderSpec) -> None:
    """Register a provider spec under ``spec.name``."""
    _REGISTRY[spec.name] = spec


def register_alias(alias: str, canonical: str) -> None:
    """Register ``alias`` as another lookup name for the ``canonical`` provider.

    Aliases exist for deploy compatibility: an armed webhook naming a
    strategy's former id must keep resolving to the renamed strategy
    (``yt-dlp`` → ``youtube``).
    """
    _ALIASES[alias] = canonical


def canonical_provider_name(name: str) -> str:
    """Resolve an alias to its canonical provider name (identity otherwise)."""
    _ensure_registered()
    return _ALIASES.get(name, name)


def provider_spec(name: str) -> TranscriptProviderSpec | None:
    """Return the registered spec for ``name`` (or an alias) or ``None``."""
    _ensure_registered()
    spec = _REGISTRY.get(name)
    if spec is None:
        canonical = _ALIASES.get(name)
        spec = _REGISTRY.get(canonical) if canonical is not None else None
    return spec


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


def resolve_provider(settings: Settings, url: str = "") -> TranscriptProviderStrategy:
    """Build and return the transcript provider that serves this job.

    Exactly one provider runs per job, chosen by the URL's source: a captionless
    source (Drive) resolves to the same-named strategy, and a caption-bearing
    source (YouTube) resolves to the ``youtube`` strategy. There is no fallback
    chain: an unconfigured provider, a cloud provider while live calls are
    disabled, or a URL no source claims raises (``ValueError`` or a permanent
    service error) so the job fails loudly instead of silently switching
    strategies.
    """
    name = _provider_name_for(url)
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


def _provider_name_for(url: str) -> str:
    """The strategy serving ``url``: source-named for captionless, else YouTube."""
    if url:
        source = resolve_source(url)
        if not source.native_captions:
            return source.name
    return DEFAULT_PROVIDER
