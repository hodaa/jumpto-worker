"""Unit tests for the transcript provider registry and factory."""

import pytest

from app.core.config import Settings
from app.providers.base import TranscriptProviderStrategy
from app.providers.drive import DriveTranscriptProvider
from app.providers.registry import (
    TranscriptProviderSpec,
    build_provider,
    canonical_provider_name,
    ordered_specs,
    provider_spec,
)
from app.providers.youtube import YouTubeTranscriptProvider


def _settings(**overrides) -> Settings:
    """Build settings for the worker's local strategies."""
    values = {"live_external_calls": True}
    values.update(overrides)
    return Settings(_env_file=None, **values)


class TestRegistry:
    """Tests for the strategy registry declaration."""

    def test_registers_the_two_source_strategies_in_standard_order(self) -> None:
        assert [spec.name for spec in ordered_specs()] == ["youtube", "drive"]

    def test_provider_spec_is_registered(self) -> None:
        spec = provider_spec("drive")
        assert spec is not None
        assert spec.description

    def test_provider_spec_unknown_returns_none(self) -> None:
        assert provider_spec("bogus") is None

    def test_spec_carries_factory(self) -> None:
        assert isinstance(provider_spec("youtube").build(_settings()), YouTubeTranscriptProvider)
        assert isinstance(provider_spec("drive").build(_settings()), DriveTranscriptProvider)


class TestBuildProvider:
    """Tests for the factory function."""

    def test_constructs_the_local_strategies(self) -> None:
        settings = _settings()
        assert isinstance(build_provider("youtube", settings), YouTubeTranscriptProvider)
        assert isinstance(build_provider("drive", settings), DriveTranscriptProvider)

    def test_legacy_ytdlp_alias_builds_the_youtube_strategy(self) -> None:
        assert isinstance(build_provider("yt-dlp", _settings()), YouTubeTranscriptProvider)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            build_provider("bogus", _settings())

    def test_providers_are_strategies(self) -> None:
        settings = _settings()
        for name in ("youtube", "drive", "yt-dlp"):
            assert isinstance(build_provider(name, settings), TranscriptProviderStrategy)


class TestAliases:
    """Tests for deploy-compatibility alias resolution."""

    def test_ytdlp_alias_canonicalizes_to_youtube(self) -> None:
        assert canonical_provider_name("yt-dlp") == "youtube"

    def test_canonical_name_is_identity(self) -> None:
        assert canonical_provider_name("youtube") == "youtube"
        assert canonical_provider_name("drive") == "drive"

    def test_alias_resolves_to_the_canonical_spec(self) -> None:
        assert provider_spec("yt-dlp") is provider_spec("youtube")


class TestTranscriptProviderSpec:
    """Tests for the spec dataclass."""

    def test_earlier_order_sorts_first(self) -> None:
        spec = TranscriptProviderSpec(name="anything", build=lambda s: None, order=0)
        assert spec.order == 0
        assert spec.description == ""
