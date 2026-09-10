"""Unit tests for the transcript provider registry and factory."""

import pytest

from app.core.config import Settings
from app.providers.base import TranscriptProviderStrategy
from app.providers.registry import (
    TranscriptProviderSpec,
    build_provider,
    ordered_specs,
    provider_spec,
)
from app.providers.supadata import SupadataTranscriptProvider
from app.providers.transcriptfetch import TranscriptFetchTranscriptProvider
from app.providers.vidwords import VidWordsTranscriptProvider
from app.providers.ytdlp import YtDlpTranscriptProvider


def _settings(**overrides) -> Settings:
    """Build settings with every cloud provider configured."""
    values = {
        "jumpto_live_external_calls": True,
        "transcriptfetch_api_key": "tf-key",
        "transcriptfetch_lang": "en",
        "transcriptfetch_mode": "auto",
        "supadata_api_key": "sp-key",
        "supadata_lang": "en",
        "supadata_mode": "auto",
        "vidwords_api_key": "vw-key",
        "vidwords_lang": "en",
        "vidwords_api_url": "https://vidwords.com",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class TestRegistry:
    """Tests for the strategy registry declaration."""

    def test_registers_expected_providers_in_standard_order(self) -> None:
        assert [spec.name for spec in ordered_specs()] == [
            "yt-dlp",
            "transcriptfetch",
            "supadata",
            "vidwords",
        ]

    def test_provider_spec_is_registered(self) -> None:
        spec = provider_spec("supadata")
        assert spec is not None
        assert spec.description

    def test_provider_spec_unknown_returns_none(self) -> None:
        assert provider_spec("bogus") is None

    def test_spec_carries_factory(self) -> None:
        assert isinstance(provider_spec("yt-dlp").build(_settings()), YtDlpTranscriptProvider)


class TestBuildProvider:
    """Tests for the factory function."""

    def test_constructs_configured_providers(self) -> None:
        settings = _settings()
        assert isinstance(
            build_provider("transcriptfetch", settings), TranscriptFetchTranscriptProvider
        )
        assert isinstance(build_provider("supadata", settings), SupadataTranscriptProvider)
        assert isinstance(build_provider("vidwords", settings), VidWordsTranscriptProvider)

    def test_ytdlp_is_always_available(self) -> None:
        assert isinstance(build_provider("yt-dlp", _settings()), YtDlpTranscriptProvider)

    def test_unconfigured_provider_returns_none(self) -> None:
        settings = _settings(supadata_api_key="")
        assert build_provider("supadata", settings) is None

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            build_provider("bogus", _settings())

    def test_providers_are_strategies(self) -> None:
        settings = _settings()
        for name in ("transcriptfetch", "supadata", "vidwords", "yt-dlp"):
            assert isinstance(build_provider(name, settings), TranscriptProviderStrategy)


class TestTranscriptProviderSpec:
    """Tests for the spec dataclass."""

    def test_earlier_order_sorts_first(self) -> None:
        spec = TranscriptProviderSpec(name="anything", build=lambda s: None, order=0)
        assert spec.order == 0
        assert spec.description == ""
