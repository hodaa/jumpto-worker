"""Unit tests for the transcript provider registry and factory."""

import pytest

from app.core.config import Settings
from app.providers.base import TranscriptProviderStrategy
from app.providers.local import YtDlpTranscriptStrategy
from app.providers.registry import (
    TranscriptProviderSpec,
    build_provider,
    build_provider_chain,
    ordered_specs,
    provider_spec,
)
from app.providers.supadata import SupadataTranscriptProvider
from app.providers.transcriptfetch import TranscriptFetchTranscriptProvider
from app.providers.vidwords import VidWordsTranscriptProvider


def _settings(**overrides) -> Settings:
    """Build settings with every cloud provider configured."""
    values = {
        "jumpto_live_external_calls": True,
        "jumpto_transcript_mode": "real",
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
            "transcriptfetch",
            "supadata",
            "vidwords",
            "yt-dlp",
        ]

    def test_provider_spec_is_registered(self) -> None:
        spec = provider_spec("supadata")
        assert spec is not None
        assert spec.description

    def test_provider_spec_unknown_returns_none(self) -> None:
        assert provider_spec("bogus") is None

    def test_spec_carries_factory(self) -> None:
        assert isinstance(provider_spec("yt-dlp").build(_settings()), YtDlpTranscriptStrategy)


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
        assert isinstance(build_provider("yt-dlp", _settings()), YtDlpTranscriptStrategy)

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


class TestBuildProviderChain:
    """Tests for the ordered strategy chain."""

    def test_standard_order_when_no_default(self) -> None:
        chain = build_provider_chain(_settings(default_video_provider=""))

        assert [strategy.name for strategy in chain] == [
            "transcriptfetch",
            "supadata",
            "vidwords",
            "yt-dlp",
        ]

    def test_default_provider_is_used_first(self) -> None:
        chain = build_provider_chain(_settings(default_video_provider="vidwords"))

        assert [strategy.name for strategy in chain] == [
            "vidwords",
            "transcriptfetch",
            "supadata",
            "yt-dlp",
        ]

    def test_default_ytdlp_starts_with_ytdlp(self) -> None:
        chain = build_provider_chain(_settings(default_video_provider="yt-dlp"))

        assert [strategy.name for strategy in chain] == [
            "yt-dlp",
            "transcriptfetch",
            "supadata",
            "vidwords",
        ]

    def test_skips_unconfigured_cloud_providers(self) -> None:
        settings = _settings(supadata_api_key="", vidwords_api_key="")
        chain = build_provider_chain(settings)

        assert [strategy.name for strategy in chain] == ["transcriptfetch", "yt-dlp"]

    def test_unconfigured_default_is_ignored(self) -> None:
        settings = _settings(default_video_provider="vidwords", vidwords_api_key="")
        chain = build_provider_chain(settings)

        assert [strategy.name for strategy in chain] == [
            "transcriptfetch",
            "supadata",
            "yt-dlp",
        ]

    def test_unknown_default_falls_back_to_standard_order(self) -> None:
        chain = build_provider_chain(_settings(default_video_provider="bogus"))

        assert [strategy.name for strategy in chain] == [
            "transcriptfetch",
            "supadata",
            "vidwords",
            "yt-dlp",
        ]

    def test_default_is_case_insensitive(self) -> None:
        chain = build_provider_chain(_settings(default_video_provider="Supadata"))

        assert chain[0].name == "supadata"

    def test_no_providers_configured_keeps_ytdlp_terminal(self) -> None:
        settings = _settings(
            transcriptfetch_api_key="",
            supadata_api_key="",
            vidwords_api_key="",
        )
        chain = build_provider_chain(settings)

        assert [strategy.name for strategy in chain] == ["yt-dlp"]


class TestTranscriptProviderSpec:
    """Tests for the spec dataclass."""

    def test_earlier_order_sorts_first(self) -> None:
        spec = TranscriptProviderSpec(name="anything", build=lambda s: None, order=0)
        assert spec.order == 0
        assert spec.description == ""
