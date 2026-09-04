"""Unit tests for external service providers (fake/live switch)."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.core.exceptions import ExternalServiceError
from app.providers.media import get_media_info
from app.providers.transcript import (
    AssemblyTranscriptProvider,
    FakeTranscriptProvider,
    TranscriptData,
    YouTubeCaptionTranscriptProvider,
    _caption_language,
    _download_caption,
    _parse_assembly_transcript,
    _parse_vtt,
    _preferred_vtt_file,
    _select_caption_language,
    get_transcript_provider,
)
from app.providers.ytdlp import build_ydlp_options

_FAKE_SETTINGS_FAKE_MODE = SimpleNamespace(
    jumpto_transcript_mode="fake",
    jumpto_live_external_calls=True,
    assembly_api_key="secret-key",
)


def _settings(*, mode: str, live: bool, api_key: str) -> SimpleNamespace:
    """Build a minimal settings object for provider selection."""
    return SimpleNamespace(
        jumpto_transcript_mode=mode,
        jumpto_live_external_calls=live,
        assembly_api_key=api_key,
    )


class TestMediaInfoProvider:
    """Tests for the media info provider selection."""

    def test_no_live_calls_returns_fake_media_info(self, monkeypatch) -> None:
        settings = SimpleNamespace(jumpto_live_external_calls=False)
        monkeypatch.setattr("app.providers.media.get_settings", lambda: settings)

        info = get_media_info("abcde12345", "https://youtu.be/abcde12345")

        assert info.duration_seconds == 300
        assert "abcde12345" in info.title

    def test_live_provider_error_is_wrapped(self, monkeypatch) -> None:
        settings = SimpleNamespace(jumpto_live_external_calls=True)
        monkeypatch.setattr("app.providers.media.get_settings", lambda: settings)

        def boom(url: str):
            raise OSError("network down")

        monkeypatch.setattr("app.providers.media._fetch_from_yt_dlp", boom)

        with pytest.raises(ExternalServiceError):
            get_media_info("abcde12345", "https://youtu.be/abcde12345")


class TestTranscriptProviderSelection:
    """Tests for transcript provider factory selection."""

    def test_fake_mode_wins_even_with_credentials(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "app.providers.transcript.get_settings", lambda: _FAKE_SETTINGS_FAKE_MODE
        )

        provider = get_transcript_provider()

        assert isinstance(provider, FakeTranscriptProvider)

    def test_falls_back_to_fake_when_live_not_configured(self, monkeypatch) -> None:
        settings = _settings(mode="real", live=False, api_key="")
        monkeypatch.setattr("app.providers.transcript.get_settings", lambda: settings)

        provider = get_transcript_provider()

        assert isinstance(provider, FakeTranscriptProvider)

    def test_returns_assembly_when_fully_configured(self, monkeypatch) -> None:
        settings = _settings(mode="real", live=True, api_key="key-123")
        monkeypatch.setattr("app.providers.transcript.get_settings", lambda: settings)

        provider = get_transcript_provider()

        assert isinstance(provider, AssemblyTranscriptProvider)
        assert provider.api_key == "key-123"


class TestFakeTranscriptProvider:
    """Tests for the deterministic fake transcript."""

    @pytest.mark.asyncio
    async def test_fetch_returns_timestamped_words(self) -> None:
        provider = FakeTranscriptProvider()

        transcript = await provider.fetch("https://youtu.be/abcde12345")

        assert transcript.language == "en"
        assert len(transcript.words) > 0
        assert transcript.words[0].start_time == 0.0
        assert transcript.text == " ".join(word.word for word in transcript.words)


class TestAssemblyParser:
    """Tests for Assembly.ai response parsing."""

    def test_parse_assembly_transcript(self) -> None:
        data = {
            "language_code": "en",
            "text": "hello world",
            "words": [
                {"text": "hello", "start": 100, "end": 250},
                {"text": "world", "start": 250, "end": 500},
            ],
        }

        parsed = _parse_assembly_transcript(data)

        assert parsed.language == "en"
        assert parsed.text == "hello world"
        assert parsed.words[0].start_time == 0.1
        assert parsed.words[1].end_time == 0.5


class TestAssignmentFetcher:
    """Tests for the Assembly.ai HTTP flow."""

    @pytest.mark.asyncio
    async def test_fetch_downloads_uploads_and_polls(self, monkeypatch, tmp_path) -> None:
        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"fake-audio")

        monkeypatch.setattr("app.providers.transcript._download_audio", lambda url: str(audio_file))

        upload_response = Mock(status_code=200)
        upload_response.json.return_value = {"upload_url": "https://cdn.assemblyai.com/fake"}

        submit_response = Mock(status_code=200)
        submit_response.json.return_value = {"id": "transcript-1"}

        completed_body = {
            "status": "completed",
            "language_code": "en",
            "text": "hello world",
            "words": [{"text": "hello", "start": 0, "end": 100}],
        }
        poll_response = Mock(status_code=200)
        poll_response.json.return_value = completed_body

        client = AsyncMock()
        client.post.side_effect = [upload_response, submit_response]
        client.get.return_value = poll_response
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr("app.providers.transcript.httpx.AsyncClient", lambda: client)

        provider = AssemblyTranscriptProvider("key")
        transcript = await provider.fetch("https://youtu.be/abcde12345")

        assert transcript.text == "hello world"
        assert transcript.words[0].word == "hello"
        assert client.post.call_count == 2
        assert not audio_file.exists()


class TestCaptionSelection:
    """Tests for the caption file selection across downloaded tracks."""

    @staticmethod
    def _paths(names: list[str]) -> list:
        from pathlib import Path

        return [Path(name) for name in names]

    def test_prefers_supported_language_file(self) -> None:
        files = self._paths(["video.de.vtt", "video.en.vtt", "video.fr.vtt"])

        chosen = _preferred_vtt_file(files)

        assert chosen.name == "video.en.vtt"

    def test_reduces_original_language_files(self) -> None:
        assert _caption_language("video.ar-orig.vtt") == "ar"


class TestCaptionLanguageSelection:
    """Tests for the single caption-language selection."""

    def test_prefers_original_supported_language(self) -> None:
        info = {
            "automatic_captions": {"en": [], "ar": [], "ar-orig": []},
            "subtitles": {},
        }

        assert _select_caption_language(info, ("en", "ar")) == "ar"

    def test_prefers_en_plain_when_no_original(self) -> None:
        info = {"automatic_captions": {"ar": [], "en": []}, "subtitles": {}}

        assert _select_caption_language(info, ("en", "ar")) == "en"


class TestCaptionParser:
    """Tests for VTT caption parsing into timestamped words."""

    def test_parses_inline_word_timestamps(self) -> None:
        vtt = (
            "WEBVTT\n"
            "Kind: captions\n\n"
            "00:00:01.199 --> 00:00:03.389 align:start\n"
            "hello<00:00:01.480><c> world</c><00:00:02.240><c> from</c>\n"
        )

        transcript = _parse_vtt(vtt, "en")

        assert transcript.language == "en"
        assert [w.word for w in transcript.words] == ["world", "from"]
        assert transcript.words[0].start_time == pytest.approx(1.48)
        assert transcript.text == "world from"

    def test_parses_line_level_cues_without_word_timing(self) -> None:
        vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nplain line only\n"

        transcript = _parse_vtt(vtt, "en")

        assert [w.word for w in transcript.words] == ["plain", "line", "only"]

    def test_strips_musical_note_symbols(self) -> None:
        vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.500\n\u266a HELLO WORLD \u266a\n"

        transcript = _parse_vtt(vtt, "en")

        assert [w.word for w in transcript.words] == ["HELLO", "WORLD"]
        assert transcript.text == "HELLO WORLD"


class TestYouTubeCaptionFetcher:
    """Tests for the YouTube caption transcript provider."""

    @pytest.mark.asyncio
    async def test_fetch_returns_parsed_transcript(self, monkeypatch) -> None:
        provider = YouTubeCaptionTranscriptProvider()
        monkeypatch.setattr(
            "app.providers.transcript._download_caption",
            lambda url, langs, info=None: (
                "00:00:00.000 --> 00:00:02.000\n<00:00:00.500><c> hi</c>",
                "en",
            ),
        )

        transcript = await provider.fetch("https://www.youtube.com/watch?v=abcde12345")

        assert isinstance(transcript, TranscriptData)
        assert transcript.words[0].word == "hi"
        assert transcript.words[0].start_time == pytest.approx(0.5)


class TestDownloadCaptionMetadataReuse:
    """Tests that a pre-fetched metadata dict avoids a redundant extract_info."""

    def test_reuses_provided_info_without_re_extracting(self, monkeypatch, tmp_path) -> None:
        from unittest.mock import Mock

        extract = Mock()
        monkeypatch.setattr("app.providers.transcript._extract_video_info", extract)
        info = {
            "automatic_captions": {"en-orig": [{"ext": "vtt"}]},
            "subtitles": {},
        }

        with pytest.raises(ExternalServiceError):  # no temp captions written
            _download_caption("https://youtu.be/abcde12345", ("en", "ar"), info)

        extract.assert_not_called()

    def test_extracts_when_info_not_provided(self, monkeypatch, tmp_path) -> None:
        from unittest.mock import Mock

        extract = Mock(return_value={"automatic_captions": {}, "subtitles": {}})
        monkeypatch.setattr("app.providers.transcript._extract_video_info", extract)

        with pytest.raises(ExternalServiceError):
            _download_caption("https://youtu.be/abcde12345", ("en", "ar"))

        extract.assert_called_once_with("https://youtu.be/abcde12345")


class TestYdlpOptions:
    """Tests for the shared yt-dlp options builder."""

    def _settings(self, cookie_file):
        return SimpleNamespace(
            resolved_ytdlp_cookie_file=cookie_file,
            ytdlp_proxy="http://user:pass@residential:8080",
            ytdlp_bgutil_url="",
        )

    def test_sets_writable_cookie_copy_and_proxy_when_configured(
        self, monkeypatch, tmp_path
    ) -> None:
        source = tmp_path / "cookies.txt"
        source.write_text("# Netscape HTTP Cookie File\n")
        monkeypatch.setattr("app.providers.ytdlp.get_settings", lambda: self._settings(str(source)))

        options = build_ydlp_options()

        cookie_path = options["cookiefile"]
        assert cookie_path != str(source)
        assert Path(cookie_path).read_text() == source.read_text()
        assert os.access(cookie_path, os.W_OK)
        assert options["proxy"] == "http://user:pass@residential:8080"

    def test_omits_cookiefile_and_proxy_when_unset(self, monkeypatch) -> None:
        settings = SimpleNamespace(
            resolved_ytdlp_cookie_file=None,
            ytdlp_proxy="",
            ytdlp_bgutil_url="",
        )
        monkeypatch.setattr("app.providers.ytdlp.get_settings", lambda: settings)

        options = build_ydlp_options()

        assert "cookiefile" not in options
        assert "proxy" not in options
        assert "extractor_args" not in options

    def test_sets_bgutil_extractor_args_when_configured(self, monkeypatch, tmp_path) -> None:
        source = tmp_path / "cookies.txt"
        source.write_text("# Netscape HTTP Cookie File\n")
        settings = SimpleNamespace(
            resolved_ytdlp_cookie_file=str(source),
            ytdlp_proxy="",
            ytdlp_bgutil_url="http://bgutil-pot:4416",
        )
        monkeypatch.setattr("app.providers.ytdlp.get_settings", lambda: settings)

        options = build_ydlp_options()

        assert options["extractor_args"] == {
            "youtubepot-bgutilhttp": ["base_url=http://bgutil-pot:4416"],
        }

    def test_overrides_win_over_base_options(self, monkeypatch, tmp_path) -> None:
        source = tmp_path / "cookies.txt"
        source.write_text("# Netscape HTTP Cookie File\n")
        monkeypatch.setattr("app.providers.ytdlp.get_settings", lambda: self._settings(str(source)))

        options = build_ydlp_options(proxy="http://override:3128", noplaylist=False)

        assert options["proxy"] == "http://override:3128"
        assert options["noplaylist"] is False
        assert options["cookiefile"] != str(source)
