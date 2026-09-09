"""Unit tests for external service providers (fake/live switch)."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.core.exceptions import ExternalServiceError
from app.integrations.ytdlp import build_ydlp_options
from app.providers.assembly import (
    AssemblyTranscriptProvider,
    _parse_assembly_transcript,
    _run_download,
)
from app.providers.media import get_media_info
from app.providers.transcript import (
    FakeTranscriptProvider,
    TranscriptData,
    TranscriptJobPending,
    YouTubeCaptionTranscriptProvider,
    _caption_language,
    _caption_targets,
    _download_caption,
    _parse_vtt,
    _preferred_vtt_file,
    get_transcript_provider,
)

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

        monkeypatch.setattr(
            "app.providers.assembly._download_audio", lambda url, info=None: str(audio_file)
        )

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

        monkeypatch.setattr("app.providers.assembly.httpx.AsyncClient", lambda: client)

        provider = AssemblyTranscriptProvider("key")
        transcript = await provider.fetch("https://youtu.be/abcde12345")

        assert transcript.text == "hello world"
        assert transcript.words[0].word == "hello"
        assert client.post.call_count == 2
        assert not audio_file.exists()

    @pytest.mark.asyncio
    async def test_resume_checks_once_and_raises_pending(self, monkeypatch) -> None:
        poll_response = Mock(status_code=200)
        poll_response.json.return_value = {"status": "processing"}
        client = AsyncMock()
        client.get.return_value = poll_response
        monkeypatch.setattr("app.providers.assembly.httpx.AsyncClient", lambda: client)

        provider = AssemblyTranscriptProvider("key")
        with pytest.raises(TranscriptJobPending) as excinfo:
            await provider.fetch(
                "https://youtu.be/abcde12345", resume_token="transcript-1"
            )

        assert excinfo.value.resume_token == "transcript-1"
        assert excinfo.value.resumable is True
        client.get.assert_awaited_once()
        client.post.assert_not_awaited()


class TestCaptionSelection:
    """Tests for the caption file selection across downloaded tracks."""

    @staticmethod
    def _paths(names: list[str]) -> list:
        from pathlib import Path

        return [Path(name) for name in names]

    def test_prefers_original_audio_track_file(self) -> None:
        files = self._paths(["video.en.vtt", "video.en-orig.vtt", "video.fr.vtt"])

        chosen = _preferred_vtt_file(files)

        assert chosen.name == "video.en-orig.vtt"

    def test_picks_any_language(self) -> None:
        files = self._paths(["video.it.vtt"])

        chosen = _preferred_vtt_file(files)

        assert chosen.name == "video.it.vtt"

    def test_reduces_original_language_files(self) -> None:
        assert _caption_language("video.ar-orig.vtt") == "ar"


class TestCaptionLanguageSelection:
    """Tests for the caption-track selection (language-agnostic)."""

    def test_prefers_manual_subtitles_over_auto(self) -> None:
        info = {
            "automatic_captions": {"en-orig": []},
            "subtitles": {"it": []},
            "original_language": "it",
        }

        assert _caption_targets(info) == ["it"]

    def test_prefers_original_language_and_orig_track(self) -> None:
        info = {
            "automatic_captions": {"en-orig": [], "it-1-orig": [], "fr": []},
            "subtitles": {},
            "original_language": "it",
        }

        targets = _caption_targets(info)

        assert targets[0] == "it-1-orig"

    def test_falls_back_to_any_auto_caption(self) -> None:
        info = {"automatic_captions": {"ar": [], "en": []}, "subtitles": {}}

        assert _caption_targets(info) == ["ar"] or _caption_targets(info) == ["en"]

    def test_no_captions_raises(self) -> None:
        from app.core.exceptions import ExternalServiceError

        info = {"automatic_captions": {}, "subtitles": {}}

        with pytest.raises(ExternalServiceError):
            _caption_targets(info)


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


class _ReplayYoutubeDL:
    """Stands in for yt-dlp and records whether it replayed cached info."""

    def __init__(self, options: dict) -> None:
        self.options = options
        self.replayed: tuple[dict, bool] | None = None
        self.downloaded: list[str] | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def process_ie_result(self, info: dict, download: bool = True) -> dict:
        self.replayed = (info, download)
        return info

    def download(self, url_list: list[str]) -> None:
        self.downloaded = url_list


class TestDownloadCaptionMetadataReuse:
    """Tests that pre-fetched yt-dlp metadata is replayed, not re-extracted."""

    def test_reuses_provided_info_without_re_extracting(self, monkeypatch, tmp_path) -> None:
        extract = Mock()
        fake = _ReplayYoutubeDL({})
        monkeypatch.setattr("app.providers.transcript._extract_video_info", extract)
        monkeypatch.setattr("yt_dlp.YoutubeDL", lambda options: fake)
        info = {
            "automatic_captions": {"en-orig": [{"ext": "vtt"}]},
            "subtitles": {},
        }

        with pytest.raises(ExternalServiceError):  # no temp captions written
            _download_caption("https://youtu.be/abcde12345", info)

        extract.assert_not_called()
        assert fake.replayed == (info, True)

    def test_run_download_replays_info_without_re_extracting(self, monkeypatch) -> None:
        fake = _ReplayYoutubeDL({})
        monkeypatch.setattr("yt_dlp.YoutubeDL", lambda options: fake)
        info = {"id": "abc", "title": "t", "formats": []}

        _run_download({}, "https://youtu.be/abc", info)

        assert fake.replayed == (info, True)

    def test_run_download_still_extracts_when_no_info(self, monkeypatch) -> None:
        fake = _ReplayYoutubeDL({})
        monkeypatch.setattr("yt_dlp.YoutubeDL", lambda options: fake)

        _run_download({}, "https://youtu.be/abc")

        assert fake.replayed is None
        assert fake.downloaded == ["https://youtu.be/abc"]

    def test_extracts_when_info_not_provided(self, monkeypatch, tmp_path) -> None:
        from unittest.mock import Mock

        extract = Mock(return_value={"automatic_captions": {}, "subtitles": {}})
        monkeypatch.setattr("app.providers.transcript._extract_video_info", extract)

        with pytest.raises(ExternalServiceError):
            _download_caption("https://youtu.be/abcde12345")

        extract.assert_called_once_with("https://youtu.be/abcde12345")


class TestYdlpOptions:
    """Tests for the shared yt-dlp options builder."""

    def _settings(self, cookie_file):
        return SimpleNamespace(
            resolved_ytdlp_cookie_file=cookie_file,
            ytdlp_proxy="http://user:pass@residential:8080",
            ytdlp_bgutil_url="",
            ytdlp_socket_timeout=30,
        )

    def test_sets_writable_cookie_copy_and_proxy_when_configured(
        self, monkeypatch, tmp_path
    ) -> None:
        source = tmp_path / "cookies.txt"
        source.write_text("# Netscape HTTP Cookie File\n")
        monkeypatch.setattr("app.integrations.ytdlp.get_settings", lambda: self._settings(str(source)))

        options = build_ydlp_options()

        cookie_path = options["cookiefile"]
        assert cookie_path != str(source)
        assert Path(cookie_path).read_text() == source.read_text()
        assert os.access(cookie_path, os.W_OK)
        assert options["proxy"] == "http://user:pass@residential:8080"

    def test_cookie_copy_sanitizes_broken_netscape_rows(self, monkeypatch, tmp_path) -> None:
        source = tmp_path / "cookies.txt"
        source.write_text(
            "# Netscape HTTP Cookie File\n"
            ".youtube.com\tFALSE\t/\tTRUE\t1804257102\t__Secure-YNID\t21.YT=abc\n"
            ".youtube.com\tFALSE\t/\tTRUE\t-1\tYSC\tp6lOdrlT3gs\n"
            "youtube.com\tTRUE\t/\tFALSE\t-1\tCONSENT\tYES\n"
            "accounts.google.com\tTRUE\t/\tTRUE\t1791297119\tOTZ\t8773352\n"
        )
        monkeypatch.setattr("app.integrations.ytdlp.get_settings", lambda: self._settings(str(source)))

        options = build_ydlp_options()

        body = Path(options["cookiefile"]).read_text()
        assert ".youtube.com\tTRUE\t/\tTRUE\t1804257102\t__Secure-YNID\t21.YT=abc\n" in body
        assert ".youtube.com\tTRUE\t/\tTRUE\t0\tYSC\tp6lOdrlT3gs\n" in body
        assert "youtube.com\tFALSE\t/\tFALSE\t0\tCONSENT\tYES\n" in body
        assert "accounts.google.com\tFALSE\t/\tTRUE\t1791297119\tOTZ\t8773352\n" in body

    def test_cookie_copy_sanitizes_httponly_dotted_domain(
        self, monkeypatch, tmp_path
    ) -> None:
        source = tmp_path / "cookies.txt"
        source.write_text("# Netscape HTTP Cookie File\n#HttpOnly_.youtube.com\tFALSE\t/\tTRUE\t0\tSID\tv\n")
        monkeypatch.setattr("app.integrations.ytdlp.get_settings", lambda: self._settings(str(source)))

        options = build_ydlp_options()

        body = Path(options["cookiefile"]).read_text()
        assert "#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tv\n" in body

    def test_omits_cookiefile_and_proxy_when_unset(self, monkeypatch) -> None:
        settings = SimpleNamespace(
            resolved_ytdlp_cookie_file=None,
            ytdlp_proxy="",
            ytdlp_bgutil_url="",
            ytdlp_socket_timeout=30,
        )
        monkeypatch.setattr("app.integrations.ytdlp.get_settings", lambda: settings)

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
            ytdlp_socket_timeout=30,
        )
        monkeypatch.setattr("app.integrations.ytdlp.get_settings", lambda: settings)

        options = build_ydlp_options()

        assert options["extractor_args"] == {
            "youtubepot-bgutilhttp": {"base_url": ["http://bgutil-pot:4416"]},
        }

    def test_overrides_win_over_base_options(self, monkeypatch, tmp_path) -> None:
        source = tmp_path / "cookies.txt"
        source.write_text("# Netscape HTTP Cookie File\n")
        monkeypatch.setattr("app.integrations.ytdlp.get_settings", lambda: self._settings(str(source)))

        options = build_ydlp_options(proxy="http://override:3128", noplaylist=False)

        assert options["proxy"] == "http://override:3128"
        assert options["noplaylist"] is False
        assert options["cookiefile"] != str(source)
