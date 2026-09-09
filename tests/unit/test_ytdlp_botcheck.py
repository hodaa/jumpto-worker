"""Unit tests for yt-dlp bot-check detection and cookie-refresh triggering."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yt_dlp

from app.core.exceptions import ExternalServiceError
from app.providers import media, transcript
from app.providers.ytdlp import (
    is_youtube_bot_check,
    request_cookie_refresh,
)
from scripts.refresh_cookies import EXIT_NOT_LOGGED_IN, EXIT_OK, to_netscape_rows, write_cookies

_BOT_CHECK_ERROR = "ERROR: [youtube] xxx: Sign in to confirm you're not a bot"
_UNRELATED_ERROR = "ERROR: [youtube] xxx: Video unavailable"


class _BoomYoutubeDL:
    """Stands in for yt-dlp and always raises a bot-check DownloadError."""

    def __init__(self, *_: object) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def extract_info(self, url: str, download: bool = False) -> dict:
        raise yt_dlp.utils.DownloadError(_BOT_CHECK_ERROR)

    def download(self, url_list: list[str]) -> None:
        raise yt_dlp.utils.DownloadError(_BOT_CHECK_ERROR)

    def process_ie_result(self, info: dict, download: bool = True) -> dict:
        raise yt_dlp.utils.DownloadError(_BOT_CHECK_ERROR)


def _media_settings(tmp_path) -> SimpleNamespace:
    """Settings that enable live media calls with a refresh marker."""
    return SimpleNamespace(
        jumpto_live_external_calls=True,
        cookie_refresh_marker_path=str(tmp_path / "refresh-requested"),
    )


def _transcript_settings(tmp_path) -> SimpleNamespace:
    """Settings needed by transcript helpers when options are built."""
    return SimpleNamespace(
        ytdlp_cookie_file="",
        mounted_ytdlp_cookie_file="",
        ytdlp_proxy=None,
        ytdlp_bgutil_url=None,
        cookie_refresh_marker_path=str(tmp_path / "refresh-requested"),
        ytdlp_subtitle_max_retries=1,
    )


class TestIsYoutubeBotCheck:
    """Tests for bot-check detection."""

    def test_matches_bot_check_text(self) -> None:
        assert is_youtube_bot_check(_BOT_CHECK_ERROR)

    def test_matches_case_insensitively(self) -> None:
        assert is_youtube_bot_check("SIGN IN TO CONFIRM YOU'RE NOT A BOT")

    def test_ignores_unrelated_errors(self) -> None:
        assert not is_youtube_bot_check(_UNRELATED_ERROR)

    def test_accepts_exception_instances(self) -> None:
        assert is_youtube_bot_check(yt_dlp.utils.DownloadError(_BOT_CHECK_ERROR))


class TestRequestCookieRefresh:
    """Tests for marker-file triggering."""

    def test_writes_marker_when_configured(self, monkeypatch, tmp_path) -> None:
        marker = tmp_path / "refresh-requested"
        settings = SimpleNamespace(cookie_refresh_marker_path=str(marker))
        monkeypatch.setattr("app.providers.ytdlp.get_settings", lambda: settings)

        request_cookie_refresh()

        assert marker.is_file()

    def test_noop_when_marker_already_exists(self, monkeypatch, tmp_path) -> None:
        marker = tmp_path / "refresh-requested"
        marker.touch()
        settings = SimpleNamespace(cookie_refresh_marker_path=str(marker))
        monkeypatch.setattr("app.providers.ytdlp.get_settings", lambda: settings)

        request_cookie_refresh()

        assert marker.read_bytes() == b""

    def test_noop_when_marker_path_is_empty(self, monkeypatch) -> None:
        settings = SimpleNamespace(cookie_refresh_marker_path="")
        monkeypatch.setattr("app.providers.ytdlp.get_settings", lambda: settings)

        request_cookie_refresh()

    def test_does_not_raise_on_writable_errors(self, monkeypatch, tmp_path) -> None:
        marker = tmp_path / "no" / "such" / "dir" / "refresh-requested"
        settings = SimpleNamespace(cookie_refresh_marker_path=str(marker))
        monkeypatch.setattr("app.providers.ytdlp.get_settings", lambda: settings)

        request_cookie_refresh()


class TestBotCheckWiredIntoMediaInfo:
    """Bot-check errors must flag a refresh instead of retrying blindly."""

    def test_media_fetch_requests_refresh_and_wraps_error(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr("app.providers.media.get_settings", lambda: _media_settings(tmp_path))
        monkeypatch.setattr("app.providers.media.build_ydlp_options", lambda **_: {})
        monkeypatch.setattr("yt_dlp.YoutubeDL", _BoomYoutubeDL)
        refresh = Mock()
        monkeypatch.setattr("app.providers.media.request_cookie_refresh", refresh)

        with pytest.raises(ExternalServiceError):
            media.get_media_info("abc", "https://youtu.be/abc")

        refresh.assert_called_once()


class TestBotCheckWiredIntoTranscript:
    """Bot-check errors during caption/audio download must flag a refresh."""

    def test_caption_download_requests_refresh(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            "app.providers.transcript.get_settings", lambda: _transcript_settings(tmp_path)
        )
        monkeypatch.setattr("app.providers.transcript.build_ydlp_options", lambda **_: {})
        monkeypatch.setattr("yt_dlp.YoutubeDL", _BoomYoutubeDL)
        refresh = Mock()
        monkeypatch.setattr("app.providers.transcript.request_cookie_refresh", refresh)

        info = {"subtitles": {"en": [{"ext": "vtt"}]}}
        with pytest.raises(ExternalServiceError):
            transcript._download_caption("https://youtu.be/abc", info=info)

        refresh.assert_called_once()

    def test_extract_video_info_requests_refresh(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(
            "app.providers.transcript.get_settings", lambda: _transcript_settings(tmp_path)
        )
        monkeypatch.setattr("app.providers.transcript.build_ydlp_options", lambda **_: {})
        monkeypatch.setattr("yt_dlp.YoutubeDL", _BoomYoutubeDL)
        refresh = Mock()
        monkeypatch.setattr("app.providers.transcript.request_cookie_refresh", refresh)

        with pytest.raises(yt_dlp.utils.DownloadError):
            transcript._extract_video_info("https://youtu.be/abc")

        refresh.assert_called_once()

    def test_run_download_requests_refresh_and_re_raises(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr("yt_dlp.YoutubeDL", _BoomYoutubeDL)
        refresh = Mock()
        monkeypatch.setattr("app.providers.transcript.request_cookie_refresh", refresh)

        with pytest.raises(yt_dlp.utils.DownloadError):
            transcript._run_download({}, "https://youtu.be/abc")

        refresh.assert_called_once()


class TestNetscapeExport:
    """Tests for CDP-cookie -> Netscape conversion."""

    def test_filters_to_relevant_domains(self) -> None:
        rows = to_netscape_rows(
            [
                {"name": "SID", "value": "1", "domain": ".youtube.com", "path": "/"},
                {"name": "NID", "value": "2", "domain": ".google.com", "path": "/"},
                {"name": "tracking", "value": "3", "domain": ".example.com", "path": "/"},
            ]
        )
        assert {row["name"] for row in rows} == {"SID", "NID"}

    def test_sets_include_subdomain_flag_from_leading_dot(self) -> None:
        rows = to_netscape_rows(
            [
                {"name": "a", "value": "1", "domain": ".youtube.com", "path": "/"},
                {"name": "b", "value": "2", "domain": "accounts.youtube.com", "path": "/"},
            ]
        )
        by_name = {row["name"]: row for row in rows}
        assert by_name["a"]["include_subdomains"] == "TRUE"
        assert by_name["b"]["include_subdomains"] == "FALSE"

    def test_maps_session_expires_to_zero(self) -> None:
        rows = to_netscape_rows(
            [
                {"name": "YSC", "value": "1", "domain": ".youtube.com", "expires": -1},
                {"name": "NID", "value": "2", "domain": ".google.com", "expires": None},
            ]
        )
        by_name = {row["name"]: row for row in rows}
        assert by_name["YSC"]["expires"] == 0
        assert by_name["NID"]["expires"] == 0

    def test_defaults_path_and_secure(self) -> None:
        rows = to_netscape_rows(
            [{"name": "SID", "value": "1", "domain": ".google.com", "expires": 0}]
        )
        assert rows[0]["path"] == "/"
        assert rows[0]["secure"] == "FALSE"
        assert rows[0]["expires"] == 0

    def test_write_cookies_rejects_when_not_logged_in(self, tmp_path) -> None:
        output = tmp_path / "cookies.txt"
        rc = write_cookies([], output)
        assert rc == EXIT_NOT_LOGGED_IN
        assert not output.exists()

    def test_write_cookies_rejects_without_auth_cookie(self, tmp_path) -> None:
        output = tmp_path / "cookies.txt"
        cookies = [{"name": "ANID", "value": "1", "domain": ".google.com", "path": "/"}]
        rc = write_cookies(cookies, output)
        assert rc == EXIT_NOT_LOGGED_IN
        assert not output.exists()

    def test_write_cookies_writes_netscape_file(self, tmp_path) -> None:
        output = tmp_path / "cookies.txt"
        cookies = [{"name": "SID", "value": "v", "domain": ".youtube.com", "path": "/"}]
        rc = write_cookies(cookies, output)
        assert rc == EXIT_OK
        body = output.read_text().splitlines()
        assert body[0] == "# Netscape HTTP Cookie File"
        assert ".youtube.com\tTRUE\t/\tFALSE\t0\tSID\tv" in body
