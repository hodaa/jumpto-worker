"""Unit tests for the video source registry and its adapters."""

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import ExternalServiceError, PermanentExternalServiceError
from app.sources import drive as drive_module
from app.sources import resolve_source
from app.sources.base import VideoRef
from app.sources.drive import DriveSource
from app.sources.youtube import YouTubeSource, is_youtube_host

DRIVE_ID = "1DOoJhC6NHpLSu2nAvaWy3HukVimEyyhE"
DRIVE_FILE_URL = f"https://drive.google.com/file/d/{DRIVE_ID}/view"

_INTERSTITIAL = (
    '<html><body><form action="https://drive.usercontent.google.com/download">'
    f'<input type="hidden" name="id" value="{DRIVE_ID}">'
    '<input type="hidden" name="export" value="download">'
    '<input type="hidden" name="confirm" value="t">'
    '<input type="hidden" name="uuid" value="uuid-1">'
    "</form></body></html>"
)

_PAGE_HEADERS = {"content-type": "text/html; charset=utf-8"}
_VIDEO_HEADERS = {
    "content-type": "video/x-matroska",
    "content-disposition": 'attachment; filename="Lec 2.m4v"',
}


def _use_transport(monkeypatch, handler) -> list[httpx.Request]:
    """Route Drive HTTP through a MockTransport and record its requests."""
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    monkeypatch.setattr(drive_module, "_transport", lambda: httpx.MockTransport(record))
    return requests


def _use_settings(monkeypatch, **overrides) -> None:
    """Pin the Drive source's settings for one test."""
    values = {"google_api_key": ""}
    values.update(overrides)
    settings = Settings(**values)
    monkeypatch.setattr(drive_module, "get_settings", lambda: settings)


def _stream_response(body: bytes, headers: dict) -> httpx.Response:
    return httpx.Response(200, headers=headers, content=body)


class TestResolveSource:
    """The registry picks the source that owns the URL's host."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=abcde12345",
            "https://youtu.be/abcde12345",
            "https://m.youtube.com/watch?v=abcde12345",
            "https://music.youtube.com/watch?v=abcde12345",
        ],
    )
    def test_youtube_hosts_resolve_to_the_caption_bearing_source(self, url: str) -> None:
        source = resolve_source(url)

        assert isinstance(source, YouTubeSource)
        assert source.name == "youtube"
        assert source.native_captions is True

    @pytest.mark.parametrize(
        "url",
        [
            DRIVE_FILE_URL,
            f"https://drive.google.com/open?id={DRIVE_ID}",
            f"https://drive.google.com/uc?export=download&id={DRIVE_ID}",
        ],
    )
    def test_drive_links_resolve_to_the_captionless_source(self, url: str) -> None:
        source = resolve_source(url)

        assert isinstance(source, DriveSource)
        assert source.name == "drive"
        assert source.native_captions is False

    @pytest.mark.parametrize("url", ["https://vimeo.com/123", "", "not a url"])
    def test_unclaimed_urls_fail_permanently(self, url: str) -> None:
        with pytest.raises(PermanentExternalServiceError) as excinfo:
            resolve_source(url)

        assert "Unsupported video URL" in str(excinfo.value)


class TestIsYoutubeHost:
    """YouTube host matching claims every subdomain and nothing else."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://youtube.com/watch?v=x",
            "https://www.youtube.com/watch?v=x",
            "https://m.youtube.com/watch?v=x",
            "https://youtu.be/x",
        ],
    )
    def test_youtube_shapes_match(self, url: str) -> None:
        assert is_youtube_host(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://notyoutube.com/watch?v=x",
            "https://youtube.com.evil.test/x",
            "https://vimeo.com/x",
        ],
    )
    def test_lookalike_hosts_do_not_match(self, url: str) -> None:
        assert is_youtube_host(url) is False


class TestYouTubeSourceIdentify:
    """YouTube identification delegates to the shared video-id parser."""

    def test_watch_url_yields_the_video_id(self) -> None:
        ref = YouTubeSource().identify("https://www.youtube.com/watch?v=abcde12345")

        assert ref == VideoRef(
            source="youtube",
            source_id="abcde12345",
            canonical_url="https://www.youtube.com/watch?v=abcde12345",
        )

    def test_short_url_yields_the_video_id(self) -> None:
        ref = YouTubeSource().identify("https://youtu.be/abcde12345?t=30")

        assert ref.source_id == "abcde12345"

    def test_url_without_an_id_raises(self) -> None:
        with pytest.raises(ExternalServiceError):
            YouTubeSource().identify("https://www.youtube.com/")


class TestDriveIdentify:
    """Drive identification accepts file links and rejects everything else."""

    @pytest.mark.parametrize(
        "url",
        [
            f"https://drive.google.com/file/d/{DRIVE_ID}/view?usp=sharing",
            f"https://drive.google.com/open?id={DRIVE_ID}",
            f"https://drive.google.com/uc?export=download&id={DRIVE_ID}",
            f"https://drive.google.com/drive/u/0/my-drive?id={DRIVE_ID}",
        ],
    )
    def test_file_links_normalize_to_the_canonical_url(self, url: str) -> None:
        ref = DriveSource().identify(url)

        assert ref == VideoRef(
            source="drive",
            source_id=DRIVE_ID,
            canonical_url=f"https://drive.google.com/file/d/{DRIVE_ID}",
        )

    def test_folder_links_fail_permanently(self) -> None:
        with pytest.raises(PermanentExternalServiceError) as excinfo:
            DriveSource().identify(f"https://drive.google.com/drive/folders/{DRIVE_ID}")

        assert "does not point at a file" in str(excinfo.value)

    def test_short_ids_are_not_file_ids(self) -> None:
        with pytest.raises(PermanentExternalServiceError):
            DriveSource().identify("https://drive.google.com/file/d/abc123/view")


class TestDownloadMedia:
    """download_media streams the file to its advertised filename."""

    @pytest.fixture(autouse=True)
    def _web_endpoint(self, monkeypatch) -> None:
        """Default download tests to the key-less browser endpoint."""
        _use_settings(monkeypatch)

    def test_direct_download_keeps_the_uploader_filename(self, monkeypatch, tmp_path: Path) -> None:
        requests = _use_transport(
            monkeypatch, lambda request: _stream_response(b"video-bytes", _VIDEO_HEADERS)
        )

        produced = drive_module.download_media(DRIVE_ID, tmp_path)

        assert produced == tmp_path / "Lec 2.m4v"
        assert produced.read_bytes() == b"video-bytes"
        assert not (tmp_path / drive_module._PARTIAL_FILENAME).exists()
        assert str(requests[0].url) == drive_module._download_url(DRIVE_ID)

    def test_confirm_interstitial_is_resubmitted_with_its_fields(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "drive.google.com":
                return _stream_response(_INTERSTITIAL.encode(), _PAGE_HEADERS)
            return _stream_response(b"video-bytes", _VIDEO_HEADERS)

        requests = _use_transport(monkeypatch, handler)

        produced = drive_module.download_media(DRIVE_ID, tmp_path)

        assert produced.read_bytes() == b"video-bytes"
        assert len(requests) == 2
        confirm = requests[1].url
        assert confirm.host == "drive.usercontent.google.com"
        assert confirm.params["confirm"] == "t"
        assert confirm.params["uuid"] == "uuid-1"
        assert confirm.params["id"] == DRIVE_ID

    def test_access_error_page_fails_permanently(self, monkeypatch, tmp_path: Path) -> None:
        page = "<html><body>This file is in your trash.</body></html>"
        _use_transport(monkeypatch, lambda request: _stream_response(page.encode(), _PAGE_HEADERS))

        with pytest.raises(PermanentExternalServiceError) as excinfo:
            drive_module.download_media(DRIVE_ID, tmp_path)

        assert "not public or was not found" in str(excinfo.value)
        assert not (tmp_path / drive_module._PARTIAL_FILENAME).exists()

    def test_html_after_the_confirm_also_fails_permanently(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _stream_response(_INTERSTITIAL.encode() + b"<html>", _PAGE_HEADERS)

        _use_transport(monkeypatch, handler)

        with pytest.raises(PermanentExternalServiceError):
            drive_module.download_media(DRIVE_ID, tmp_path)

    def test_empty_body_is_a_transient_failure(self, monkeypatch, tmp_path: Path) -> None:
        _use_transport(monkeypatch, lambda request: _stream_response(b"", _VIDEO_HEADERS))

        with pytest.raises(ExternalServiceError) as excinfo:
            drive_module.download_media(DRIVE_ID, tmp_path)

        assert "produced no file" in str(excinfo.value)

    def test_missing_disposition_falls_back_to_a_safe_name(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        _use_transport(
            monkeypatch,
            lambda request: _stream_response(b"x", {"content-type": "video/mp4"}),
        )

        produced = drive_module.download_media(DRIVE_ID, tmp_path)

        assert produced.name == drive_module._DEFAULT_FILENAME

    def test_hostile_disposition_cannot_escape_the_directory(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        headers = {
            "content-type": "video/mp4",
            "content-disposition": 'attachment; filename="../../evil.mp4"',
        }
        _use_transport(monkeypatch, lambda request: _stream_response(b"x", headers))

        produced = drive_module.download_media(DRIVE_ID, tmp_path)

        assert produced.parent == tmp_path
        assert "/" not in produced.name

    def test_api_key_downloads_through_the_drive_api(self, monkeypatch, tmp_path: Path) -> None:
        _use_settings(
            monkeypatch,
            google_api_key="test-key",
            google_drive_api_url="https://drive.test/v3",
        )
        requests = _use_transport(
            monkeypatch, lambda request: _stream_response(b"video-bytes", _VIDEO_HEADERS)
        )

        produced = drive_module.download_media(DRIVE_ID, tmp_path)

        assert produced.read_bytes() == b"video-bytes"
        assert str(requests[0].url) == drive_module._api_download_url(
            "https://drive.test/v3", DRIVE_ID
        )
        assert requests[0].headers["x-goog-api-key"] == "test-key"

    def test_api_access_denied_fails_permanently(self, monkeypatch, tmp_path: Path) -> None:
        _use_settings(monkeypatch, google_api_key="test-key")
        _use_transport(monkeypatch, lambda request: httpx.Response(403))

        with pytest.raises(PermanentExternalServiceError):
            drive_module.download_media(DRIVE_ID, tmp_path)

    def test_api_invalid_key_fails_permanently(self, monkeypatch, tmp_path: Path) -> None:
        _use_settings(monkeypatch, google_api_key="bad-key")
        _use_transport(monkeypatch, lambda request: httpx.Response(400))

        with pytest.raises(PermanentExternalServiceError) as excinfo:
            drive_module.download_media(DRIVE_ID, tmp_path)

        assert "GOOGLE_API_KEY" in str(excinfo.value)

    def test_api_server_error_is_transient(self, monkeypatch, tmp_path: Path) -> None:
        _use_settings(monkeypatch, google_api_key="test-key")
        _use_transport(monkeypatch, lambda request: httpx.Response(503))

        with pytest.raises(ExternalServiceError) as excinfo:
            drive_module.download_media(DRIVE_ID, tmp_path)

        assert "could not serve" in str(excinfo.value)


class TestDriveDownload:
    """DriveSource.download fetches the whole file, cutting when asked."""

    def test_plain_download_returns_the_media_without_ffmpeg(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        media = tmp_path / "Lec 2.m4v"
        media.write_bytes(b"data")
        monkeypatch.setattr(drive_module, "download_media", lambda file_id, directory: media)
        monkeypatch.setattr(
            drive_module, "_run_ffmpeg", lambda *args: pytest.fail("no cut expected")
        )

        produced = DriveSource().download(DRIVE_FILE_URL, tmp_path, height=480)

        assert produced == media

    def test_section_download_cuts_from_the_fetched_file(self, monkeypatch, tmp_path: Path) -> None:
        media = tmp_path / "Lec 2.m4v"
        media.write_bytes(b"data")
        cuts: list[list[str]] = []

        def fake_ffmpeg(args: list[str], failure_message: str) -> None:
            cuts.append(args)
            Path(args[-1]).write_bytes(b"cut")

        monkeypatch.setattr(drive_module, "download_media", lambda file_id, directory: media)
        monkeypatch.setattr(drive_module, "_run_ffmpeg", fake_ffmpeg)

        produced = DriveSource().download(
            DRIVE_FILE_URL, tmp_path, start_seconds=75, end_seconds=120
        )

        assert produced == tmp_path / "Lec 2.mp4"
        assert produced.read_bytes() == b"cut"
        assert cuts[0][1:3] == ["-ss", "75"]
        assert cuts[0][5:7] == ["-t", "45"]
        assert "-c" in cuts[0] and "copy" in cuts[0]

    def test_open_section_runs_to_the_end_of_the_file(self, monkeypatch, tmp_path: Path) -> None:
        media = tmp_path / "movie.mkv"
        media.write_bytes(b"data")
        cuts: list[list[str]] = []
        monkeypatch.setattr(drive_module, "download_media", lambda file_id, directory: media)
        monkeypatch.setattr(
            drive_module,
            "_run_ffmpeg",
            lambda args, failure_message: (cuts.append(args), Path(args[-1]).write_bytes(b"x")),
        )

        DriveSource().download(DRIVE_FILE_URL, tmp_path, start_seconds=5)

        assert "-t" not in cuts[0]

    def test_ffmpeg_producing_no_file_fails(self, monkeypatch, tmp_path: Path) -> None:
        media = tmp_path / "movie.mkv"
        media.write_bytes(b"data")
        monkeypatch.setattr(drive_module, "download_media", lambda file_id, directory: media)
        monkeypatch.setattr(drive_module, "_run_ffmpeg", lambda args, failure_message: None)

        with pytest.raises(ExternalServiceError) as excinfo:
            DriveSource().download(DRIVE_FILE_URL, tmp_path, start_seconds=1)

        assert "no file" in str(excinfo.value)


class TestProbeDuration:
    """Duration probing degrades to 0 instead of failing the job."""

    def test_reads_the_format_duration(self, monkeypatch) -> None:
        monkeypatch.setattr(
            drive_module.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0, stdout='{"format": {"duration": "12.7"}}'
            ),
        )

        assert drive_module.probe_duration_seconds(Path("x.mp4")) == 12

    def test_failed_probe_yields_zero(self, monkeypatch) -> None:
        monkeypatch.setattr(
            drive_module.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""),
        )

        assert drive_module.probe_duration_seconds(Path("x.mp4")) == 0

    def test_unparseable_output_yields_zero(self, monkeypatch) -> None:
        monkeypatch.setattr(
            drive_module.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="not json"),
        )

        assert drive_module.probe_duration_seconds(Path("x.mp4")) == 0


class TestExtractAudio:
    """Audio extraction produces the AAC file the transcription uploads."""

    def test_successful_extraction_returns_the_audio(self, monkeypatch, tmp_path: Path) -> None:
        media = tmp_path / "Lec 2.m4v"
        media.write_bytes(b"data")
        monkeypatch.setattr(
            drive_module,
            "_run_ffmpeg",
            lambda args, failure_message: Path(args[-1]).write_bytes(b"audio"),
        )

        produced = drive_module.extract_audio(media, tmp_path)

        assert produced == tmp_path / drive_module._AUDIO_FILENAME
        assert produced.read_bytes() == b"audio"

    def test_empty_audio_fails(self, monkeypatch, tmp_path: Path) -> None:
        media = tmp_path / "Lec 2.m4v"
        media.write_bytes(b"data")
        monkeypatch.setattr(drive_module, "_run_ffmpeg", lambda args, failure_message: None)

        with pytest.raises(ExternalServiceError):
            drive_module.extract_audio(media, tmp_path)


class TestRunFfmpeg:
    """ffmpeg failures surface as typed errors, never a silent success."""

    def test_nonzero_exit_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(
            drive_module.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="boom"),
        )

        with pytest.raises(ExternalServiceError) as excinfo:
            drive_module._run_ffmpeg(["-i", "x", "y"], "Could not cut the requested section")

        assert "Could not cut the requested section" in str(excinfo.value)

    def test_zero_exit_passes(self, monkeypatch) -> None:
        monkeypatch.setattr(
            drive_module.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=""),
        )

        drive_module._run_ffmpeg(["-i", "x", "y"], "unused")


class TestPrepareAudio:
    """prepare_audio fetches once and reads title, duration and audio from it."""

    def test_reads_metadata_and_removes_the_media(self, monkeypatch, tmp_path: Path) -> None:
        media = tmp_path / "Lec 2.m4v"
        media.write_bytes(b"data")
        monkeypatch.setattr(drive_module, "download_media", lambda file_id, directory: media)
        monkeypatch.setattr(drive_module, "probe_duration_seconds", lambda path: 3600)
        monkeypatch.setattr(
            drive_module,
            "_run_ffmpeg",
            lambda args, failure_message: Path(args[-1]).write_bytes(b"audio"),
        )

        prepared = drive_module.DriveSource().prepare_audio(DRIVE_FILE_URL, tmp_path)

        assert prepared.title == "Lec 2"
        assert prepared.duration_seconds == 3600
        assert prepared.audio_path == tmp_path / drive_module._AUDIO_FILENAME
        assert prepared.directory == tmp_path
        assert not media.exists()

    def test_media_is_removed_even_when_extraction_fails(self, monkeypatch, tmp_path: Path) -> None:
        media = tmp_path / "Lec 2.m4v"
        media.write_bytes(b"data")
        monkeypatch.setattr(drive_module, "download_media", lambda file_id, directory: media)
        monkeypatch.setattr(drive_module, "probe_duration_seconds", lambda path: 0)

        def boom(args: list[str], failure_message: str) -> None:
            raise ExternalServiceError(failure_message, service="drive")

        monkeypatch.setattr(drive_module, "_run_ffmpeg", boom)

        with pytest.raises(ExternalServiceError):
            drive_module.DriveSource().prepare_audio(DRIVE_FILE_URL, tmp_path)

        assert not media.exists()
