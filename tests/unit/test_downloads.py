"""Unit tests for offloaded download production and format probing."""

from pathlib import Path

import pytest
import yt_dlp

from app.core.config import Settings
from app.core.exceptions import ExternalServiceError
from app.services import downloads
from app.services.downloads import (
    FAILURE_SAFE_MESSAGE,
    _format_selector,
    _ladder_from_info,
    _section_ranges,
    fail_download,
    fetch_video_file,
    probe_and_store_formats,
    probe_heights,
    run_download,
)

_BOT_CHECK_MESSAGE = "ERROR: Sign in to confirm you're not a bot"


def _settings() -> Settings:
    """Build isolated worker settings with no cookie or bgutil wiring."""
    return Settings(
        _env_file=None,
        ytdlp_cookie_file="",
        mounted_ytdlp_cookie_file="",
        ytdlp_bgutil_url="",
    )


class _FakeYoutubeDL:
    """Stand-in for yt-dlp that writes an output file when asked to download.

    ``errors``, ``info`` and ``write_output`` are class-level knobs the tests
    set before calling; the fixture resets them between tests.
    """

    instances: list["_FakeYoutubeDL"] = []
    errors: list[BaseException] = []
    info: dict = {}
    write_output: bool = True

    def __init__(self, options: dict) -> None:
        self.options = options
        self.downloaded: list[str] = []
        self.extracted: list[str] = []
        self.instances.append(self)

    def __enter__(self) -> "_FakeYoutubeDL":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def download(self, urls: list[str]) -> None:
        self.downloaded.extend(urls)
        if type(self).errors:
            raise type(self).errors.pop(0)
        if type(self).write_output:
            Path(self.options["outtmpl"].replace("%(ext)s", "mp4")).write_bytes(b"video")

    def extract_info(self, url: str, download: bool = True) -> dict:
        self.extracted.append(url)
        if type(self).errors:
            raise type(self).errors.pop(0)
        return type(self).info


@pytest.fixture
def fake_ytdlp(monkeypatch) -> type[_FakeYoutubeDL]:
    """Install a fake YoutubeDL and reset its state per test."""
    _FakeYoutubeDL.instances = []
    _FakeYoutubeDL.errors = []
    _FakeYoutubeDL.info = {}
    _FakeYoutubeDL.write_output = True
    monkeypatch.setattr(downloads.yt_dlp, "YoutubeDL", _FakeYoutubeDL)
    return _FakeYoutubeDL


@pytest.fixture
def released(monkeypatch) -> list[dict]:
    """Record release_temp_cookie calls and stub them out."""
    calls: list[dict] = []
    monkeypatch.setattr(downloads, "release_temp_cookie", calls.append)
    return calls


@pytest.fixture
def cookie_refreshed(monkeypatch) -> list[bool]:
    """Record request_cookie_refresh calls and stub them out."""
    calls: list[bool] = []
    monkeypatch.setattr(downloads, "request_cookie_refresh", lambda: calls.append(True))
    return calls


class TestFormatSelector:
    """The format ladder must mirror the sync implementation's ordering."""

    def test_uncapped_selector_is_the_documented_default(self) -> None:
        assert _format_selector(None) == downloads._FORMAT_SELECTOR

    def test_capped_selector_keeps_order_and_uncapped_fallback(self) -> None:
        selector = _format_selector(480)
        assert selector == (
            "bv*[height<=480][vcodec^=avc1]+ba[ext=m4a]/b[height<=480][ext=mp4]/bv*+ba/b"
        )


class TestSectionRanges:
    """Section ranges drive yt-dlp's --download-sections equivalent."""

    def test_closed_range_yields_start_and_end(self) -> None:
        ranges = _section_ranges(10, 30)
        assert list(ranges({"id": "x"}, None)) == [{"start_time": 10.0, "end_time": 30.0}]

    def test_open_range_runs_to_the_end_of_the_video(self) -> None:
        ranges = _section_ranges(5, None)
        assert list(ranges({"id": "x"}, None)) == [{"start_time": 5.0, "end_time": float("inf")}]


class TestLadderFromInfo:
    """The probe ladder filters to the decodable H.264 heights."""

    def test_collects_avc1_heights_deduplicated_and_descending(self) -> None:
        info = {
            "formats": [
                {"vcodec": "avc1.640028", "height": 720},
                {"vcodec": "avc1.640028", "height": 720},
                {"vcodec": "avc1.640028", "height": 1080},
                {"vcodec": "vp9", "height": 2160},
                {"vcodec": "avc1.640028", "height": 2160},
                {"vcodec": "avc1.640028", "height": 100},
                {"vcodec": "avc1.640028", "height": "720"},
                "not-a-dict",
            ]
        }
        assert _ladder_from_info(info) == [1080, 720]

    def test_empty_metadata_yields_empty_ladder(self) -> None:
        assert _ladder_from_info({}) == []


class TestFetchVideoFile:
    """fetch_video_file downloads and validates the produced MP4."""

    def test_downloads_with_default_options(
        self, tmp_path: Path, fake_ytdlp, released: list[dict]
    ) -> None:
        produced = fetch_video_file("https://youtu.be/x", tmp_path, settings=_settings())

        assert produced == tmp_path / "video.mp4"
        instance = fake_ytdlp.instances[0]
        assert instance.downloaded == ["https://youtu.be/x"]
        assert instance.options["format"] == downloads._FORMAT_SELECTOR
        assert instance.options["merge_output_format"] == "mp4"
        assert instance.options["outtmpl"] == str(tmp_path / "video.%(ext)s")
        assert "download_ranges" not in instance.options
        assert released == [instance.options]

    def test_capped_download_wires_height_and_section(
        self, tmp_path: Path, fake_ytdlp, released: list[dict]
    ) -> None:
        fetch_video_file(
            "https://youtu.be/x",
            tmp_path,
            start_seconds=10,
            end_seconds=30,
            height=480,
            settings=_settings(),
        )

        instance = fake_ytdlp.instances[0]
        assert "[height<=480]" in instance.options["format"]
        ranges = instance.options["download_ranges"]
        assert list(ranges({"id": "x"}, None)) == [{"start_time": 10.0, "end_time": 30.0}]

    def test_missing_output_raises_external_error(
        self, tmp_path: Path, fake_ytdlp, released: list[dict]
    ) -> None:
        fake_ytdlp.write_output = False

        with pytest.raises(ExternalServiceError):
            fetch_video_file("https://youtu.be/x", tmp_path, settings=_settings())

    def test_bot_check_error_requests_a_cookie_refresh(
        self, tmp_path: Path, fake_ytdlp, released: list[dict], cookie_refreshed: list[bool]
    ) -> None:
        fake_ytdlp.errors = [yt_dlp.utils.DownloadError(_BOT_CHECK_MESSAGE)]

        with pytest.raises(yt_dlp.utils.DownloadError):
            fetch_video_file("https://youtu.be/x", tmp_path, settings=_settings())

        assert cookie_refreshed == [True]
        assert released == [fake_ytdlp.instances[0].options]

    def test_other_download_errors_do_not_request_a_refresh(
        self, tmp_path: Path, fake_ytdlp, released: list[dict], cookie_refreshed: list[bool]
    ) -> None:
        fake_ytdlp.errors = [yt_dlp.utils.DownloadError("HTTP Error 403: Forbidden")]

        with pytest.raises(yt_dlp.utils.DownloadError):
            fetch_video_file("https://youtu.be/x", tmp_path, settings=_settings())

        assert cookie_refreshed == []


class TestProbeHeights:
    """probe_heights reads the ladder from metadata without downloading."""

    def test_metadata_only_and_cleanup(self, fake_ytdlp, released: list[dict]) -> None:
        fake_ytdlp.info = {"formats": [{"vcodec": "avc1.640028", "height": 720}]}

        heights = probe_heights("https://youtu.be/x", settings=_settings())

        assert heights == [720]
        instance = fake_ytdlp.instances[0]
        assert instance.extracted == ["https://youtu.be/x"]
        assert "format" not in instance.options
        assert released == [instance.options]


class _FakeClient:
    """Backend client double recording download/formats calls."""

    def __init__(self, payload: dict | None = None) -> None:
        self.calls: list[tuple] = []
        self.payload = payload or {
            "download_id": "d-1",
            "video_id": "video-1",
            "youtube_video_id": "abcde12345",
            "youtube_url": "https://www.youtube.com/watch?v=abcde12345",
            "status": "pending",
            "start_seconds": 10,
            "end_seconds": None,
            "height": 720,
        }
        self.fail_ready = False

    async def get_video_download(self, download_id: str) -> dict:
        self.calls.append(("get", download_id))
        return self.payload

    async def advance_download(self, download_id: str) -> None:
        self.calls.append(("advance", download_id))

    async def mark_download_ready(self, download_id: str, blob_url: str) -> None:
        self.calls.append(("ready", download_id, blob_url))

    async def mark_download_failed(self, download_id: str, error: str) -> None:
        self.calls.append(("failed", download_id, error))
        if self.fail_ready:
            raise RuntimeError("backend down")

    async def store_video_formats(self, youtube_video_id: str, heights: list[int]) -> None:
        self.calls.append(("formats", youtube_video_id, tuple(heights)))


class TestRunDownload:
    """run_download resolves, produces, uploads and reports, in order."""

    @pytest.mark.asyncio
    async def test_reports_blob_url_after_producing(self, monkeypatch) -> None:
        client = _FakeClient()
        fetched: list[dict] = []

        def fake_fetch(youtube_url, directory, **kwargs) -> Path:
            fetched.append({"youtube_url": youtube_url, **kwargs})
            return directory / "video.mp4"

        monkeypatch.setattr(downloads, "get_settings", _settings)
        monkeypatch.setattr(downloads, "fetch_video_file", fake_fetch)
        monkeypatch.setattr(downloads, "upload_mp4", lambda path: "https://blob/x.mp4")

        blob_url = await run_download("d-1", client_factory=lambda settings: client)

        assert blob_url == "https://blob/x.mp4"
        assert client.calls == [
            ("get", "d-1"),
            ("advance", "d-1"),
            ("ready", "d-1", "https://blob/x.mp4"),
        ]
        assert fetched == [
            {
                "youtube_url": "https://www.youtube.com/watch?v=abcde12345",
                "start_seconds": 10,
                "end_seconds": None,
                "height": 720,
                "settings": downloads.get_settings(),
            }
        ]

    @pytest.mark.asyncio
    async def test_production_errors_propagate_before_ready(self, monkeypatch) -> None:
        client = _FakeClient()

        def boom(youtube_url, directory, **kwargs) -> Path:
            raise RuntimeError("yt-dlp exploded")

        monkeypatch.setattr(downloads, "get_settings", _settings)
        monkeypatch.setattr(downloads, "fetch_video_file", boom)

        with pytest.raises(RuntimeError):
            await run_download("d-1", client_factory=lambda settings: client)

        assert client.calls == [("get", "d-1"), ("advance", "d-1")]

    @pytest.mark.asyncio
    async def test_temp_directory_is_removed(self, monkeypatch) -> None:
        client = _FakeClient()
        directories: list[Path] = []

        def fake_fetch(youtube_url, directory, **kwargs) -> Path:
            directories.append(directory)
            return directory / "video.mp4"

        monkeypatch.setattr(downloads, "get_settings", _settings)
        monkeypatch.setattr(downloads, "fetch_video_file", fake_fetch)
        monkeypatch.setattr(downloads, "upload_mp4", lambda path: "https://blob/x.mp4")

        await run_download("d-1", client_factory=lambda settings: client)

        assert directories and not directories[0].exists()


class TestFailDownload:
    """fail_download reports the failure and never raises on backend errors."""

    @pytest.mark.asyncio
    async def test_reports_the_safe_message(self, monkeypatch) -> None:
        client = _FakeClient()
        monkeypatch.setattr(downloads, "get_settings", _settings)

        await fail_download("d-1", FAILURE_SAFE_MESSAGE, client_factory=lambda settings: client)

        assert client.calls == [("failed", "d-1", FAILURE_SAFE_MESSAGE)]

    @pytest.mark.asyncio
    async def test_swallows_backend_errors(self, monkeypatch) -> None:
        client = _FakeClient()
        client.fail_ready = True
        monkeypatch.setattr(downloads, "get_settings", _settings)

        await fail_download("d-1", FAILURE_SAFE_MESSAGE, client_factory=lambda settings: client)

        assert client.calls == [("failed", "d-1", FAILURE_SAFE_MESSAGE)]


class TestProbeAndStoreFormats:
    """probe_and_store_formats probes on a thread and caches the ladder."""

    @pytest.mark.asyncio
    async def test_probes_and_stores(self, monkeypatch) -> None:
        client = _FakeClient()
        monkeypatch.setattr(downloads, "get_settings", _settings)
        monkeypatch.setattr(downloads, "probe_heights", lambda url, settings=None: [1080, 720])

        heights = await probe_and_store_formats(
            "abcde12345",
            "https://www.youtube.com/watch?v=abcde12345",
            client_factory=lambda settings: client,
        )

        assert heights == [1080, 720]
        assert client.calls == [("formats", "abcde12345", (1080, 720))]
