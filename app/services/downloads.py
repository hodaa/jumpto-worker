"""Produce offloaded video downloads and probe quality ladders.

The backend queues ``video_downloads`` rows (and occasional format probes)
onto the shared worker queue; this module is the worker-side side of that
work: resolve the request through the backend API, download the asked MP4
through the URL's video source (yt-dlp for YouTube, an HTTP fetch for Drive),
upload the artifact to blob storage, and report the blob URL back. The probe
path reuses the same yt-dlp plumbing to read a video's H.264 quality ladder
for the backend's formats cache; sources without a ladder report an empty one.

yt-dlp is blocking, so every run is pushed onto a thread. A bot-check
touches the host's cookie-refresh marker, exactly like the other yt-dlp
call sites in the worker.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import yt_dlp
from yt_dlp.utils import download_range_func

from app.client import BackendClient
from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.integrations.blob import upload_mp4
from app.integrations.ytdlp import (
    build_ydlp_options,
    is_youtube_bot_check,
    release_temp_cookie,
    request_cookie_refresh,
)
from app.models import DownloadData
from app.services.jobs import default_client_factory
from app.sources import resolve_source

logger = get_logger(__name__)

# H.264 (avc1) plus AAC where possible - the codecs Apple players and TVs
# decode - falling back to a pre-merged MP4 and finally to whatever plays.
_FORMAT_SELECTOR = "bv*[vcodec^=avc1]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b"

# Container merge output; also the produced file's extension.
_MERGE_FORMAT = "mp4"

# Bounds of the offerable H.264 ladder: below the minimum a quality tier is
# useless, above the maximum YouTube only serves VP9/AV1 which Apple players
# cannot decode.
_MIN_HEIGHT = 144
_MAX_HEIGHT = 1080

#: User-safe message reported when production fails.
FAILURE_SAFE_MESSAGE = "Could not prepare the video download. Please try again later."

_TEMP_PREFIX = "jumpto-download-"
_OUTPUT_STEM = "video"
_VIDEO_SERVICE = "yt-dlp"

T = TypeVar("T")


def fetch_video_file(
    youtube_url: str,
    directory: Path,
    *,
    start_seconds: int | None = None,
    end_seconds: int | None = None,
    height: int | None = None,
    settings=None,
) -> Path:
    """Download one MP4 into ``directory``; the URL's source does the fetching.

    With ``start_seconds`` set only that section is downloaded, ending at
    ``end_seconds`` or, when omitted, at the end of the video; with ``height``
    set the pick is capped at that quality where a ladder exists.
    """
    source = resolve_source(youtube_url)
    return source.download(
        youtube_url,
        directory,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        height=height,
        settings=settings,
    )


def fetch_video_file_with_ytdlp(
    youtube_url: str,
    directory: Path,
    *,
    start_seconds: int | None = None,
    end_seconds: int | None = None,
    height: int | None = None,
    settings=None,
) -> Path:
    """Download and merge one YouTube MP4 into ``directory``; return its path.

    This is the YouTube implementation behind
    :class:`~app.sources.youtube.YouTubeSource`; section and quality behavior
    is unchanged from the days it was the only source.
    """
    options = build_ydlp_options(
        settings=settings,
        format=_format_selector(height),
        merge_output_format=_MERGE_FORMAT,
        outtmpl=str(directory / f"{_OUTPUT_STEM}.%(ext)s"),
    )
    if start_seconds is not None:
        options["download_ranges"] = _section_ranges(start_seconds, end_seconds)
    _run_ytdlp(options, lambda ydl: ydl.download([youtube_url]))
    produced = directory / f"{_OUTPUT_STEM}.{_MERGE_FORMAT}"
    if not produced.is_file() or produced.stat().st_size == 0:
        raise ExternalServiceError("yt-dlp produced no video file", service=_VIDEO_SERVICE)
    return produced


def probe_heights(youtube_url: str, *, settings=None) -> list[int]:
    """Probe a video's H.264 quality ladder (metadata only), tallest first.

    Only caption-bearing sources (YouTube) have a ladder to report; a Drive
    file is a single upload, so its probe is an empty ladder without a call.
    """
    if not resolve_source(youtube_url).native_captions:
        return []
    options = build_ydlp_options(settings=settings)
    info = _run_ytdlp(options, lambda ydl: ydl.extract_info(youtube_url, download=False))
    return _ladder_from_info(info or {})


async def run_download(
    download_id: str,
    client_factory: Callable[..., BackendClient] | None = None,
) -> str:
    """Produce one download end to end and report its blob URL."""
    settings = get_settings()
    client = (client_factory or default_client_factory)(settings)
    request = DownloadData(**await client.get_video_download(download_id))
    await client.advance_download(download_id)
    blob_url = await _produce_and_upload(request, settings)
    await client.mark_download_ready(download_id, blob_url)
    return blob_url


async def fail_download(
    download_id: str,
    message: str,
    client_factory: Callable[..., BackendClient] | None = None,
) -> None:
    """Report a failed download to the backend, best effort."""
    settings = get_settings()
    client = (client_factory or default_client_factory)(settings)
    try:
        await client.mark_download_failed(download_id, message)
    except Exception:
        logger.exception("Could not report download failure", download_id=download_id)


async def probe_and_store_formats(
    youtube_video_id: str,
    youtube_url: str,
    client_factory: Callable[..., BackendClient] | None = None,
) -> list[int]:
    """Probe a video's ladder and cache it in the backend."""
    settings = get_settings()
    client = (client_factory or default_client_factory)(settings)
    heights = await asyncio.to_thread(probe_heights, youtube_url, settings=settings)
    await client.store_video_formats(youtube_video_id, heights)
    return heights


async def _produce_and_upload(request: DownloadData, settings) -> str:
    """Download the requested MP4 into a temp dir, upload it, clean up."""
    directory = Path(tempfile.mkdtemp(prefix=_TEMP_PREFIX))
    try:
        path = await asyncio.to_thread(
            fetch_video_file,
            request.youtube_url,
            directory,
            start_seconds=request.start_seconds,
            end_seconds=request.end_seconds,
            height=request.height,
            settings=settings,
        )
        return await asyncio.to_thread(upload_mp4, path)
    finally:
        await asyncio.to_thread(shutil.rmtree, directory, ignore_errors=True)


def _run_ytdlp(options: dict, action: Callable[[yt_dlp.YoutubeDL], T]) -> T:
    """Run one yt-dlp session, handling the bot-check marker and cleanup."""
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            return action(ydl)
    except yt_dlp.utils.DownloadError as exc:
        if is_youtube_bot_check(exc):
            request_cookie_refresh()
        raise
    finally:
        release_temp_cookie(options)


def _format_selector(height: int | None) -> str:
    """Build the format selector, capped at ``height`` when one is asked for.

    The capped ladder keeps the documented preference order and ends with the
    uncapped fallback, so a video without that height still yields a file.
    """
    if height is None:
        return _FORMAT_SELECTOR
    return (
        f"bv*[height<={height}][vcodec^=avc1]+ba[ext=m4a]"
        f"/b[height<={height}][ext=mp4]"
        "/bv*+ba/b"
    )


def _section_ranges(start_seconds: int, end_seconds: int | None) -> download_range_func:
    """Build yt-dlp section ranges; an open end runs to the end of the video."""
    end = float(end_seconds) if end_seconds is not None else float("inf")
    return download_range_func(None, [(float(start_seconds), end)])


def _ladder_from_info(info: dict) -> list[int]:
    """Collect the distinct H.264 heights in probe metadata, tallest first."""
    heights = set()
    for fmt in info.get("formats") or []:
        if not isinstance(fmt, dict):
            continue
        height = fmt.get("height")
        if (
            str(fmt.get("vcodec") or "").startswith("avc1")
            and isinstance(height, int)
            and _MIN_HEIGHT <= height <= _MAX_HEIGHT
        ):
            heights.add(height)
    return sorted(heights, reverse=True)
