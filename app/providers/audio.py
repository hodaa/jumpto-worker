"""Audio stream download helpers for the yt-dlp provider tree.

Both audio-transcription leaves (Assembly, Deepgram) transcribe the same
downloaded audio stream. This module is the single place that knows how to
fetch a temp audio file with yt-dlp, so that logic stays reusable and is never
duplicated across leaves.
"""

import contextlib
import os
import tempfile
from pathlib import Path

import yt_dlp

from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.integrations.ytdlp import (
    build_ydlp_options,
    is_youtube_bot_check,
    release_temp_cookie,
    request_cookie_refresh,
)

logger = get_logger(__name__)


def download_audio(youtube_url: str) -> str:
    """Download a YouTube audio stream to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".webm")
    os.close(fd)
    destination = Path(path)
    destination.unlink(missing_ok=True)
    options = build_ydlp_options(
        format="bestaudio/best",
        outtmpl=path,
    )
    try:
        run_download(options, youtube_url)
        if not destination.exists() or destination.stat().st_size == 0:
            logger.error("Audio download produced no file", path=path)
            raise ExternalServiceError("Audio download produced no file", service="yt-dlp")
        return path
    except ExternalServiceError:
        remove_file(path)
        raise
    except Exception as exc:
        remove_file(path)
        logger.error("Audio download failed", error=str(exc))
        raise ExternalServiceError("Could not download audio", service="yt-dlp") from exc


def run_download(options: dict, youtube_url: str) -> None:
    """Run a fresh yt-dlp audio download for a URL.

    Always extract and download the URL rather than replaying a previously
    extracted ``info`` dict through ``process_ie_result``: YouTube expires the
    video-serving URLs inside a stored ``info`` within seconds, so replays fail
    with HTTP 403 when the caption fast path missed and we fall back to audio.
    """
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([youtube_url])
    except yt_dlp.utils.DownloadError as exc:
        if is_youtube_bot_check(exc):
            request_cookie_refresh()
        raise
    finally:
        release_temp_cookie(options)


def remove_file(path: str) -> None:
    """Best-effort removal of a temp audio file."""
    with contextlib.suppress(OSError):
        Path(path).unlink()
