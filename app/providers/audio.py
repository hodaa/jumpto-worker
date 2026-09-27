"""Audio stream download helpers for the yt-dlp provider tree.

Both audio-transcription leaves (Assembly, Deepgram) transcribe the same
downloaded audio stream. This module is the single place that knows how to
fetch a temp audio file with yt-dlp, so that logic stays reusable and is never
duplicated across leaves.

Downloads are persisted on disk keyed by YouTube video id: a failed
transcription attempt or a new job for the same video finds the file instead of
re-downloading it. The file is deleted eagerly when a job settles (transcript
obtained and cached, or the retry budget exhausted); the TTL sweep in
``download_audio`` only reclaims stragglers left behind by crashed workers or
jobs that never settled.
"""

import asyncio
import contextlib
import os
import tempfile
import time
from pathlib import Path

import yt_dlp

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.integrations.ytdlp import (
    build_ydlp_options,
    is_youtube_bot_check,
    release_temp_cookie,
    request_cookie_refresh,
)

logger = get_logger(__name__)

# Upper bound on how long a persisted audio file may outlive the job that
# downloaded it (crashes, stuck jobs); normal settlement deletes eagerly.
_AUDIO_STALE_TTL_SECONDS = 3600


def audio_cache_path(video_id: str, settings=None) -> Path:
    """Deterministic on-disk path for a video's persisted audio file."""
    settings = settings or get_settings()
    return Path(settings.audio_cache_directory) / f"{video_id}.webm"


def _sweep_stale_audio(directory: Path) -> None:
    """Best-effort removal of audio files older than the stale TTL."""
    now = time.time()
    try:
        for entry in directory.iterdir():
            try:
                if entry.is_file() and now - entry.stat().st_mtime > _AUDIO_STALE_TTL_SECONDS:
                    entry.unlink(missing_ok=True)
            except OSError:
                continue
    except OSError:
        return


def download_audio(youtube_url: str, video_id: str = "", settings=None) -> str:
    """Download a YouTube audio stream to a file and return its path.

    With a ``video_id`` the audio is stored at a deterministic cache path and
    reused across attempts and jobs for the same video: a caller that finds the
    file skips the network download entirely. The download itself is written to
    a temp file and atomically renamed into place, so concurrent callers never
    observe a partial file, and stale files are swept on each call. Without a
    ``video_id`` the audio is a per-attempt temp file, as before.
    """
    cached = audio_cache_path(video_id, settings) if video_id else None
    if cached is not None:
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            _sweep_stale_audio(cached.parent)
            if cached.is_file() and cached.stat().st_size > 0:
                logger.info(
                    "Audio cache hit",
                    video_id=video_id,
                    path=str(cached),
                    size_mib=round(cached.stat().st_size / (1024 * 1024), 1),
                )
                return str(cached)
        except OSError as exc:
            logger.warning(
                "Audio cache unavailable; using a temp file",
                video_id=video_id,
                error=str(exc),
            )
            cached = None

    fd, path = tempfile.mkstemp(suffix=".webm")
    os.close(fd)
    destination = Path(path)
    destination.unlink(missing_ok=True)
    options = build_ydlp_options(
        settings=settings,
        format="bestaudio/best",
        outtmpl=path,
    )
    try:
        run_download(options, youtube_url)
        if not destination.exists() or destination.stat().st_size == 0:
            logger.error("Audio download produced no file", path=path)
            raise ExternalServiceError("Audio download produced no file", service="yt-dlp")
        if cached is not None:
            Path(path).replace(cached)
            return str(cached)
        return path
    except ExternalServiceError:
        remove_file(path)
        raise
    except Exception as exc:
        remove_file(path)
        logger.error("Audio download failed", error=str(exc))
        raise ExternalServiceError("Could not download audio", service="yt-dlp") from exc


def remove_audio_cache(video_id: str, settings=None) -> None:
    """Best-effort removal of a video's persisted audio file (job settled)."""
    if not video_id:
        return
    remove_file(str(audio_cache_path(video_id, settings)))


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


class AudioFileLease:
    """One job's hold on the audio file for a video.

    The owner asks for the file on every transcription attempt
    (:meth:`acquire`) but downloads it at most once: a failed *transcription*
    re-submits the same file, and only a failed *download* is retried. The
    shared file is handed back when the job settles (:meth:`release`); an
    unkeyed lease additionally owns a per-attempt temp file, which
    :meth:`aclose` deletes — so it belongs in the ``finally`` of the attempt
    loop.

    A keyed lease (a live job with a keyable ``video_id``) reads and releases
    the shared per-video file, so a failed job's download survives into the next
    job for the same video. A resume never downloads: it hands the leaf the
    shared path so the leaf does not fetch one of its own, and a job that ends
    pending leaves the file in place for whatever resumes it.
    """

    def __init__(self, video_id: str = "", settings=None, *, keyed: bool = False) -> None:
        self._video_id = video_id
        self._settings = settings
        self._keyed = keyed
        self._path = ""
        self._owned_temp = ""

    async def acquire(self, youtube_url: str, *, resume: bool = False) -> str:
        """The path to hand the audio leaf, downloading it on first use.

        Returns ``""`` for a resume with no keyable video id, which tells the
        leaf to fetch (and clean up) its own temp audio.
        """
        if resume:
            # A resume polls an already-submitted job, so it never downloads;
            # it still hands over the shared path so the leaf treats the file as
            # caller-owned instead of fetching its own.
            if self._keyed:
                self._path = str(audio_cache_path(self._video_id, self._settings))
            return self._path
        if not self._path:
            self._path = await asyncio.to_thread(
                download_audio,
                youtube_url,
                video_id=self._video_id if self._keyed else "",
                settings=self._settings,
            )
            if not self._keyed:
                self._owned_temp = self._path
        return self._path

    async def release(self) -> None:
        """Release the shared per-video audio file: the job has settled."""
        if not self._keyed:
            return
        await asyncio.to_thread(remove_audio_cache, self._video_id, self._settings)

    async def aclose(self) -> None:
        """Delete the temp file this lease owns, if it downloaded one."""
        if self._owned_temp:
            await asyncio.to_thread(remove_file, self._owned_temp)
