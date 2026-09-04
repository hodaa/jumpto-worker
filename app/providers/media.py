"""Media metadata providers (yt-dlp live / deterministic fake)."""

import time
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.providers.ytdlp import build_ydlp_options

logger = get_logger(__name__)

_FAKE_DURATION_SECONDS = 300
_TITLE_PREFIX = "JumpTo test video"

_METADATA_RETRY_ATTEMPTS = 3
_METADATA_RETRY_DELAY_SECONDS = 2


@dataclass
class MediaInfo:
    """Title and duration for a YouTube video."""

    title: str
    duration_seconds: int


def get_media_info(video_id: str, youtube_url: str) -> MediaInfo:
    """
    Return media info, using live yt-dlp only when explicitly enabled.

    Args:
        video_id: YouTube video ID
        youtube_url: Original YouTube URL

    Returns:
        MediaInfo with title and duration
    """
    media, _ = get_media_info_with_raw(video_id, youtube_url)
    return media


def get_media_info_with_raw(video_id: str, youtube_url: str) -> tuple[MediaInfo, dict | None]:
    """
    Return media info plus the raw yt-dlp metadata dict, reusing a single
    ``extract_info`` call so downstream providers don't re-fetch it.

    Args:
        video_id: YouTube video ID
        youtube_url: Original YouTube URL

    Returns:
        A tuple of (MediaInfo, raw yt-dlp info dict). The dict is ``None`` when
        live calls are disabled.
    """
    if get_settings().jumpto_live_external_calls:
        try:
            info = _fetch_from_yt_dlp(youtube_url)
            media = MediaInfo(
                title=str(info.get("title") or "Untitled video"),
                duration_seconds=int(info.get("duration") or 0),
            )
            return media, info
        except ExternalServiceError:
            raise
        except Exception as exc:
            logger.error("yt-dlp media fetch failed", video_id=video_id, error=str(exc))
            raise ExternalServiceError(
                "Could not fetch video metadata",
                service="yt-dlp",
            ) from exc
    return _fake_media_info(video_id), None


def _fake_media_info(video_id: str) -> MediaInfo:
    """Build deterministic media info without external calls."""
    return MediaInfo(title=f"{_TITLE_PREFIX} {video_id}", duration_seconds=_FAKE_DURATION_SECONDS)


def _fetch_from_yt_dlp(youtube_url: str) -> dict:
    """Fetch live metadata with yt-dlp (no download) and return the raw info dict.

    Transient ``DownloadError``s (rate limits, network blips, YouTube hiccups)
    are retried with a short backoff so a single failure doesn't kill the job.
    """
    import yt_dlp  # Optional dependency, only needed for live calls

    options = build_ydlp_options()
    last_error: yt_dlp.utils.DownloadError | None = None
    for attempt in range(_METADATA_RETRY_ATTEMPTS):
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                return ydl.extract_info(youtube_url, download=False)
        except yt_dlp.utils.DownloadError as exc:
            last_error = exc
            logger.warning(
                "yt-dlp metadata fetch attempt failed; retrying",
                youtube_url=youtube_url,
                attempt=attempt + 1,
                error=str(exc),
            )
            if attempt + 1 < _METADATA_RETRY_ATTEMPTS:
                time.sleep(_METADATA_RETRY_DELAY_SECONDS * (attempt + 1))
    assert last_error is not None
    logger.exception(
        "yt-dlp failed to fetch video metadata after retries",
        youtube_url=youtube_url,
        error=str(last_error),
    )
    raise ExternalServiceError(
        "Could not fetch video metadata",
        service="yt-dlp",
    ) from last_error
