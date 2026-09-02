"""Media metadata providers (yt-dlp live / deterministic fake)."""

from dataclasses import dataclass

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.providers.ytdlp import build_ydlp_options

logger = get_logger(__name__)

_FAKE_DURATION_SECONDS = 300
_TITLE_PREFIX = "JumpTo test video"


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
    if get_settings().jumpto_live_external_calls:
        try:
            return _fetch_from_yt_dlp(youtube_url)
        except ExternalServiceError:
            raise
        except Exception as exc:
            logger.error("yt-dlp media fetch failed", video_id=video_id, error=str(exc))
            raise ExternalServiceError(
                "Could not fetch video metadata",
                service="yt-dlp",
            ) from exc
    return _fake_media_info(video_id)


def _fake_media_info(video_id: str) -> MediaInfo:
    """Build deterministic media info without external calls."""
    return MediaInfo(title=f"{_TITLE_PREFIX} {video_id}", duration_seconds=_FAKE_DURATION_SECONDS)


def _fetch_from_yt_dlp(youtube_url: str) -> MediaInfo:
    """Fetch live metadata with yt-dlp (no download)."""
    import yt_dlp  # Optional dependency, only needed for live calls

    options = build_ydlp_options()
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
        title = str(info.get("title") or "Untitled video")
        duration = int(info.get("duration") or 0)
        return MediaInfo(title=title, duration_seconds=duration)
    except yt_dlp.utils.DownloadError as exc:
        logger.exception(
            "yt-dlp failed to fetch video metadata",
            youtube_url=youtube_url,
            error=str(exc),
        )
        raise ExternalServiceError(
            "Could not fetch video metadata",
            service="yt-dlp",
        ) from exc
