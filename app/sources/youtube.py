"""YouTube as a worker video source.

URL identification stays late-bound to the shared video-id parser so the
accepted link shapes keep living in one place, and downloads delegate to the
existing yt-dlp pipeline, which this source wraps without changing its
behavior.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from app.core.config import Settings
from app.core.exceptions import ExternalServiceError
from app.sources.base import VideoRef, VideoSource
from app.utils.video import extract_youtube_video_id

YOUTUBE_HOSTS = frozenset({"youtube.com", "youtu.be"})


def is_youtube_host(url: str) -> bool:
    """Whether the URL points at a YouTube host, whatever its shape."""
    host = urlsplit(url).netloc.lower()
    return host in YOUTUBE_HOSTS or host.endswith(".youtube.com")


class YouTubeSource(VideoSource):
    """YouTube videos: captions first, yt-dlp audio as the fallback."""

    name = "youtube"
    native_captions = True

    def matches(self, url: str) -> bool:
        """Whether the URL is on a YouTube host."""
        return is_youtube_host(url)

    def identify(self, url: str) -> VideoRef:
        """Parse a YouTube video id out of the URL, or raise when there is none."""
        source_id = extract_youtube_video_id(url)
        if not source_id:
            raise ExternalServiceError("Could not parse a YouTube video id", service=self.name)
        return VideoRef(source=self.name, source_id=source_id, canonical_url=url)

    def download(
        self,
        url: str,
        directory: Path,
        *,
        start_seconds: int | None = None,
        end_seconds: int | None = None,
        height: int | None = None,
        settings: Settings | None = None,
    ) -> Path:
        """Download through the existing yt-dlp pipeline (behavior unchanged)."""
        from app.services.downloads import fetch_video_file_with_ytdlp

        return fetch_video_file_with_ytdlp(
            url,
            directory,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            height=height,
            settings=settings,
        )
