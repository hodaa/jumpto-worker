"""Registry mapping a job URL onto the video source that can fetch it."""

from __future__ import annotations

from app.core.exceptions import PermanentExternalServiceError
from app.sources.base import VideoSource
from app.sources.drive import DriveSource
from app.sources.youtube import YouTubeSource

_SOURCES: tuple[VideoSource, ...] = (YouTubeSource(), DriveSource())


def resolve_source(url: str) -> VideoSource:
    """Return the source that owns ``url``'s host.

    An empty or unknown URL raises a permanent failure: nothing can download
    a link no source claims, so the job fails with the standard user-safe
    message instead of being retried against providers that cannot help.
    """
    for source in _SOURCES:
        if url and source.matches(url):
            return source
    raise PermanentExternalServiceError("Unsupported video URL", service="sources")
