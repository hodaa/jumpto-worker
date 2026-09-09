"""Shared strategy interface and result type for video transcript providers.

The provider chain is built with the strategy pattern: every transcript source
(cloud APIs, local yt-dlp/Assembly) implements the same
:class:`TranscriptProviderStrategy` interface, so strategies are
interchangeable and the chain order is decided by the registry/factory in
:mod:`app.providers.registry`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.providers.transcript import TranscriptData


@dataclass(frozen=True)
class VideoTranscriptResult:
    """Metadata plus transcript produced by a provider strategy."""

    title: str
    author: str
    duration_seconds: int
    is_generated: bool
    transcript: TranscriptData


class TranscriptProviderStrategy(ABC):
    """A single interchangeable video-transcript strategy.

    ``name`` is the stable provider id used for registry lookup, resume-token
    routing and submission provenance. ``uses_cloud`` marks strategies that
    need live external calls (API keys, network access); the local yt-dlp
    strategy sets it to ``False`` so it stays available as the free first
    strategy even when live calls are disabled.

    ``fetch`` returns a :class:`VideoTranscriptResult`, ``None`` for a soft
    miss (so the caller tries the next strategy), or raises
    :class:`~app.core.exceptions.ExternalServiceError` (or a subclass).
    """

    name: str = ""
    uses_cloud: bool = True

    @abstractmethod
    async def fetch(
        self,
        youtube_url: str,
        youtube_video_id: str = "",
        resume_token: str = "",
    ) -> VideoTranscriptResult | None:
        """Fetch transcript data, returning ``None`` on a soft miss."""
