"""Shared strategy interface and result type for video transcript providers.

The provider chain is built with the strategy pattern: every transcript source
(cloud APIs, the local YouTube pipeline, captionless sources) implements the
same :class:`TranscriptProviderStrategy` interface, so strategies are
interchangeable and the chain order is decided by the registry/factory in
:mod:`app.providers.registry`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.providers.models import TranscriptData


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
    routing and submission provenance. ``supports_resume`` marks strategies
    that can resume a previously queued background job (e.g. Assembly audio
    transcription) via ``resume_token``; the
    pipeline only routes resumes to such strategies. Deployment metadata that
    is not part of the transcript contract (e.g. whether the strategy needs
    live external API calls) lives on the registry spec, not here.

    ``fetch`` returns a :class:`VideoTranscriptResult`, ``None`` for a soft
    miss (so the caller tries the next strategy), or raises
    :class:`~app.core.exceptions.ExternalServiceError` (or a subclass). A
    terminal strategy (youtube, drive) never returns ``None`` — it either
    produces a result or raises — so a soft-miss return is not part of its
    contract.
    """

    name: str = ""
    supports_resume: bool = False
    supports_webhook: bool = False

    @abstractmethod
    async def fetch(
        self,
        youtube_url: str,
        youtube_video_id: str = "",
        resume_token: str = "",
        *,
        webhook_url: str = "",
    ) -> VideoTranscriptResult | None:
        """Fetch transcript data, returning ``None`` on a soft miss.

        ``webhook_url`` is an opaque public callback URL the pipeline built for
        this job; only asynchronous strategies that support completion
        callbacks (Assembly) use it. Others must ignore it.
        """
