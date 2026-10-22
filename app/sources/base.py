"""Video sources the worker can acquire media from.

Mirrors the backend's host-based source registry: each source owns URL
identification and media acquisition for its own host, so the download and
audio paths are chosen once from the URL instead of branching on hosts at
every call site. The registry in :mod:`app.sources.registry` is the single
entry point callers use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings


@dataclass(frozen=True)
class VideoRef:
    """A video identified from a URL: its source, id, and canonical URL."""

    source: str
    source_id: str
    canonical_url: str


@dataclass(frozen=True)
class PreparedAudio:
    """Caller-owned transcription audio plus the metadata seen while fetching it.

    Sources without a per-track endpoint must fetch the whole media file to
    reach its audio, so the title and duration are read from that same file at
    no extra network cost. ``directory`` is the workspace the caller owns and
    deletes once the transcription settles.
    """

    title: str
    duration_seconds: int
    audio_path: Path
    directory: Path


class VideoSource(ABC):
    """One video host the worker knows how to acquire media from."""

    name: str = ""
    native_captions: bool = False

    @abstractmethod
    def matches(self, url: str) -> bool:
        """Whether this source owns the URL's host."""

    @abstractmethod
    def identify(self, url: str) -> VideoRef:
        """Identify the video a URL points at; raises on an unusable link."""

    @abstractmethod
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
        """Download the media (one section of it when asked) into ``directory``.

        ``height`` caps the resolution only where a quality ladder exists;
        sources that serve a single original file ignore it.
        """


class CaptionlessSource(VideoSource):
    """A source with no caption tracks: its transcript comes from audio only.

    Captionless sources also have no yt-dlp metadata to read, so they must
    produce the transcription audio themselves, together with whatever title
    and duration they can observe while acquiring it.
    """

    @abstractmethod
    def prepare_audio(
        self, url: str, directory: Path, settings: Settings | None = None
    ) -> PreparedAudio:
        """Acquire the transcription audio file inside ``directory``."""
