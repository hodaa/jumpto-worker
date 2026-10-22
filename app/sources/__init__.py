"""Video source registry: one adapter per supported host."""

from app.sources.base import CaptionlessSource, PreparedAudio, VideoRef, VideoSource
from app.sources.registry import resolve_source

__all__ = [
    "CaptionlessSource",
    "PreparedAudio",
    "VideoRef",
    "VideoSource",
    "resolve_source",
]
