"""External service providers for the transcription pipeline."""

from app.providers.base import TranscriptProviderStrategy, VideoTranscriptResult
from app.providers.drive import DriveTranscriptProvider
from app.providers.media import MediaInfo, get_media_info, get_media_info_with_raw
from app.providers.models import (
    TranscriptData,
    TranscriptJobPending,
    TranscriptWordData,
)
from app.providers.registry import (
    TranscriptProviderSpec,
    build_provider,
    ordered_specs,
    provider_spec,
    register_provider,
)
from app.providers.transcript import YouTubeCaptionTranscriptService
from app.providers.youtube import YouTubeTranscriptProvider

__all__ = [
    "DriveTranscriptProvider",
    "MediaInfo",
    "TranscriptData",
    "TranscriptJobPending",
    "TranscriptProviderSpec",
    "TranscriptProviderStrategy",
    "TranscriptWordData",
    "VideoTranscriptResult",
    "YouTubeCaptionTranscriptService",
    "YouTubeTranscriptProvider",
    "build_provider",
    "get_media_info",
    "get_media_info_with_raw",
    "ordered_specs",
    "provider_spec",
    "register_provider",
]
