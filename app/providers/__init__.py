"""External service providers for the transcription pipeline."""

from app.providers.assembly import AssemblyTranscriptProvider
from app.providers.base import TranscriptProviderStrategy, VideoTranscriptResult
from app.providers.local import YtDlpTranscriptStrategy
from app.providers.media import MediaInfo, get_media_info, get_media_info_with_raw
from app.providers.registry import (
    TranscriptProviderSpec,
    build_provider,
    ordered_specs,
    provider_spec,
    register_provider,
)
from app.providers.supadata import SupadataResult, SupadataTranscriptProvider
from app.providers.transcript import (
    FakeTranscriptProvider,
    TranscriptData,
    TranscriptJobPending,
    TranscriptProvider,
    TranscriptWordData,
    YouTubeCaptionTranscriptProvider,
    get_transcript_provider,
)
from app.providers.transcriptfetch import (
    TranscriptFetchPermanentError,
    TranscriptFetchResult,
    TranscriptFetchTranscriptProvider,
)
from app.providers.vidwords import VidWordsResult, VidWordsTranscriptProvider

__all__ = [
    "AssemblyTranscriptProvider",
    "FakeTranscriptProvider",
    "MediaInfo",
    "SupadataResult",
    "SupadataTranscriptProvider",
    "TranscriptData",
    "TranscriptFetchPermanentError",
    "TranscriptFetchResult",
    "TranscriptFetchTranscriptProvider",
    "TranscriptJobPending",
    "TranscriptProvider",
    "TranscriptProviderSpec",
    "TranscriptProviderStrategy",
    "TranscriptWordData",
    "VideoTranscriptResult",
    "VidWordsResult",
    "VidWordsTranscriptProvider",
    "YtDlpTranscriptStrategy",
    "YouTubeCaptionTranscriptProvider",
    "build_provider",
    "get_media_info",
    "get_media_info_with_raw",
    "get_transcript_provider",
    "ordered_specs",
    "provider_spec",
    "register_provider",
]
