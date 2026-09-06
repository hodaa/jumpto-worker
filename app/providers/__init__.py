"""External service providers for the transcription pipeline."""

from app.providers.media import MediaInfo, get_media_info, get_media_info_with_raw
from app.providers.supadata import SupadataResult, SupadataTranscriptProvider
from app.providers.transcript import (
    AssemblyTranscriptProvider,
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
    "TranscriptWordData",
    "VidWordsResult",
    "VidWordsTranscriptProvider",
    "YouTubeCaptionTranscriptProvider",
    "get_media_info",
    "get_media_info_with_raw",
    "get_transcript_provider",
]
