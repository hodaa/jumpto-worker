"""External service providers for the transcription pipeline."""

from app.providers.assembly import (
    AssemblyTranscriptService,
    get_transcript_provider,
)
from app.providers.base import TranscriptProviderStrategy, TranscriptService, VideoTranscriptResult
from app.providers.deepgram import DeepgramTranscriptService, get_deepgram_provider
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
from app.providers.speech_to_text import get_speech_to_text_provider
from app.providers.supadata import SupadataResult, SupadataTranscriptProvider
from app.providers.transcript import YouTubeCaptionTranscriptService
from app.providers.transcriptfetch import (
    TranscriptFetchPermanentError,
    TranscriptFetchResult,
    TranscriptFetchTranscriptProvider,
)
from app.providers.vidwords import VidWordsResult, VidWordsTranscriptProvider
from app.providers.ytdlp import YtDlpTranscriptProvider

__all__ = [
    "AssemblyTranscriptService",
    "DeepgramTranscriptService",
    "MediaInfo",
    "SupadataResult",
    "SupadataTranscriptProvider",
    "TranscriptData",
    "TranscriptFetchPermanentError",
    "TranscriptFetchResult",
    "TranscriptFetchTranscriptProvider",
    "TranscriptJobPending",
    "TranscriptService",
    "TranscriptProviderSpec",
    "TranscriptProviderStrategy",
    "TranscriptWordData",
    "VideoTranscriptResult",
    "VidWordsResult",
    "VidWordsTranscriptProvider",
    "YtDlpTranscriptProvider",
    "YouTubeCaptionTranscriptService",
    "build_provider",
    "get_deepgram_provider",
    "get_media_info",
    "get_media_info_with_raw",
    "get_speech_to_text_provider",
    "get_transcript_provider",
    "ordered_specs",
    "provider_spec",
    "register_provider",
]
