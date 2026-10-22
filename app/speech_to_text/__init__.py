"""Speech-to-text leaf services and the selector for the configured one."""

from app.speech_to_text.assembly import AssemblyTranscriptService, get_transcript_provider
from app.speech_to_text.base import SpeechToTextProvider
from app.speech_to_text.deepgram import DeepgramTranscriptService, get_deepgram_provider
from app.speech_to_text.registry import get_speech_to_text_provider

__all__ = [
    "AssemblyTranscriptService",
    "DeepgramTranscriptService",
    "SpeechToTextProvider",
    "get_deepgram_provider",
    "get_speech_to_text_provider",
    "get_transcript_provider",
]
