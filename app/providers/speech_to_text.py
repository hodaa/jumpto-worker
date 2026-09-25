"""Factory for the audio-transcription leaf selected by SPEECH_TO_TEXT_PROVIDER.

The yt-dlp composite falls back to a single audio-transcription service when no
captions exist. Which service that is (Deepgram by default, or Assembly) comes
from the ``speech_to_text_provider`` setting; this module is the one place that
maps the setting to a concrete :class:`TranscriptService`, so adding a new
audio provider requires no changes in the composite.
"""

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.providers.assembly import get_transcript_provider
from app.providers.base import TranscriptService
from app.providers.deepgram import get_deepgram_provider

logger = get_logger(__name__)

_SPEECH_TO_TEXT_PROVIDERS = {
    "assembly": get_transcript_provider,
    "deepgram": get_deepgram_provider,
}


def get_speech_to_text_provider(settings: Settings | None = None) -> TranscriptService | None:
    """
    Return the configured audio-transcription leaf service, or ``None``.

    ``None`` means audio transcription is unavailable (live calls disabled, no
    API key for the selected provider, or an unknown provider name), so the
    composite falls back to the caption fast path and fails captionless jobs
    cleanly. ``settings`` may be injected by callers that already resolved
    config; when ``None`` the global settings are fetched.
    """
    settings = settings or get_settings()
    factory = _SPEECH_TO_TEXT_PROVIDERS.get(settings.speech_to_text_provider)
    if factory is None:
        logger.error(
            "Unknown speech-to-text provider",
            provider=settings.speech_to_text_provider,
            providers=", ".join(sorted(_SPEECH_TO_TEXT_PROVIDERS)),
        )
        return None
    return factory(settings)
