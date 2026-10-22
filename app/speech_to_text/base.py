"""Abstract leaf-fetcher interface shared by the speech-to-text services.

Every internal leaf (the Assembly and Deepgram audio services, and the caption
downloaders) implements :class:`SpeechToTextProvider`, so registry-level
strategies can run and swap them without knowing which service is configured;
:mod:`app.speech_to_text.registry` maps the ``speech_to_text_provider`` setting
to one of the audio leaves.
"""

from abc import ABC, abstractmethod

from app.providers.models import TranscriptData


class SpeechToTextProvider(ABC):
    """Abstract speech-to-text source (internal leaf fetcher).

    Subclasses provide a single raw transcript for a URL: Assembly or Deepgram
    audio transcription, caption downloads, or deterministic fakes.
    Registry-level strategies (``*TranscriptProvider``) orchestrate one or more
    of these.
    """

    @abstractmethod
    async def fetch(self, youtube_url: str) -> TranscriptData | None:
        """Fetch transcript data for a source URL.

        Returns ``None`` when the source has no usable transcript (a soft
        miss), so the orchestrating strategy can fall back. Subclasses may add
        implementation-specific keyword parameters (a pre-extracted ``info``
        metadata dict, a ``resume_token``); callers must hold the concrete
        service type to use them.
        """
