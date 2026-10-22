"""Drive transcript strategy: fetched media, then audio transcription.

Google Drive exposes no captions API, so the strategy fetches the media file
itself (the only way to reach its audio). The download also carries the
metadata — title from the file name, duration from ffprobe — and the extracted
audio is handed to the configured audio-transcription leaf (Deepgram by
default, or Assembly). A resume skips acquisition entirely (title/duration
come from the media cache) and only re-polls the pending cloud job.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import ExternalServiceError, PermanentExternalServiceError
from app.core.logging import get_logger
from app.providers.audio_jobs import AudioJob, run_download_attempts, run_transcription_attempts
from app.providers.base import VideoTranscriptResult
from app.providers.local import LocalTranscriptProvider, build_video_result
from app.providers.media import MediaInfo
from app.providers.registry import TranscriptProviderSpec, provider_spec, register_provider
from app.sources import CaptionlessSource, PreparedAudio, resolve_source
from app.speech_to_text.base import SpeechToTextProvider
from app.storage.cache import TranscriptCache
from app.utils.text import detect_language_from_title

logger = get_logger(__name__)

_ORDER = 6

# Workspace holding a captionless source's fetched media and extracted audio.
_SOURCE_WORKSPACE_PREFIX = "jumpto-source-"


def register_self() -> None:
    """Register this strategy's spec. Idempotent; safe on every import."""
    if provider_spec(DriveTranscriptProvider.name) is not None:
        return
    register_provider(
        TranscriptProviderSpec(
            name=DriveTranscriptProvider.name,
            build=lambda settings: DriveTranscriptProvider(settings=settings),
            order=_ORDER,
            description="Google Drive: fetched media and audio transcription.",
            uses_cloud=True,
            cache_config_fields=(
                "speech_to_text_provider",
                "drive_stt_language",
            ),
        )
    )


class DriveTranscriptProvider(LocalTranscriptProvider):
    """Fetches a Drive video's media and transcribes its audio."""

    name = "drive"
    supports_resume = True
    supports_webhook = True

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        speech_to_text_provider: Callable[..., SpeechToTextProvider | None] | None = None,
        cache: TranscriptCache | None = None,
    ) -> None:
        """Build the strategy; collaborators default to the concrete services."""
        super().__init__(
            settings=settings,
            speech_to_text_provider=speech_to_text_provider,
            cache=cache,
        )

    async def fetch(
        self,
        youtube_url: str,
        youtube_video_id: str = "",
        resume_token: str = "",
        *,
        webhook_url: str = "",
    ) -> VideoTranscriptResult:
        """Transcribe a captionless source's audio.

        The audio leaf is resolved before any download so an unconfigured
        transcription fails fast and fetches nothing.
        """
        source = resolve_source(youtube_url)
        if source.native_captions:
            raise PermanentExternalServiceError(
                "The drive strategy can only transcribe captionless sources",
                service=self.name,
            )

        video_id = youtube_video_id or source.identify(youtube_url).source_id
        cached = await self._read_cache(video_id)
        if cached is not None:
            return cached

        provider = self._speech_to_text_provider(self._resolve_settings())
        if provider is None:
            raise ExternalServiceError(
                "Audio transcription is not configured for this source",
                service=source.name,
            )

        media = await self._read_media(video_id) if resume_token else None
        workspace: PreparedAudio | None = None
        if media is None:
            workspace = await self._acquire_audio(source, youtube_url)
            media = MediaInfo(workspace.title, workspace.duration_seconds)

        language = self._resolve_settings().drive_stt_language or detect_language_from_title(
            media.title
        )
        job = AudioJob(
            youtube_url,
            video_id=video_id,
            resume_token=resume_token,
            webhook_url=webhook_url,
            language=language,
            audio_path=str(workspace.audio_path) if workspace is not None else "",
        )
        try:
            transcript = await run_transcription_attempts(provider, job, strategy=self.name)
        finally:
            if workspace is not None:
                await asyncio.to_thread(shutil.rmtree, workspace.directory, ignore_errors=True)
        result = build_video_result(media, transcript)

        await self._write_cache(video_id, result)
        await self._write_media(video_id, media)
        return result

    async def _acquire_audio(self, source: CaptionlessSource, youtube_url: str) -> PreparedAudio:
        """Fetch a captionless source's media and audio into a fresh workspace.

        The workspace is removed when acquisition never succeeds, so failed
        jobs leak nothing; a successful workspace is owned by the caller and
        deleted once the transcription settles.
        """
        directory = Path(tempfile.mkdtemp(prefix=_SOURCE_WORKSPACE_PREFIX))
        try:
            return await run_download_attempts(
                source, youtube_url, directory, self._resolve_settings()
            )
        except Exception:
            await asyncio.to_thread(shutil.rmtree, directory, ignore_errors=True)
            raise


register_self()
