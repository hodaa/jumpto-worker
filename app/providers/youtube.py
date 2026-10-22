"""YouTube transcript strategy: captions first, audio transcription second.

Runs entirely on the worker. yt-dlp is only a tool this strategy uses
internally: the caption fast path reads caption tracks, and when no captions
exist the audio stream is downloaded once and handed to the configured
audio-transcription leaf (Deepgram by default, or Assembly). It is the
terminal strategy for YouTube — it either produces a transcript or raises, so
a job that reaches it with no result is failed, not completed empty.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from app.core.config import Settings, _live_pipeline_enabled
from app.core.exceptions import ExternalServiceError, PermanentExternalServiceError
from app.core.logging import get_logger
from app.providers.audio import AudioFileLease
from app.providers.audio_jobs import AudioJob, run_transcription_attempts
from app.providers.base import VideoTranscriptResult
from app.providers.local import LocalTranscriptProvider, build_video_result
from app.providers.media import get_media_info_with_raw
from app.providers.models import TranscriptData
from app.providers.registry import (
    TranscriptProviderSpec,
    provider_spec,
    register_alias,
    register_provider,
)
from app.providers.transcript import YouTubeCaptionTranscriptService
from app.sources import resolve_source
from app.speech_to_text.base import SpeechToTextProvider
from app.storage.cache import TranscriptCache, extract_youtube_video_id
from app.utils.text import detect_language_from_title

logger = get_logger(__name__)

_ORDER = 5


def register_self() -> None:
    """Register this strategy's spec and its legacy alias. Idempotent."""
    if provider_spec(YouTubeTranscriptProvider.name) is None:
        register_provider(
            TranscriptProviderSpec(
                name=YouTubeTranscriptProvider.name,
                build=lambda settings: YouTubeTranscriptProvider(settings=settings),
                order=_ORDER,
                description=("Local YouTube pipeline: captions with audio transcription fallback."),
                uses_cloud=False,
                cache_config_fields=("speech_to_text_provider",),
            )
        )
    # ``yt-dlp`` named this strategy before Drive forced the split; armed
    # webhooks written before the rename still use it.
    register_alias("yt-dlp", YouTubeTranscriptProvider.name)


class YouTubeTranscriptProvider(LocalTranscriptProvider):
    """Downloads the transcript directly (yt-dlp captions, then audio transcription)."""

    name = "youtube"
    supports_resume = True
    supports_webhook = True

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        captions_service: YouTubeCaptionTranscriptService | None = None,
        speech_to_text_provider: Callable[..., SpeechToTextProvider | None] | None = None,
        cache: TranscriptCache | None = None,
    ) -> None:
        """Build the strategy, optionally wiring collaborators explicitly.

        ``captions_service`` abstracts the caption fast path; when omitted the
        concrete service is used, built from the injected settings.
        """
        super().__init__(
            settings=settings,
            speech_to_text_provider=speech_to_text_provider,
            cache=cache,
        )
        self._captions_service = captions_service

    def _resolve_captions_service(self) -> YouTubeCaptionTranscriptService:
        # The injected settings are threaded in so the caption download builds
        # its yt-dlp options (cookie file, socket timeout) from the caller's
        # config instead of re-reading the process-wide singleton.
        return self._captions_service or YouTubeCaptionTranscriptService(
            settings=self._resolve_settings()
        )

    async def fetch(
        self,
        youtube_url: str,
        youtube_video_id: str = "",
        resume_token: str = "",
        *,
        webhook_url: str = "",
    ) -> VideoTranscriptResult:
        """Fetch media metadata and the best available transcript.

        The caption fast path is tried first; captionless or failing videos
        fall back to audio transcription. ``resume_token`` resumes a previously
        queued Assembly job so a retry re-polls it instead of re-downloading
        the audio. ``webhook_url`` is a public callback URL for the pending
        Assembly job (built by the pipeline); it arms a completion webhook
        instead of in-worker polling.

        When the live pipeline is on, finished transcripts are cached on disk
        keyed by the video id, so a later job for the same video skips yt-dlp
        entirely. The id comes from the job when present or is parsed from the
        URL otherwise; unparseable ids simply disable caching.

        A resume of a pending Assembly job reads the title/duration from the
        cache (written when the job was first submitted) instead of re-running
        the yt-dlp metadata extract — the only work left is a single Assembly
        poll. A cache miss falls back to a fresh metadata fetch.

        The video title is used to derive a language tag (Arabic-script titles
        transcribe as Arabic, else English) that is forwarded to the audio
        leaf, which Deepgram consumes for the transcription request.
        """
        if not resolve_source(youtube_url).native_captions:
            raise PermanentExternalServiceError(
                "The youtube strategy cannot transcribe a captionless source",
                service=self.name,
            )

        video_id = youtube_video_id or extract_youtube_video_id(youtube_url)
        cached = await self._read_cache(video_id)
        if cached is not None:
            return cached

        media = None
        if resume_token:
            media = await self._read_media(video_id)

        if media is not None:
            info = None  # resume skips the caption fast path entirely
        else:
            media, info = await asyncio.to_thread(
                get_media_info_with_raw,
                youtube_video_id,
                youtube_url,
                settings=self._resolve_settings(),
            )

        language = detect_language_from_title(media.title)
        transcript = await self._fetch_transcript_with_retry(
            youtube_url,
            info,
            resume_token=resume_token,
            webhook_url=webhook_url,
            language=language,
            video_id=video_id,
        )
        result = build_video_result(media, transcript)

        await self._write_cache(video_id, result)
        await self._write_media(video_id, media)
        return result

    async def _fetch_transcript_with_retry(
        self,
        youtube_url: str,
        info: dict | None = None,
        resume_token: str = "",
        webhook_url: str = "",
        language: str = "",
        video_id: str = "",
    ) -> TranscriptData:
        """Fetch a transcript, prioritizing captions and falling back to audio.

        A ``resume_token`` skips the caption fast path and directly resumes the
        pending Assembly job. Otherwise captions are tried first; when they are
        missing or fail, the remaining parameters are handed to the audio path
        as an :class:`~app.providers.audio_jobs.AudioJob`. ``None`` from the
        audio provider means audio transcription is not configured, so
        captionless videos fail cleanly instead of fabricating data.
        """
        job = AudioJob(
            youtube_url,
            video_id=video_id,
            resume_token=resume_token,
            webhook_url=webhook_url,
            language=language,
        )
        if job.resume_token:
            return await self._fetch_audio_with_retry(job)

        captions = await self._try_captions(youtube_url, info)
        if captions is not None:
            return captions
        return await self._fetch_audio_with_retry(job)

    async def _try_captions(
        self, youtube_url: str, info: dict | None = None
    ) -> TranscriptData | None:
        """Return the caption fast path result, or ``None`` on a miss.

        Captions are skipped entirely when the live pipeline is off. Missing
        caption tracks are a normal outcome (the caller falls back to audio);
        genuine caption failures are logged and also fall through.
        """
        if not _live_pipeline_enabled(self._resolve_settings()):
            return None
        try:
            captions = await self._resolve_captions_service().fetch(youtube_url, info=info)
        except ExternalServiceError:
            logger.warning(
                "Captions fast-path failed; falling back to audio transcription",
                youtube_url=youtube_url,
                exc_info=True,
            )
            return None
        if captions is None:
            logger.info(
                "No captions available; falling back to audio transcription",
                youtube_url=youtube_url,
            )
            return None
        return captions

    def _require_speech_to_text_leaf(self, resume_token: str) -> SpeechToTextProvider:
        """The configured audio leaf, or a clean failure when none is configured.

        Fails the job rather than fabricating data: a captionless video with no
        audio transcription available has no transcript to produce.
        """
        provider = self._speech_to_text_provider(self._resolve_settings())
        if provider is not None:
            return provider
        message = (
            "Audio transcription is not configured; cannot resume the job"
            if resume_token
            else "No usable captions and audio transcription is not configured"
        )
        raise ExternalServiceError(message, service=self.name)

    async def _fetch_audio_with_retry(self, job: AudioJob) -> TranscriptData:
        """Transcribe audio via the configured leaf, retrying transient failures.

        Resolves the leaf, holds the job's audio file for the duration (see
        :class:`~app.providers.audio.AudioFileLease`), and runs the retry
        budget: the audio is downloaded once and reused across attempts, and
        only a failed download is retried. A pending Assembly job bubbles up
        re-routed to this strategy's name so the task resumes it (or ends
        cleanly when a completion webhook was armed).
        """
        provider = self._require_speech_to_text_leaf(job.resume_token)
        settings = self._resolve_settings()
        lease = AudioFileLease(
            job.video_id,
            settings,
            keyed=bool(job.video_id) and _live_pipeline_enabled(settings),
        )
        try:
            return await run_transcription_attempts(provider, job, lease, strategy=self.name)
        finally:
            await lease.aclose()


register_self()
