"""VidWords transcript provider (cloud fallback).

VidWords proxies YouTube on their own infrastructure, so transcripts and
video title arrive without yt-dlp, cookies, or PO-token providers — useful for
videos whose local yt-dlp extraction is blocked by datacenter-IP bot-checks.

The API returns caption ``segments`` (cue-level ``start``/``duration``), not
per-word timestamps; word records are derived from each cue so the pipeline's
word-level search keeps working at caption-line granularity.
"""

import math
import re
from dataclasses import dataclass

import httpx

from app.client.http import get_shared_http_client
from app.core.exceptions import ExternalServiceError, PermanentExternalServiceError
from app.core.logging import get_logger
from app.providers.base import TranscriptProviderStrategy, VideoTranscriptResult
from app.providers.models import (
    TranscriptData,
    TranscriptWordData,
    _close_word_times,
    _language_base,
)
from app.providers.registry import TranscriptProviderSpec, register_provider

logger = get_logger(__name__)

_TIMEOUT_SECONDS = 60.0

# Per-video errors that mean "this video/account can never yield a transcript"
# (so moving on to yt-dlp would just waste time).
_PERMANENT_ERRORS = {
    "invalid_id",
    "video_unavailable",
    "age_restricted",
    "tier_limit",
    "insufficient_credits",
}

# Per-video errors meaning "no captions exist": a soft miss, caller should
# fall back to the audio-transcription path.
_NO_TRANSCRIPT_ERRORS = {"no_transcript", "transcripts_disabled"}


class VidWordsPermanentError(PermanentExternalServiceError):
    """VidWords failure that no fallback can recover (fast-fail)."""


@dataclass(frozen=True)
class VidWordsResult(VideoTranscriptResult):
    """Transcript plus the metadata VidWords returns for a video."""


class VidWordsTranscriptProvider(TranscriptProviderStrategy):
    """Fetches transcripts and basic metadata from the VidWords API."""

    name = "vidwords"
    supports_resume = False

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://vidwords.com",
        lang: str = "en",
        timeout: float = _TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.lang = lang
        self.timeout = timeout
        self.transport = transport

    async def fetch(
        self,
        youtube_url: str,
        youtube_video_id: str = "",
        resume_token: str = "",
        *,
        webhook_url: str = "",
    ) -> VidWordsResult | None:
        """Fetch a transcript for ``youtube_url``.

        Returns ``None`` when the video has no caption track (caller should
        fall back). Raises :class:`ExternalServiceError` on API/account
        failures or permanent per-video errors.
        """
        if resume_token:
            logger.warning(
                "VidWords is synchronous and does not use resume tokens; ignoring",
                youtube_url=youtube_url,
            )
        payload = {"ids": [youtube_url], "lang": self.lang}
        headers = {
            "Authorization": f"Basic {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            client = get_shared_http_client(timeout=self.timeout, transport=self.transport)
            response = await client.post(
                f"{self.base_url}/api/transcripts", json=payload, headers=headers
            )
        except httpx.HTTPError as exc:
            logger.error("VidWords request failed", youtube_url=youtube_url, error=str(exc))
            raise ExternalServiceError(
                "Could not fetch the transcript from the transcription service",
                service="vidwords",
            ) from exc

        if response.status_code != 200:
            self._raise_http_error(response)

        try:
            results = response.json().get("results") or []
        except ValueError:
            raise ExternalServiceError(
                "Unexpected response from transcription service", service="vidwords"
            ) from None
        if not results:
            raise ExternalServiceError(
                "Transcription service returned no results", service="vidwords"
            )

        item = results[0]
        error = item.get("error")
        if error:
            if error in _NO_TRANSCRIPT_ERRORS:
                logger.info("VidWords found no captions", youtube_url=youtube_url, error=error)
                return None
            if error in _PERMANENT_ERRORS:
                raise VidWordsPermanentError(
                    "Could not fetch the transcript for this video",
                    service="vidwords",
                    details={"vidwords_error": error, "message": item.get("message")},
                )
            logger.warning(
                "VidWords transient per-video error",
                youtube_url=youtube_url,
                error=error,
                message=item.get("message"),
            )
            raise ExternalServiceError(
                "Could not fetch the transcript for this video",
                service="vidwords",
                details={"vidwords_error": error, "message": item.get("message")},
            )

        return self._build_result(item)

    def _raise_http_error(self, response: httpx.Response) -> None:
        """Map an HTTP error status to an ExternalServiceError."""
        try:
            body = response.json()
            code = body.get("error", "")
            message = body.get("message", "")
        except ValueError:
            code = ""
            message = response.text[:200]
        detail = {"status_code": response.status_code, "vidwords_error": code, "message": message}
        if response.status_code in (400, 401, 402, 403):
            raise VidWordsPermanentError(
                "Transcription service rejected the request (check the VidWords API key)",
                service="vidwords",
                details=detail,
            ) from None
        if response.status_code == 429 or response.status_code >= 500:
            raise ExternalServiceError(
                "Transcription service is rate-limited or unavailable; try again later",
                service="vidwords",
                details=detail,
            ) from None
        raise ExternalServiceError(
            "Transcription service rejected the request",
            service="vidwords",
            details=detail,
        ) from None

    def _build_result(self, item: dict) -> VidWordsResult:
        """Convert a successful VidWords item into a VidWordsResult."""
        segments = list(item.get("segments") or [])
        words = _words_from_segments(segments)
        if words:
            _close_word_times(words)
        text = str(item.get("text") or "")
        if not text:
            text = " ".join(word.word for word in words)
        duration = _duration_from_segments(segments)
        transcript = TranscriptData(
            language=_language_base(str(item.get("languageCode") or "en")),
            text=text,
            words=words,
        )
        return VidWordsResult(
            title=str(item.get("title") or ""),
            author=str(item.get("author") or ""),
            duration_seconds=duration,
            is_generated=bool(item.get("isGenerated")),
            transcript=transcript,
        )


def _words_from_segments(segments: list[dict]) -> list[TranscriptWordData]:
    """Expand caption cues into word records that share the cue's timing."""
    words: list[TranscriptWordData] = []
    for segment in segments:
        start = float(segment.get("start") or 0)
        duration = float(segment.get("duration") or 0)
        text = re.sub(r"[♪♪]", "", str(segment.get("text") or "")).strip()
        for word in text.split():
            words.append(TranscriptWordData(word=word, start_time=start, end_time=start + duration))
    return words


def _duration_from_segments(segments: list[dict]) -> int:
    """Derive the video duration from the last caption cue end."""
    if not segments:
        return 0
    last_end = max(
        (float(segment.get("start") or 0) + float(segment.get("duration") or 0))
        for segment in segments
    )
    return math.ceil(last_end)


def _build(settings) -> VidWordsTranscriptProvider | None:
    """Build a VidWords strategy from settings, or ``None`` when unconfigured."""
    api_key = getattr(settings, "vidwords_api_key", "")
    if not api_key:
        return None
    return VidWordsTranscriptProvider(
        api_key=api_key,
        base_url=getattr(settings, "vidwords_api_url", "https://vidwords.com"),
        lang=getattr(settings, "vidwords_lang", "en") or "en",
    )


register_provider(
    TranscriptProviderSpec(
        name="vidwords",
        build=_build,
        order=30,
        description="VidWords YouTube transcripts API.",
    )
)
