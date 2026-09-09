"""TranscriptFetch transcript provider (cloud fallback).

TranscriptFetch proxies YouTube (and TikTok/Instagram/podcasts) on their own
infrastructure behind bearer-token auth, so transcripts AND video title arrive
without yt-dlp, cookies, or PO-token providers — useful for videos whose local
yt-dlp extraction is blocked by datacenter-IP bot-checks.

The API runs ``mode=auto``: existing captions are returned immediately (HTTP
200); videos without captions fall back to AI audio transcription. Short jobs
finish inline within the request; longer jobs escalate to an async transcript
and the API returns HTTP 202 with a ``job_id``.

TranscriptFetch runs a strict one-call policy: a single search is one API
call, and an escalated 202 is never polled or retried. It raises
:class:`TranscriptJobPending` (``resumable=False``) so the orchestrator fails
the job instead of waiting on it; callers do not pass a ``resume_token`` to
this provider.
"""

import math
import re
from dataclasses import dataclass

import httpx

from app.client.http import get_shared_http_client
from app.core.exceptions import ExternalServiceError, PermanentExternalServiceError
from app.core.logging import get_logger
from app.providers.base import TranscriptProviderStrategy, VideoTranscriptResult
from app.providers.transcript import (
    TranscriptData,
    TranscriptJobPending,
    TranscriptWordData,
    _close_word_times,
    _language_base,
)

logger = get_logger(__name__)

_BASE_URL = "https://transcriptfetch.com"
_TIMEOUT_SECONDS = 65.0

# Content-level error codes that mean "no transcript can come from this video"
# (a soft miss — the caller should fall through to the next provider/audio path)
# or "this request is permanently unfulfillable".
_NO_TRANSCRIPT_ERRORS = {
    "no_captions",
    "captions_disabled",
    "no_speech",
    "no_audio_stream",
    "audio_ineligible",
    "drm_protected",
    "private",
    "live_stream",
}

_PERMANENT_ERRORS = {
    "unauthorized",
    "idempotency_conflict",
    "not_found",
    "insufficient_credits",
    "invalid_request",
    "invalid_input",
    "unsupported_platform",
    "endpoint_platform_mismatch",
    "audio_too_long",
    "members_only",
    "age_restricted",
    "region_blocked",
    "was_live",
}


class TranscriptFetchPermanentError(PermanentExternalServiceError):
    """TranscriptFetch failure no fallback can recover (fast-fail)."""


@dataclass(frozen=True)
class TranscriptFetchResult(VideoTranscriptResult):
    """Transcript plus the metadata TranscriptFetch returns for a video."""


class TranscriptFetchTranscriptProvider(TranscriptProviderStrategy):
    """Fetches transcripts and video metadata from the TranscriptFetch API."""

    name = "transcriptfetch"

    def __init__(
        self,
        api_key: str,
        base_url: str = _BASE_URL,
        lang: str = "en",
        mode: str = "auto",
        timeout: float = _TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.lang = lang
        self.mode = mode
        self.timeout = timeout
        self.transport = transport

    async def fetch(
        self,
        youtube_url: str,
        youtube_video_id: str = "",
        resume_token: str = "",
    ) -> TranscriptFetchResult | None:
        """Fetch a transcript for ``youtube_url`` in a single API call.

        One call per search (``resume_token`` is never used here: TranscriptFetch
        jobs are not resumed, so a non-empty token is treated as a pending,
        non-resumable job). Returns ``None`` when the video has no usable
        transcript (caller falls through to the next provider). Raises
        :class:`ExternalServiceError` on API/account failures or permanent
        per-video errors, and :class:`TranscriptJobPending` (``resumable=False``)
        when the request escalated to an async job (202) — never retried.
        """
        if resume_token:
            raise TranscriptJobPending(
                message="Transcription service is still processing; will not be retried",
                provider="transcriptfetch",
                resume_token=resume_token,
                resumable=False,
            ) from None
        payload = {
            "video": youtube_url,
            "mode": self.mode,
            "timestamps": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            client = get_shared_http_client(timeout=self.timeout, transport=self.transport)
            response = await client.post(
                f"{self.base_url}/api/v2/transcripts/video",
                json=payload,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            logger.error("TranscriptFetch request failed", youtube_url=youtube_url, error=str(exc))
            raise ExternalServiceError(
                "Could not fetch the transcript from the transcription service",
                service="transcriptfetch",
            ) from exc

        if response.status_code == 202:
            self._raise_async(response, youtube_url)
        if response.status_code != 200:
            self._raise_http_error(response, youtube_url)

        try:
            body = response.json()
        except ValueError:
            raise ExternalServiceError(
                "Unexpected response from transcription service", service="transcriptfetch"
            ) from None
        if not body.get("ok"):
            return self._raise_error_block(body.get("error") or {}, youtube_url)
        data = body.get("data") or {}
        segments = list(data.get("segments") or [])
        return self._build_result(data, segments)

    def _build_result(self, data: dict, segments: list[dict]) -> TranscriptFetchResult:
        """Convert a successful TranscriptFetch payload into a result."""
        words = _words_from_segments(segments)
        if words:
            _close_word_times(words)
        text = str(data.get("text") or "")
        if not text:
            text = " ".join(word.word for word in words)
        duration = _duration_from_segments(segments)
        transcript = TranscriptData(
            language=_language_base(self.lang),
            text=text,
            words=words,
        )
        return TranscriptFetchResult(
            title=str(data.get("title") or ""),
            author="",
            duration_seconds=duration,
            is_generated=data.get("source") == "audio",
            transcript=transcript,
        )

    def _raise_http_error(self, response: httpx.Response, youtube_url: str) -> None:
        """Map an HTTP error status to a permanent or transient exception."""
        try:
            body = response.json()
            error = body.get("error") or {}
            code = error.get("code", "")
            message = error.get("message", "")
        except ValueError:
            code = ""
            message = response.text[:200]
        detail = {
            "status_code": response.status_code,
            "transcriptfetch_error": code,
            "message": message,
        }
        if response.status_code in (400, 401, 402, 403, 409, 422):
            raise TranscriptFetchPermanentError(
                "Transcription service rejected the request (check the TranscriptFetch API key)",
                service="transcriptfetch",
                details=detail,
            ) from None
        if response.status_code == 429 or response.status_code >= 500:
            raise ExternalServiceError(
                "Transcription service is rate-limited or unavailable; try again later",
                service="transcriptfetch",
                details=detail,
            ) from None
        raise ExternalServiceError(
            "Transcription service rejected the request",
            service="transcriptfetch",
            details=detail,
        ) from None

    def _raise_async(self, response: httpx.Response, youtube_url: str) -> None:
        """Raise a non-resumable pending error on an HTTP 202."""
        try:
            job_id = (response.json().get("job_id") or "").strip()
        except ValueError:
            job_id = ""
        logger.info(
            "TranscriptFetch escalated to async transcription; not retrying",
            youtube_url=youtube_url,
            job_id=job_id or None,
        )
        raise TranscriptJobPending(
            message="Transcription service is still processing; will not be retried",
            provider="transcriptfetch",
            resume_token=job_id,
            resumable=False,
            details={"transcriptfetch_job_id": job_id or ""},
        ) from None

    def _raise_error_block(self, error: dict, youtube_url: str) -> None:
        """Map a TranscriptFetch content/error block to an exception or miss."""
        code = str(error.get("code") or "")
        message = str(error.get("message") or "")
        detail = {"transcriptfetch_error": code, "message": message}
        if code in _NO_TRANSCRIPT_ERRORS:
            logger.info(
                "TranscriptFetch has no usable transcript", youtube_url=youtube_url, error=code
            )
            if code in ("no_captions", "captions_disabled"):
                # A captionless video is a miss: fall through to the next
                # provider / audio path.
                return None
            raise TranscriptFetchPermanentError(
                "Could not fetch the transcript for this video",
                service="transcriptfetch",
                details=detail,
            ) from None
        if code in _PERMANENT_ERRORS:
            raise TranscriptFetchPermanentError(
                "Could not fetch the transcript for this video",
                service="transcriptfetch",
                details=detail,
            ) from None
        logger.warning(
            "TranscriptFetch transient per-video error",
            youtube_url=youtube_url,
            error=code,
            message=message,
        )
        raise ExternalServiceError(
            "Could not fetch the transcript for this video",
            service="transcriptfetch",
            details=detail,
        ) from None


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
