"""Supadata transcript provider (cloud fast path, no YouTube access needed).

Supadata runs YouTube extraction on their own infrastructure behind
``x-api-key`` auth, so transcripts and video metadata arrive without yt-dlp,
cookies, or PO-token providers. This pairs with VidWords as a second
cloud provider: if one has no transcript or is failing, the other — and
finally the yt-dlp fallback — can still serve the job.

The generic ``/transcript`` endpoint fetches an existing caption track
(``mode=native``) or falls back to AI transcription (``mode=generate``/
``mode=auto``), so captionless videos get a transcript too. Longer
generations return HTTP 202 with a ``jobId`` instead of a result; the worker
does not block-poll that job. Each ``fetch`` makes one call — either to queue
the job (202) or to check a previously queued ``jobId`` via a resume token —
and raises :class:`TranscriptJobPending` while the job is still processing so
the Celery task can retry later with backoff.

The API returns transcript ``chunks`` with millisecond ``offset``/``duration``;
word records are derived from each chunk so word-level search keeps working at
caption-line granularity.
"""

import math
from dataclasses import dataclass

import httpx

from app.core.exceptions import ExternalServiceError
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

_BASE_URL = "https://api.supadata.ai/v1"
_TIMEOUT_SECONDS = 45.0

_PERMANENT_ERRORS = {
    "invalid-request",
    "unauthorized",
    "forbidden",
    "upgrade-required",
    "not-found",
}


class SupadataPermanentError(ExternalServiceError):
    """Supadata failure that no fallback can recover (account/video-level)."""


@dataclass(frozen=True)
class SupadataResult(VideoTranscriptResult):
    """Transcript plus the metadata Supadata returns for a video."""


class SupadataTranscriptProvider(TranscriptProviderStrategy):
    """Fetches transcripts and video metadata from the Supadata API."""

    name = "supadata"

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
        self.headers = {"x-api-key": api_key, "Content-Type": "application/json"}

    async def fetch(
        self,
        youtube_url: str,
        youtube_video_id: str = "",
        resume_token: str = "",
    ) -> SupadataResult | None:
        """Fetch a transcript for a YouTube video.

        Uses ``mode=auto``: native captions when available, otherwise Supadata
        AI-generates the transcript (no yt-dlp on our side). Each call makes
        exactly one Supadata request: without ``resume_token`` it queues a new
        transcript job (raising :class:`TranscriptJobPending` on a 202), and
        with one it checks that job's current state. Returns ``None`` when no
        transcript could be produced (caller should move to the next provider).
        Raises :class:`ExternalServiceError` on API, account or permanent
        video errors.
        """
        headers = self.headers
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
                if resume_token:
                    payload = await self._job_result(client, resume_token, youtube_url)
                else:
                    response = await client.get(
                        f"{self.base_url}/transcript",
                        params={"url": youtube_url, "lang": self.lang, "mode": self.mode},
                        headers=headers,
                    )
                    if response.status_code == 206:
                        logger.info("Supadata found no transcript", youtube_url=youtube_url)
                        return None
                    if response.status_code == 202:
                        job_id = str(response.json().get("jobId") or "")
                        if not job_id:
                            raise SupadataPermanentError(
                                "No transcript could be produced",
                                service="supadata",
                                details={"status_code": response.status_code},
                            ) from None
                        logger.info(
                            "Supadata transcript job queued",
                            job_id=job_id,
                            youtube_url=youtube_url,
                        )
                        raise TranscriptJobPending(
                            message="Transcription service is still processing; will retry later",
                            provider="supadata",
                            resume_token=job_id,
                            resumable=True,
                        ) from None
                    if response.status_code != 200:
                        self._raise_http_error(response, youtube_url)
                    payload = response.json()
                chunks = _normalize_chunks(payload.get("content"))
                if not chunks:
                    logger.info("Supadata returned no usable content", youtube_url=youtube_url)
                    return None
                metadata = await self._fetch_metadata(client, youtube_url, youtube_video_id)
        except httpx.HTTPError as exc:
            logger.error("Supadata request failed", youtube_url=youtube_url, error=str(exc))
            raise ExternalServiceError(
                "Could not fetch the transcript from the transcription service",
                service="supadata",
            ) from exc

        words = _words_from_chunks(chunks)
        if words:
            _close_word_times(words)
        text = " ".join(
            str(chunk.get("text") or "").strip() for chunk in chunks if chunk.get("text")
        )
        transcript = TranscriptData(
            language=_language_base(str(payload.get("lang") or "en")),
            text=text,
            words=words,
        )
        return SupadataResult(
            title=str(metadata.get("title") or ""),
            author=str((metadata.get("channel") or {}).get("name") or ""),
            duration_seconds=int(metadata.get("duration") or _duration_from_chunks(chunks)),
            is_generated=True,
            transcript=transcript,
        )

    async def _job_result(
        self,
        client: httpx.AsyncClient,
        resume_token: str,
        youtube_url: str,
    ) -> dict:
        """Fetch an async transcript job's current state in a single request."""
        polled = await client.get(
            f"{self.base_url}/transcript/{resume_token}", headers=self.headers
        )
        if polled.status_code != 200:
            self._raise_http_error(polled, youtube_url)
        data = polled.json()
        status = str(data.get("status") or "")
        if status == "completed":
            return {
                "content": data.get("content"),
                "lang": data.get("lang", "en"),
                "availableLangs": data.get("availableLangs", []),
            }
        if status == "failed":
            raise SupadataPermanentError(
                "Could not fetch the transcript for this video",
                service="supadata",
                details={"message": str(data.get("error") or "")},
            ) from None
        logger.info(
            "Supadata transcript job still processing",
            job_id=resume_token,
            youtube_url=youtube_url,
        )
        raise TranscriptJobPending(
            message="Transcription service is still processing; will retry later",
            provider="supadata",
            resume_token=resume_token,
            resumable=True,
        ) from None

    async def _fetch_metadata(
        self,
        client: httpx.AsyncClient,
        youtube_url: str,
        youtube_video_id: str,
    ) -> dict:
        """Fetch video metadata (title, duration, channel) from Supadata."""
        try:
            response = await client.get(
                f"{self.base_url}/youtube/video",
                params={"id": youtube_video_id or youtube_url},
                headers=self.headers,
            )
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                "Could not fetch the transcript for this video",
                service="supadata",
            ) from exc
        if response.status_code != 200:
            self._raise_http_error(response, youtube_url)
        return response.json()

    def _raise_http_error(self, response: httpx.Response, youtube_url: str) -> None:
        """Map an HTTP error response to a permanent or transient exception."""
        try:
            body = response.json()
            code = body.get("error", "")
            message = body.get("message", "")
        except ValueError:
            code = ""
            message = response.text[:200]
        logger.warning(
            "Supadata HTTP error",
            youtube_url=youtube_url,
            status_code=response.status_code,
            error=code,
            message=message,
        )
        detail = {"status_code": response.status_code, "supadata_error": code, "message": message}
        if code in _PERMANENT_ERRORS or response.status_code in (400, 401, 403, 404, 402):
            raise SupadataPermanentError(
                "Could not fetch the transcript for this video",
                service="supadata",
                details=detail,
            ) from None
        raise ExternalServiceError(
            "Transcription service is rate-limited or unavailable; try again later",
            service="supadata",
            details=detail,
        ) from None


def _normalize_chunks(content) -> list[dict]:
    """Normalize the transcript response into a list of chunks.

    Handles both shapes the API returns: a list of chunk objects (``text`` +
    ``offset`` + ``duration``) and a plain-text string (some async job results).
    """
    if isinstance(content, list):
        return [chunk for chunk in content if isinstance(chunk, dict)]
    if isinstance(content, str) and content.strip():
        return [{"text": content, "offset": 0, "duration": 0}]
    return []


def _words_from_chunks(chunks: list[dict]) -> list[TranscriptWordData]:
    """Expand transcript chunks (ms offsets) into word records."""
    words: list[TranscriptWordData] = []
    for chunk in chunks:
        offset_ms = float(chunk.get("offset") or 0)
        duration_ms = float(chunk.get("duration") or 0)
        start = offset_ms / 1000.0
        end = (offset_ms + duration_ms) / 1000.0
        for word in str(chunk.get("text") or "").split():
            words.append(TranscriptWordData(word=word, start_time=start, end_time=end))
    return words


def _duration_from_chunks(chunks: list[dict]) -> int:
    """Derive the video duration from the last transcript chunk end (ms)."""
    if not chunks:
        return 0
    last_end = max(
        (float(chunk.get("offset") or 0) + float(chunk.get("duration") or 0)) / 1000.0
        for chunk in chunks
    )
    return math.ceil(last_end)
