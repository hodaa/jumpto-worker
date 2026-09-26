"""Deepgram audio transcription provider (synchronous audio upload -> transcript).

Unlike Assembly, Deepgram returns the finished transcript inside the HTTP
response of a single pre-recorded request, so this leaf is synchronous: it
never raises ``TranscriptJobPending``, arms no webhook, and accepts no resume.
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

from app.client.http import get_shared_http_client
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    ExternalServiceError,
    PermanentExternalServiceError,
)
from app.core.logging import get_logger
from app.providers.audio import download_audio, remove_file
from app.providers.base import TranscriptService
from app.providers.models import TranscriptData, TranscriptWordData

logger = get_logger(__name__)

_DEEPGRAM_BASE_URL = "https://api.deepgram.com/v1"
_TRANSCRIBE_TIMEOUT_SECONDS = 600
_TRANSCRIBE_CHUNK_BYTES = 1_048_576
_PERMANENT_AUTH_STATUS_CODES = {401, 403}
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


_MODEL_LANGUAGE_MISMATCH_HINT = "No such model/language/tier combination"


class DeepgramTranscriptService(TranscriptService):
    """Deepgram audio transcription client (synchronous, word-level timestamps)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEEPGRAM_BASE_URL,
        model: str = "nova-3",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    @staticmethod
    def _is_model_language_mismatch(response: httpx.Response) -> bool:
        """True when Deepgram rejects the requested model/language combination.

        Some deployments do not ship every language on every model tier. When
        that happens Deepgram answers 400 with an explicit hint naming the
        ''general'' model as the working combination, and it is safe to retry
        that combo before failing the job.
        """
        return response.status_code == 400 and _MODEL_LANGUAGE_MISMATCH_HINT in response.text

    async def fetch(
        self,
        youtube_url: str,
        resume_token: str = "",
        webhook_url: str = "",
        language: str = "",
        audio_path: str = "",
    ) -> TranscriptData:
        """Transcribe a video's audio synchronously and return the transcript.

        ``resume_token`` and ``webhook_url`` are accepted for interface parity
        with the Assembly leaf but never used: Deepgram completes in a single
        request, so nothing is resumable and no callback is armed. ``language``
        is a Deepgram language tag (derived from the video title by the caller)
        threaded into the transcription request when non-empty. ``audio_path``
        is a caller-owned audio file to transcribe in place (download and
        cleanup are then the caller's job, so retries reuse the same file);
        when empty the leaf downloads its own temp file and removes it.
        """
        owns_audio = not audio_path
        if owns_audio:
            audio_path = await asyncio.to_thread(download_audio, youtube_url)
        try:
            client = get_shared_http_client()
            return await self._transcribe(client, audio_path, language)
        finally:
            if owns_audio:
                remove_file(audio_path)

    async def _transcribe(
        self,
        client: httpx.AsyncClient,
        path: str,
        language: str = "",
    ) -> TranscriptData:
        """POST an audio file to the Deepgram pre-recorded endpoint and parse it."""
        total_bytes = Path(path).stat().st_size
        params: dict[str, str] = {
            "model": self.model,
            "punctuate": "true",
            "words": "true",
            "timestamps": "true",
        }
        if language:
            params["language"] = language
        # When the requested model/language pair is unavailable (Arabic on
        # nova-3 for some deployments), Deepgram 400s with a hint to use the
        # ''general'' model. Honoring it keeps the language working before the
        # job fails; each attempt gets a fresh stream so the audio is not
        # consumed by the first request.
        if self.model.startswith("nova"):
            fallback_params = {**params, "model": "general", "tier": self.model}
        else:
            fallback_params = None
        logger.info(
            "Submitting audio to Deepgram; waiting on synchronous transcription",
            size_mib=round(total_bytes / (1024 * 1024), 1),
            language=language or "auto",
            model=self.model,
        )
        try:
            response = await client.post(
                f"{self.base_url}/listen",
                headers={
                    "authorization": f"Token {self.api_key}",
                    "content-type": "audio/webm",
                    "content-length": str(total_bytes),
                },
                params=params,
                content=_stream_file(path),
                timeout=_TRANSCRIBE_TIMEOUT_SECONDS,
            )
            if self._is_model_language_mismatch(response) and fallback_params:
                logger.warning(
                    "Deepgram rejected the configured model/language combination; "
                    "retrying with the general model",
                    model=self.model,
                    fallback_model=fallback_params["model"],
                    tier=fallback_params.get("tier", ""),
                    status_code=response.status_code,
                )
                response = await client.post(
                    f"{self.base_url}/listen",
                    headers={
                        "authorization": f"Token {self.api_key}",
                        "content-type": "audio/webm",
                        "content-length": str(total_bytes),
                    },
                    params=fallback_params,
                    content=_stream_file(path),
                    timeout=_TRANSCRIBE_TIMEOUT_SECONDS,
                )
        except httpx.HTTPError as exc:
            logger.error("Deepgram request failed", error=str(exc))
            raise ExternalServiceError(
                "Transcription service was unreachable", service="deepgram"
            ) from exc
        if response.status_code != 200:
            logger.error(
                "Deepgram transcription failed",
                status_code=response.status_code,
                body=response.text[:300],
            )
            if response.status_code in _PERMANENT_AUTH_STATUS_CODES:
                raise PermanentExternalServiceError(
                    "Transcription service rejected the credentials",
                    service="deepgram",
                    details={"status_code": response.status_code},
                )
            if response.status_code in _RETRYABLE_STATUS_CODES:
                raise ExternalServiceError(
                    "Transcription service rejected the request", service="deepgram"
                )
            raise PermanentExternalServiceError(
                "Transcription service rejected the request (unsupported configuration)",
                service="deepgram",
                details={"status_code": response.status_code},
            )
        transcript = _parse_deepgram_transcript(response.json())
        logger.info(
            "Deepgram transcription completed",
            size_mib=round(total_bytes / (1024 * 1024), 1),
            language=transcript.language,
            n_words=len(transcript.words),
        )
        return transcript


async def _stream_file(path: str) -> AsyncIterator[bytes]:
    """Yield raw audio chunks as the POST body with explicit framing.

    httpx consumes this generator as the request body; the caller supplies
    ``content-length`` so the transfer keeps explicit framing instead of
    falling back to chunked encoding.
    """
    with Path(path).open("rb") as audio:
        while chunk := audio.read(_TRANSCRIBE_CHUNK_BYTES):
            yield chunk


def _parse_deepgram_transcript(data: dict) -> TranscriptData:
    """Convert a Deepgram /listen response into TranscriptData.

    Deepgram reports word times in seconds already (unlike Assembly, which
    reports milliseconds), so start/end values are used as-is.
    """
    channels = data.get("results", {}).get("channels", [])
    if not channels:
        raise ExternalServiceError("Transcription returned no audio", service="deepgram")
    alternatives = channels[0].get("alternatives", [])
    if not alternatives:
        raise ExternalServiceError("Transcription returned no alternatives", service="deepgram")
    alternative = alternatives[0]
    words = [
        TranscriptWordData(
            word=str(raw.get("word") or ""),
            start_time=float(raw["start"]),
            end_time=float(raw["end"]),
        )
        for raw in alternative.get("words", [])
    ]
    language = channels[0].get("detected_language") or alternative.get("detected_language") or "en"
    return TranscriptData(
        language=str(language),
        text=str(alternative.get("transcript") or ""),
        words=words,
    )


def get_deepgram_provider(settings: Settings | None = None) -> TranscriptService | None:
    """
    Return the Deepgram audio transcription service, or ``None``.

    ``None`` means Deepgram transcription is unavailable (live calls disabled
    or no API key), so the caller can still use the caption fast path and fail
    cleanly when captions are absent instead of fabricating data. ``settings``
    may be injected by callers that already resolved config; when ``None`` the
    global settings are fetched.
    """
    settings = settings or get_settings()
    if settings.live_external_calls and settings.deepgram_api_key:
        return DeepgramTranscriptService(
            settings.deepgram_api_key,
            settings.deepgram_api_url,
            settings.deepgram_model,
        )
    logger.warning(
        "Audio transcription not configured",
        live_external_calls=settings.live_external_calls,
        has_api_key=bool(settings.deepgram_api_key),
    )
    return None
