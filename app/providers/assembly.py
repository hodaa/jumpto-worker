"""Assembly.ai transcription provider (audio upload -> transcript)."""

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

from app.client.http import get_shared_http_client
from app.core.config import Settings, get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.providers.audio import download_audio, remove_file
from app.providers.base import TranscriptService
from app.providers.models import (
    TranscriptData,
    TranscriptJobPending,
    TranscriptWordData,
)

logger = get_logger(__name__)

_ASSEMBLY_BASE_URL = "https://api.assemblyai.com/v2"
_UPLOAD_TIMEOUT_SECONDS = 300
_UPLOAD_CHUNK_BYTES = 1_048_576
_UPLOAD_PROGRESS_LOG_BYTES = 10_485_760
_MIB = 1 << 20


class AssemblyTranscriptService(TranscriptService):
    """Real Assembly.ai transcription client (word-level timestamps)."""

    def __init__(self, api_key: str, base_url: str = _ASSEMBLY_BASE_URL) -> None:
        self.api_key = api_key
        self.base_url = base_url

    async def fetch(
        self,
        youtube_url: str,
        resume_token: str = "",
        webhook_url: str = "",
        language: str = "",
        audio_path: str = "",
    ) -> TranscriptData:
        """Submit or resume an Assembly.ai transcription without blocking a slot.

        On the initial submit, ``webhook_url`` (a public callback URL built by
        the pipeline) is attached to the Assembly job so completion is delivered
        out-of-band; a ``TranscriptJobPending`` raised for that job is flagged
        ``webhook=True`` so the task ends instead of poller-retrying. Resume
        polls never arm webhooks (Assembly fires the callback once, on submit).

        ``language`` is accepted for interface parity with the other audio leaf
        (Deepgram) but ignored here: Assembly detects language itself.
        ``audio_path`` is a caller-owned audio file to upload (download and
        cleanup are then the caller's job, so retries reuse the same file);
        when empty the leaf downloads its own temp file and removes it.
        """
        headers = {"authorization": self.api_key}
        client = get_shared_http_client()
        if resume_token:
            return await self._poll(client, headers, resume_token)

        owns_audio = not audio_path
        if owns_audio:
            audio_path = await asyncio.to_thread(download_audio, youtube_url)
        try:
            upload_url = await self._upload(client, headers, audio_path)
            transcript_id = await self._submit(client, headers, upload_url, webhook_url)
            return await self._poll(client, headers, transcript_id, webhook=bool(webhook_url))
        finally:
            if owns_audio:
                remove_file(audio_path)

    async def _upload(
        self,
        client: httpx.AsyncClient,
        headers: dict,
        path: str,
    ) -> str:
        """Upload an audio file and return its public upload_url."""
        total_bytes = Path(path).stat().st_size
        logger.info("Assembly upload started", path=Path(path).name, size_mib=_to_mib(total_bytes))
        started = time.monotonic()
        response = await client.post(
            f"{self.base_url}/upload",
            headers={
                **headers,
                "content-type": "application/octet-stream",
                "content-length": str(total_bytes),
            },
            content=_stream_audio_with_progress(path, total_bytes),
            timeout=_UPLOAD_TIMEOUT_SECONDS,
        )
        _log_upload_finished(response.status_code, total_bytes, time.monotonic() - started)
        return _parse_upload_response(response)

    async def _submit(
        self,
        client: httpx.AsyncClient,
        headers: dict,
        audio_url: str,
        webhook_url: str = "",
    ) -> str:
        """Create a transcription job and return its id.

        ``webhook_url`` (when non-empty) arms an Assembly.ai completion
        callback so the pipeline does not need to poll for the full
        transcription duration.
        """
        payload: dict[str, object] = {"audio_url": audio_url}
        if webhook_url:
            payload["webhook_url"] = webhook_url
        response = await client.post(
            f"{self.base_url}/transcript",
            headers=headers,
            json=payload,
        )
        if response.status_code != 200:
            logger.error(
                "Assembly submit failed", status_code=response.status_code, body=response.text[:300]
            )
            raise ExternalServiceError(
                "Transcription service rejected the request", service="assemblyai"
            )
        return response.json()["id"]

    async def _poll(
        self,
        client: httpx.AsyncClient,
        headers: dict,
        transcript_id: str,
        webhook: bool = False,
    ) -> TranscriptData:
        """Check once and let the caller back off while Assembly processes.

        ``webhook=True`` marks a job whose completion is delivered externally;
        the pending signal then tells the pipeline to end the task instead of
        retrying.
        """
        response = await client.get(f"{self.base_url}/transcript/{transcript_id}", headers=headers)
        if response.status_code != 200:
            raise ExternalServiceError(
                "Transcription service status check failed", service="assemblyai"
            )
        data = response.json()
        status = str(data.get("status") or "")
        if status == "completed":
            return _parse_assembly_transcript(data)
        if status == "error":
            raise ExternalServiceError("Transcription service failed", service="assemblyai")
        raise TranscriptJobPending(
            message="Transcription service is still processing; will retry later",
            provider="assemblyai",
            resume_token=transcript_id,
            resumable=True,
            webhook=webhook,
        ) from None


def _to_mib(byte_count: int) -> float:
    """Convert a byte count to MiB with two decimal places."""
    return round(byte_count / _MIB, 2)


def _log_upload_finished(status_code: int, total_bytes: int, elapsed: float) -> None:
    """Log the upload outcome with duration and throughput."""
    logger.info(
        "Assembly upload finished",
        status_code=status_code,
        duration_seconds=round(elapsed, 2),
        throughput_mib_s=_to_mib(total_bytes) / max(elapsed, 0.001),
    )


def _parse_upload_response(response: httpx.Response) -> str:
    """Extract and validate an upload_url from an Assembly upload response."""
    if response.status_code != 200:
        logger.error("Assembly upload failed", status_code=response.status_code)
        raise ExternalServiceError("Audio upload failed", service="assemblyai")
    upload_url = str(response.json().get("upload_url") or "")
    if not upload_url:
        logger.error("Assembly upload returned no url")
        raise ExternalServiceError("Audio upload failed", service="assemblyai")
    return upload_url


async def _stream_audio_with_progress(path: str, total_bytes: int) -> AsyncIterator[bytes]:
    """Yield audio file chunks while logging upload progress.

    httpx consumes this generator as the POST body; Content-Length is provided
    by the caller so the transfer keeps explicit framing instead of chunked
    encoding. Progress is logged when each 10 MiB boundary is crossed.
    """
    next_log_bytes = _UPLOAD_PROGRESS_LOG_BYTES
    with Path(path).open("rb") as audio:
        while chunk := audio.read(_UPLOAD_CHUNK_BYTES):
            yield chunk
            sent = audio.tell()
            while sent >= next_log_bytes:
                logger.info(
                    "Assembly upload progress",
                    sent_mib=round(sent / _UPLOAD_CHUNK_BYTES, 2),
                    total_mib=round(total_bytes / _UPLOAD_CHUNK_BYTES, 2),
                    percent=round(sent * 100 / total_bytes, 1),
                )
                next_log_bytes += _UPLOAD_PROGRESS_LOG_BYTES


def _parse_assembly_transcript(data: dict) -> TranscriptData:
    """Convert an Assembly.ai response into TranscriptData."""
    words = [
        TranscriptWordData(
            word=raw["text"],
            start_time=float(raw["start"]) / 1000.0,
            end_time=float(raw["end"]) / 1000.0,
        )
        for raw in data.get("words", [])
    ]
    return TranscriptData(
        language=data.get("language_code", "en"),
        text=str(data.get("text") or ""),
        words=words,
    )


def get_transcript_provider(
    settings: Settings | None = None,
) -> TranscriptService | None:
    """
    Return the Assembly.ai audio transcription service, or ``None``.

    ``None`` means audio transcription is unavailable (live calls disabled or
    no Assembly API key), so the caller can still use the caption fast path
    and fail cleanly when captions are absent instead of fabricating data.
    ``settings`` may be injected by callers that already resolved config;
    when ``None`` the global settings are fetched.
    """
    settings = settings or get_settings()
    if settings.live_external_calls and settings.assembly_api_key:
        return AssemblyTranscriptService(settings.assembly_api_key)
    logger.warning(
        "Audio transcription not configured",
        live_external_calls=settings.live_external_calls,
        has_api_key=bool(settings.assembly_api_key),
    )
    return None
