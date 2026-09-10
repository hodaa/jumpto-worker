"""Assembly.ai transcription provider (audio upload -> transcript)."""

import asyncio
import contextlib
import os
import tempfile
from pathlib import Path

import httpx
import yt_dlp

from app.client.http import get_shared_http_client
from app.core.config import Settings, get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.integrations.ytdlp import (
    build_ydlp_options,
    is_youtube_bot_check,
    release_temp_cookie,
    request_cookie_refresh,
)
from app.providers.base import TranscriptService
from app.providers.models import (
    TranscriptData,
    TranscriptJobPending,
    TranscriptWordData,
)

logger = get_logger(__name__)

_ASSEMBLY_BASE_URL = "https://api.assemblyai.com/v2"
_UPLOAD_TIMEOUT_SECONDS = 300


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
    ) -> TranscriptData:
        """Submit or resume an Assembly.ai transcription without blocking a slot.

        On the initial submit, ``webhook_url`` (a public callback URL built by
        the pipeline) is attached to the Assembly job so completion is delivered
        out-of-band; a ``TranscriptJobPending`` raised for that job is flagged
        ``webhook=True`` so the task ends instead of poller-retrying. Resume
        polls never arm webhooks (Assembly fires the callback once, on submit).
        """
        headers = {"authorization": self.api_key}
        client = get_shared_http_client()
        if resume_token:
            return await self._poll(client, headers, resume_token)

        audio_path = await asyncio.to_thread(_download_audio, youtube_url)
        try:
            upload_url = await self._upload(client, headers, audio_path)
            transcript_id = await self._submit(client, headers, upload_url, webhook_url)
            return await self._poll(client, headers, transcript_id, webhook=bool(webhook_url))
        finally:
            _remove_file(audio_path)

    async def _upload(
        self,
        client: httpx.AsyncClient,
        headers: dict,
        path: str,
    ) -> str:
        """Upload an audio file and return its public upload_url."""
        with Path(path).open("rb") as audio:
            response = await client.post(
                f"{self.base_url}/upload",
                headers={**headers, "content-type": "application/octet-stream"},
                content=audio.read(),
                timeout=_UPLOAD_TIMEOUT_SECONDS,
            )
        if response.status_code != 200:
            logger.error("Assembly upload failed", status_code=response.status_code)
            raise ExternalServiceError("Audio upload failed", service="assemblyai")
        upload_url = str(response.json().get("upload_url") or "")
        if not upload_url:
            logger.error("Assembly upload returned no url")
            raise ExternalServiceError("Audio upload failed", service="assemblyai")
        return upload_url

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


def _download_audio(youtube_url: str) -> str:
    """Download a YouTube audio stream to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".webm")
    os.close(fd)
    destination = Path(path)
    destination.unlink(missing_ok=True)
    options = build_ydlp_options(
        format="bestaudio/best",
        outtmpl=path,
    )
    try:
        _run_download(options, youtube_url)
        if not destination.exists() or destination.stat().st_size == 0:
            logger.error("Audio download produced no file", path=path)
            raise ExternalServiceError("Audio download produced no file", service="yt-dlp")
        return path
    except ExternalServiceError:
        _remove_file(path)
        raise
    except Exception as exc:
        _remove_file(path)
        logger.error("Audio download failed", error=str(exc))
        raise ExternalServiceError("Could not download audio", service="yt-dlp") from exc


def _run_download(options: dict, youtube_url: str) -> None:
    """Run a fresh yt-dlp audio download for a URL.

    Always extract and download the URL rather than replaying a previously
    extracted ``info`` dict through ``process_ie_result``: YouTube expires the
    video-serving URLs inside a stored ``info`` within seconds, so replays fail
    with HTTP 403 when the caption fast path missed and we fall back to audio.
    """
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([youtube_url])
    except yt_dlp.utils.DownloadError as exc:
        if is_youtube_bot_check(exc):
            request_cookie_refresh()
        raise
    finally:
        release_temp_cookie(options)


def _remove_file(path: str) -> None:
    """Best-effort removal of a temp audio file."""
    with contextlib.suppress(OSError):
        Path(path).unlink()


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
    if settings.jumpto_live_external_calls and settings.assembly_api_key:
        return AssemblyTranscriptService(settings.assembly_api_key)
    logger.warning(
        "Audio transcription not configured",
        live_external_calls=settings.jumpto_live_external_calls,
        has_api_key=bool(settings.assembly_api_key),
    )
    return None
