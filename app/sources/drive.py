"""Google Drive files as a worker video source.

Public Drive files are fetched through the same ``uc?export=download``
endpoint a browser uses. Large files answer it with a virus-scan interstitial:
an HTML page carrying a one-time confirm form that must be resubmitted before
the bytes are served. Downloads stream to disk; sections are cut from the
fetched file with ffmpeg, and the transcription audio is extracted from that
same file, because Drive exposes no per-track endpoints.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx

from app.core.config import Settings, get_settings
from app.core.exceptions import ExternalServiceError, PermanentExternalServiceError
from app.core.logging import get_logger
from app.sources.base import CaptionlessSource, PreparedAudio, VideoRef

logger = get_logger(__name__)

HOST = "drive.google.com"

# Drive file ids are opaque and longer than YouTube's 11; they are 25+ chars
# and met in practice at 28 or 33. Mirrors the backend so both sides accept the
# same links.
FILE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{25,}")

# The interstitial's hidden fields carry the one-time confirm parameters.
_FORM_ACTION_PATTERN = re.compile(r'<form[^>]*action="([^"]+)"')
_FORM_FIELD_PATTERN = re.compile(r'name="([^"]+)"\s+value="([^"]*)"')

# The filename the download response advertises, quoted or not.
_FILENAME_PATTERN = re.compile(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)")

_ACCESS_FAILURE = "This Google Drive file is not public or was not found."
_NOT_A_FILE_LINK = "This Google Drive link does not point at a file."

_PARTIAL_FILENAME = "download.part"
_DEFAULT_FILENAME = "video.mp4"
_DEFAULT_TITLE = "Google Drive video"
_AUDIO_FILENAME = "audio.m4a"
_AUDIO_BITRATE = "128k"
_SECTION_SUFFIX = ".mp4"
_MIN_SECTION_SECONDS = 1

_FFMPEG = "ffmpeg"
_FFPROBE = "ffprobe"
_DOWNLOAD_TIMEOUT_SECONDS = 60
_ERROR_TAIL_CHARS = 500

# Permanent failures the Drive API answers, keyed by status code.
_DRIVE_API_FAILURE = "The Drive API could not serve the file."
_DRIVE_KEY_FAILURE = "The Drive API rejected the request; check GOOGLE_API_KEY."
_DOWNLOAD_FAILURES = {
    400: _DRIVE_KEY_FAILURE,
    401: _ACCESS_FAILURE,
    403: _ACCESS_FAILURE,
    404: _ACCESS_FAILURE,
}
_BYTES_PER_MB = 1024 * 1024
_SECONDS_FLOOR = 0.001


def extract_file_id(url: str) -> str | None:
    """Extract a Drive file id from /file/d/<id> paths or ?id=<id> queries.

    Returns None for anything else, including folder links: a folder is not a
    video, and claiming it here would only move the failure downstream.
    """
    parts = urlsplit(url)
    segments = [segment for segment in parts.path.split("/") if segment]
    if len(segments) >= 3 and segments[0] == "file" and segments[1] == "d":
        return segments[2] if FILE_ID_PATTERN.fullmatch(segments[2]) else None
    query_ids = httpx.QueryParams(parts.query).get_list("id")
    if query_ids and FILE_ID_PATTERN.fullmatch(query_ids[0]):
        return query_ids[0]
    return None


def is_drive_url(url: str) -> bool:
    """Whether the URL points at the Drive host."""
    return urlsplit(url).netloc.lower() == HOST


def normalize_drive_url(file_id: str) -> str:
    """Build the canonical Drive file URL for a file id."""
    return f"https://{HOST}/file/d/{file_id}"


class DriveSource(CaptionlessSource):
    """Drive files. Searchable through our own transcription; no quality ladder."""

    name = "drive"
    native_captions = False

    def matches(self, url: str) -> bool:
        """Whether the URL points at the Drive host."""
        return is_drive_url(url)

    def identify(self, url: str) -> VideoRef:
        """Parse the URL into a VideoRef, or raise when it is malformed."""
        file_id = _require_file_id(url)
        logger.debug("Parsed Drive URL", file_id=file_id, url=url)
        return VideoRef(
            source=self.name,
            source_id=file_id,
            canonical_url=normalize_drive_url(file_id),
        )

    def download(
        self,
        url: str,
        directory: Path,
        *,
        start_seconds: int | None = None,
        end_seconds: int | None = None,
        height: int | None = None,
        settings: Settings | None = None,
    ) -> Path:
        """Fetch the file into ``directory``, cut to a section when one is asked.

        ``height`` is accepted for interface parity and ignored: Drive serves
        the file exactly as uploaded, with no ladder to cap.
        """
        media_path = download_media(_require_file_id(url), directory)
        if start_seconds is None:
            return media_path
        return _cut_section(media_path, directory, start_seconds, end_seconds)

    def prepare_audio(
        self, url: str, directory: Path, settings: Settings | None = None
    ) -> PreparedAudio:
        """Fetch the file, extract its audio, and read the metadata on the way."""
        media_path = download_media(_require_file_id(url), directory)
        try:
            title = media_path.stem or _DEFAULT_TITLE
            duration = probe_duration_seconds(media_path)
            audio_path = extract_audio(media_path, directory)
        finally:
            media_path.unlink(missing_ok=True)
        return PreparedAudio(
            title=title,
            duration_seconds=duration,
            audio_path=audio_path,
            directory=directory,
        )


def download_media(file_id: str, directory: Path) -> Path:
    """Stream a Drive file into ``directory`` and return its named path.

    The final name comes from the response's content-disposition, so the
    uploader's own title is kept; bytes land in a temp file and are renamed
    only once the stream completed.
    """
    directory.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    with _session() as client:
        response = _open_download(client, file_id, settings)
        try:
            filename = _response_filename(response)
            temp_path = directory / _PARTIAL_FILENAME
            _stream_download(response, temp_path, settings)
        finally:
            response.close()
    destination = directory / filename
    temp_path.replace(destination)
    return destination


def probe_duration_seconds(media_path: Path) -> int:
    """Read the media duration in seconds with ffprobe; 0 when unreadable.

    Duration is nice-to-have (progress bars, moment deep links); a file whose
    container ffprobe cannot parse is still transcribed and searched.
    """
    command = [
        _FFPROBE,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(media_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.warning("ffprobe could not read the duration", path=str(media_path))
        return 0
    try:
        return int(float(json.loads(result.stdout)["format"]["duration"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 0


def extract_audio(media_path: Path, directory: Path) -> Path:
    """Extract an AAC audio file for the transcription upload."""
    output = directory / _AUDIO_FILENAME
    args = [
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        _AUDIO_BITRATE,
        str(output),
    ]
    _run_ffmpeg(args, "Could not extract the video audio")
    if not output.is_file() or output.stat().st_size == 0:
        raise ExternalServiceError("Audio extraction produced no file", service="drive")
    return output


def _require_file_id(url: str) -> str:
    """The Drive file id in ``url``, or a permanent failure when there is none."""
    file_id = extract_file_id(url)
    if not file_id:
        raise PermanentExternalServiceError(_NOT_A_FILE_LINK, service="drive")
    return file_id


def _download_url(file_id: str) -> str:
    """The endpoint that serves the file (or the confirm interstitial)."""
    return f"https://{HOST}/uc?export=download&id={file_id}"


def _api_download_url(base_url: str, file_id: str) -> str:
    """The Drive API media endpoint for one file."""
    return f"{base_url.rstrip('/')}/files/{file_id}?alt=media&supportsAllDrives=true"


def _open_download(client: httpx.Client, file_id: str, settings: Settings) -> httpx.Response:
    """Open the file stream, preferring the Drive API when a key is set."""
    api_key = settings.google_api_key.strip()
    if api_key:
        return _open_api_download(client, file_id, api_key, settings.google_drive_api_url)
    return _open_web_download(client, file_id)


def _open_api_download(
    client: httpx.Client, file_id: str, api_key: str, base_url: str
) -> httpx.Response:
    """Stream a public file from the faster Drive API media endpoint."""
    request = client.build_request(
        "GET",
        _api_download_url(base_url, file_id),
        headers={"X-Goog-Api-Key": api_key},
    )
    response = client.send(request, stream=True)
    failure = _DOWNLOAD_FAILURES.get(response.status_code)
    if failure is not None:
        response.read()
        response.close()
        raise PermanentExternalServiceError(failure, service="drive")
    if response.is_error or _is_html(response):
        response.read()
        response.close()
        raise ExternalServiceError(_DRIVE_API_FAILURE, service="drive")
    return response


def _open_web_download(client: httpx.Client, file_id: str) -> httpx.Response:
    """Open the file stream, resubmitting the confirm form when Drive asks.

    Public files stream directly; larger ones answer with the virus-scan
    interstitial whose one-time confirm form must be submitted to get the
    bytes. Any HTML answer without that form is an access failure.
    """
    request = client.build_request("GET", _download_url(file_id))
    response = client.send(request, stream=True)
    if not _is_html(response):
        return response
    confirm = _confirm_url(response.read())
    response.close()
    if confirm is None:
        raise PermanentExternalServiceError(_ACCESS_FAILURE, service="drive")
    response = client.send(client.build_request("GET", confirm), stream=True)
    if _is_html(response):
        response.read()
        response.close()
        raise PermanentExternalServiceError(_ACCESS_FAILURE, service="drive")
    return response


def _confirm_url(body: bytes) -> httpx.URL | None:
    """The interstitial's confirm request URL, or None when it is not one."""
    text = body.decode("utf-8", errors="replace")
    action = _FORM_ACTION_PATTERN.search(text)
    if action is None:
        return None
    fields = dict(_FORM_FIELD_PATTERN.findall(text))
    if not fields.get("id"):
        return None
    return httpx.URL(action.group(1), params=fields)


def _stream_download(response: httpx.Response, path: Path, settings: Settings) -> None:
    """Stream the body to disk and log throughput for diagnosis."""
    started = time.monotonic()
    _stream_to_file(response, path)
    elapsed = max(time.monotonic() - started, _SECONDS_FLOOR)
    megabytes = path.stat().st_size / _BYTES_PER_MB
    logger.info(
        "Drive media downloaded",
        endpoint="api" if settings.google_api_key.strip() else "web",
        megabytes=round(megabytes, 1),
        seconds=round(elapsed, 1),
        mb_per_second=round(megabytes / elapsed, 2),
    )


def _stream_to_file(response: httpx.Response, path: Path) -> None:
    """Write a streamed response to ``path``; a zero-byte file is a failure."""
    with path.open("wb") as handle:
        for chunk in response.iter_bytes():
            handle.write(chunk)
    if not path.is_file() or path.stat().st_size == 0:
        raise ExternalServiceError("Drive download produced no file", service="drive")


def _is_html(response: httpx.Response) -> bool:
    """Whether the response is Drive's HTML page rather than file bytes."""
    return response.headers.get("content-type", "").startswith("text/html")


def _response_filename(response: httpx.Response) -> str:
    """The sanitized filename the download advertises, with a safe default."""
    disposition = response.headers.get("content-disposition", "")
    match = _FILENAME_PATTERN.search(disposition)
    name = unquote(match.group(1)).strip() if match else ""
    name = re.sub(r"[\\/]", "_", name).strip().strip(".")
    return name or _DEFAULT_FILENAME


def _cut_section(
    media_path: Path, directory: Path, start_seconds: int, end_seconds: int | None
) -> Path:
    """Cut one section out of a fetched file with ffmpeg, stream-copying."""
    output = directory / f"{media_path.stem}{_SECTION_SUFFIX}"
    args = ["-y", "-ss", str(start_seconds), "-i", str(media_path)]
    if end_seconds is not None:
        end = max(end_seconds - start_seconds, _MIN_SECTION_SECONDS)
        args += ["-t", str(end)]
    args += ["-c", "copy", "-movflags", "+faststart", str(output)]
    _run_ffmpeg(args, "Could not cut the requested section")
    if not output.is_file() or output.stat().st_size == 0:
        raise ExternalServiceError("Section cut produced no file", service="drive")
    return output


def _run_ffmpeg(args: list[str], failure_message: str) -> None:
    """Run one ffmpeg invocation, turning a failure into a typed error."""
    command = [_FFMPEG, "-hide_banner", "-loglevel", "error", *args]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("ffmpeg failed", stderr=result.stderr[-_ERROR_TAIL_CHARS:])
        raise ExternalServiceError(failure_message, service="drive")


def _transport() -> httpx.BaseTransport | None:
    """The HTTP transport Drive requests go through; a seam for tests."""
    return None


def _session() -> httpx.Client:
    """One short-lived client per fetch; the file stream is one-shot."""
    return httpx.Client(
        follow_redirects=True,
        timeout=_DOWNLOAD_TIMEOUT_SECONDS,
        transport=_transport(),
    )
