"""Transcript providers (Assembly.ai live / YouTube captions / deterministic fake)."""

import asyncio
import contextlib
import os
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.providers.ytdlp import build_ydlp_options

logger = get_logger(__name__)

_ASSEMBLY_BASE_URL = "https://api.assemblyai.com/v2"
_POLL_INTERVAL_SECONDS = 5
_MAX_POLL_ATTEMPTS = 120
_UPLOAD_TIMEOUT_SECONDS = 300


@dataclass
class TranscriptWordData:
    """Single word with timing information."""

    word: str
    start_time: float
    end_time: float


@dataclass
class TranscriptData:
    """Full transcript with per-word timestamps."""

    language: str
    text: str
    words: list[TranscriptWordData]


class TranscriptProvider(ABC):
    """Abstract transcript source."""

    @abstractmethod
    async def fetch(
        self,
        youtube_url: str,
        on_progress: Callable[[int], Awaitable[None]] | None = None,
    ) -> TranscriptData:
        """Fetch transcript data for a YouTube URL."""


class FakeTranscriptProvider(TranscriptProvider):
    """Deterministic transcript used when external calls are disabled."""

    _SENTENCE_WORDS = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "the",
        "lazy",
        "dog",
        "hello",
        "world",
        "welcome",
        "to",
        "jumpto",
        "find",
        "exact",
        "timestamps",
        "for",
        "any",
        "phrase",
        "in",
        "the",
        "video",
        "good",
        "luck",
        "with",
        "your",
        "searches",
    ]
    _WORD_GAP_SECONDS = 0.5

    async def fetch(
        self,
        youtube_url: str,
        on_progress: Callable[[int], Awaitable[None]] | None = None,
    ) -> TranscriptData:
        """Build a deterministic transcript from a fixed corpus."""
        words = [
            TranscriptWordData(
                word=word,
                start_time=index * self._WORD_GAP_SECONDS,
                end_time=(index + 1) * self._WORD_GAP_SECONDS,
            )
            for index, word in enumerate(self._SENTENCE_WORDS)
        ]
        text = " ".join(word.word for word in words)
        return TranscriptData(language="en", text=text, words=words)


_SUPPORTED_CAPTION_LANGUAGES = ("en", "ar")


class YouTubeCaptionTranscriptProvider(TranscriptProvider):
    """Transcript provider that downloads YouTube's own caption track (fast path)."""

    async def fetch(
        self,
        youtube_url: str,
        on_progress: Callable[[int], Awaitable[None]] | None = None,
    ) -> TranscriptData:
        """Download and parse the best available caption track for a video."""
        vtt_text, language_code = await asyncio.to_thread(
            _download_caption, youtube_url, _SUPPORTED_CAPTION_LANGUAGES
        )
        return _parse_vtt(vtt_text, language_code)


def _download_caption(youtube_url: str, languages: tuple[str, ...]) -> tuple[str, str]:
    """
    Download a caption track with yt-dlp and return its (text, language).

    yt-dlp handles YouTube client impersonation and retries so the caption
    endpoint is reached without the rate limiting that raw HTTP fetches hit.
    A single best-language track is downloaded to minimize caption requests.
    """
    info = _extract_video_info(youtube_url)
    target = _select_caption_language(info, languages)
    temp_dir = tempfile.mkdtemp(prefix="jumpto-captions-")
    options = build_ydlp_options(
        skip_download=True,
        writesubtitles=True,
        writeautomaticsub=True,
        subtitleslangs=[target],
        outtmpl=str(Path(temp_dir) / "%(id)s.%(ext)s"),
    )
    try:
        import yt_dlp  # Optional dependency, only needed for live calls

        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.extract_info(youtube_url, download=True)
        except yt_dlp.utils.DownloadError as exc:
            logger.warning(
                "yt-dlp failed to download captions",
                error=str(exc),
            )
            raise ExternalServiceError(
                "Failed to download captions", service="youtube-captions"
            ) from exc
        vtt_files = sorted(Path(temp_dir).glob("*.vtt"))
        if not vtt_files:
            raise ExternalServiceError("No captions available", service="youtube-captions")
        chosen = _preferred_vtt_file(vtt_files)
        text = chosen.read_text(encoding="utf-8", errors="replace")
        return text, _caption_language(chosen.name)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _extract_video_info(youtube_url: str) -> dict:
    """Extract full video metadata (including caption tracks) with yt-dlp."""
    import yt_dlp  # Optional dependency, only needed for live calls

    options = build_ydlp_options(skip_download=True)
    with yt_dlp.YoutubeDL(options) as ydl:
        return ydl.extract_info(youtube_url, download=False)


def _select_caption_language(info: dict, supported: tuple[str, ...]) -> str:
    """
    Pick a single caption language to download for a video.

    Prefers the video's original-language caption (``xx-orig``) when it is a
    supported language, then a plain supported code, then a supported
    regional variant.
    """
    lowered = [
        key.lower()
        for key in list((info.get("automatic_captions") or {}).keys())
        + list((info.get("subtitles") or {}).keys())
    ]
    if not lowered:
        raise ExternalServiceError("No captions available", service="youtube-captions")
    for language in supported:
        if f"{language}-orig" in lowered:
            return language
    for language in supported:
        if language in lowered:
            return language
    for language in supported:
        if any(key.startswith(f"{language}-") or key.startswith(f"{language}_") for key in lowered):
            return language
    raise ExternalServiceError("No captions available", service="youtube-captions")


def _preferred_vtt_file(files: list[Path]) -> Path:
    """Pick the caption file in the most preferred supported language."""
    return min(files, key=lambda path: _caption_rank(_caption_language(path.name)))


def _caption_rank(language: str) -> tuple[int, int]:
    """Rank a caption language, supported codes first."""
    if language in _SUPPORTED_CAPTION_LANGUAGES:
        return (0, _SUPPORTED_CAPTION_LANGUAGES.index(language))
    return (1, 0)


def _caption_language(filename: str) -> str:
    """Extract the language code from a caption filename."""
    return _language_base(Path(filename).stem.rsplit(".", 1)[-1])


_VTT_WORD_RE = re.compile(r"<(\d{2}):(\d{2}):(\d{2})\.(\d{3})><c>(.*?)</c>")
_VTT_CUE_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})"
)


def _parse_vtt(vtt_text: str, language_code: str) -> TranscriptData:
    """
    Parse a VTT caption body into timestamped words.

    YouTube auto-captions embed per-word timestamps as inline
    ``<HH:MM:SS.mmm><c>word</c>`` tokens; those carry the timing we need.
    Falls back to line-level timestamps when per-word tokens are absent
    (human-created captions).
    """
    words: list[TranscriptWordData] = []
    for match in _VTT_WORD_RE.finditer(vtt_text):
        hours, minutes, seconds, millis = (int(part) for part in match.groups()[:4])
        raw = match.group(5).strip()
        if not raw:
            continue
        start = hours * 3600 + minutes * 60 + seconds + millis / 1000.0
        words.append(TranscriptWordData(word=raw, start_time=start, end_time=start))
    if not words:
        words = _parse_vtt_lines(vtt_text)
    _close_word_times(words)
    text = " ".join(word.word for word in words)
    return TranscriptData(language=_language_base(language_code), text=text, words=words)


def _parse_vtt_lines(vtt_text: str) -> list[TranscriptWordData]:
    """Parse line-level VTT captions when per-word timestamps are absent."""
    words: list[TranscriptWordData] = []
    lines = vtt_text.splitlines()
    i = 0
    while i < len(lines):
        cue_match = _VTT_CUE_RE.search(lines[i])
        if cue_match:
            groups = [int(x) for x in cue_match.groups()[:4]]
            end_groups = [int(x) for x in cue_match.groups()[4:8]]
            start = groups[0] * 3600 + groups[1] * 60 + groups[2] + groups[3] / 1000.0
            end = end_groups[0] * 3600 + end_groups[1] * 60 + end_groups[2] + end_groups[3] / 1000.0
            i += 1
            cue_lines: list[str] = []
            while i < len(lines) and lines[i].strip() and not _VTT_CUE_RE.search(lines[i]):
                cue_lines.append(lines[i].strip())
                i += 1
            raw_text = " ".join(cue_lines)
            raw_text = re.sub(r"[♪♪]", "", raw_text).strip()
            if raw_text:
                for word in raw_text.split():
                    words.append(TranscriptWordData(word=word, start_time=start, end_time=end))
        else:
            i += 1
    return words


def _close_word_times(words: list[TranscriptWordData]) -> None:
    """Set each word's end_time from the next word's start (or a fallback gap)."""
    for index in range(len(words) - 1):
        words[index].end_time = words[index + 1].start_time
    if words:
        words[-1].end_time = words[-1].start_time + 0.3


def _language_base(code: str) -> str:
    """Reduce a locale/region language code to its base portion."""
    return code.split("-")[0].split("_")[0].lower()


class AssemblyTranscriptProvider(TranscriptProvider):
    """Real Assembly.ai transcription client (word-level timestamps)."""

    def __init__(self, api_key: str, base_url: str = _ASSEMBLY_BASE_URL) -> None:
        self.api_key = api_key
        self.base_url = base_url

    async def fetch(
        self,
        youtube_url: str,
        on_progress: Callable[[int], Awaitable[None]] | None = None,
    ) -> TranscriptData:
        """Download the audio and transcribe it via Assembly.ai."""
        await _report_progress(on_progress, 58, "downloading audio")
        audio_path = await asyncio.to_thread(_download_audio, youtube_url)
        try:
            headers = {"authorization": self.api_key}
            async with httpx.AsyncClient() as client:
                upload_url = await self._upload(client, headers, audio_path, on_progress)
                transcript_id = await self._submit(client, headers, upload_url, on_progress)
                return await self._poll(client, headers, transcript_id, on_progress)
        finally:
            _remove_file(audio_path)

    async def _upload(
        self,
        client: httpx.AsyncClient,
        headers: dict,
        path: str,
        on_progress: Callable[[int], Awaitable[None]] | None = None,
    ) -> str:
        """Upload an audio file and return its public upload_url."""
        await _report_progress(on_progress, 61, "uploading audio")
        audio = Path(path).read_bytes()
        response = await client.post(
            f"{self.base_url}/upload",
            headers={**headers, "content-type": "application/octet-stream"},
            content=audio,
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
        on_progress: Callable[[int], Awaitable[None]] | None = None,
    ) -> str:
        """Create a transcription job and return its id."""
        await _report_progress(on_progress, 64, "starting transcription")
        response = await client.post(
            f"{self.base_url}/transcript",
            headers=headers,
            json={"audio_url": audio_url},
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
        on_progress: Callable[[int], Awaitable[None]] | None = None,
    ) -> TranscriptData:
        """Poll until the transcript is ready and parse word timestamps."""
        for attempt in range(_MAX_POLL_ATTEMPTS):
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            # Climb from 65 toward 69 as polling proceeds.
            await _report_progress(on_progress, min(69, 65 + attempt), "transcribing audio")
            response = await client.get(
                f"{self.base_url}/transcript/{transcript_id}", headers=headers
            )
            if response.status_code != 200:
                continue
            data = response.json()
            if data["status"] == "completed":
                return _parse_assembly_transcript(data)
            if data["status"] == "error":
                raise ExternalServiceError("Transcription service failed", service="assemblyai")
        raise ExternalServiceError("Transcription timed out", service="assemblyai")


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
    """Run a yt-dlp audio download for a URL."""
    import yt_dlp

    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([youtube_url])


def _remove_file(path: str) -> None:
    """Best-effort removal of a temp audio file."""
    with contextlib.suppress(OSError):
        Path(path).unlink()


async def _report_progress(
    on_progress: Callable[[int], Awaitable[None]] | None,
    progress: int,
    stage: str,
) -> None:
    """Report a transcript stage to the progress callback, if provided."""
    if on_progress is None:
        return
    try:
        await on_progress(progress)
    except Exception:
        logger.exception("Progress callback failed", stage=stage, progress=progress)


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


def get_transcript_provider() -> TranscriptProvider:
    """
    Return the transcript provider for the current configuration.

    Returns the fake provider unless fake mode is off AND live calls are
    explicitly enabled with an Assembly.ai API key.
    """
    settings = get_settings()
    if settings.jumpto_transcript_mode.lower() == "fake":
        return FakeTranscriptProvider()
    if settings.jumpto_live_external_calls and settings.assembly_api_key:
        return AssemblyTranscriptProvider(settings.assembly_api_key)
    logger.warning(
        "Live transcription not configured; falling back to fake provider",
        live_external_calls=settings.jumpto_live_external_calls,
        has_api_key=bool(settings.assembly_api_key),
    )
    return FakeTranscriptProvider()
