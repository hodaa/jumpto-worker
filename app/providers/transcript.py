"""Transcript providers (YouTube captions / deterministic fake)."""

import asyncio
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yt_dlp

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.integrations.ytdlp import (
    build_ydlp_options,
    is_youtube_bot_check,
    release_temp_cookie,
    request_cookie_refresh,
)

logger = get_logger(__name__)


class TranscriptJobPending(ExternalServiceError):
    """A cloud provider escalated transcription to an async job not yet done.

    Raised by providers that queue work on their side (e.g. Supadata AI
    generation). ``resume_token`` lets a later attempt resume the same job
    instead of queuing a fresh transcription; the orchestrator routes it back
    to the originating provider via ``provider``. ``resumable`` separates jobs
    the caller may wait on (retry with backoff, carrying the resume token)
    from providers running a strict one-call policy (e.g. TranscriptFetch),
    where the job must be failed rather than polled or retried.
    """

    def __init__(
        self,
        message: str = "Transcription service is still processing",
        *,
        provider: str,
        resume_token: str = "",
        resumable: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.resume_token = resume_token
        self.resumable = resumable
        super().__init__(
            message,
            service=provider,
            details={**(details or {}), "resume_token": resume_token},
        )


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
        info: dict | None = None,
        resume_token: str = "",
    ) -> TranscriptData:
        """Fetch transcript data for a YouTube URL.

        ``info`` carries an already-extracted yt-dlp metadata dict so the
        provider can skip a redundant ``extract_info`` call.
        """


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
        info: dict | None = None,
        resume_token: str = "",
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


class YouTubeCaptionTranscriptProvider(TranscriptProvider):
    """YouTube caption downloader (works for any subtitle language)."""

    async def fetch(
        self,
        youtube_url: str,
        info: dict | None = None,
        resume_token: str = "",
    ) -> TranscriptData:
        """Download and parse the best available caption track for a video."""
        vtt_text, language_code = await asyncio.to_thread(_download_caption, youtube_url, info)
        return _parse_vtt(vtt_text, language_code)


def _download_caption(
    youtube_url: str, info: dict | None = None
) -> tuple[str, str]:
    """
    Download the best available caption track and return its (text, language).

    yt-dlp handles YouTube client impersonation and retries so the caption
    endpoint is reached without the rate limiting that raw HTTP fetches hit.
    A single best track is downloaded to minimize caption requests.

    When ``info`` is provided it is replayed through ``process_ie_result`` so
    yt-dlp downloads captions without re-extracting the video metadata; the
    ``requested_subtitles`` list is recomputed from the cached caption tracks.
    """
    if info is None:
        info = _extract_video_info(youtube_url)
    targets = _caption_targets(info)
    vtt_files: list[Path] = []
    for target in targets:
        temp_dir = tempfile.mkdtemp(prefix="jumpto-captions-")
        options = build_ydlp_options(
            skip_download=True,
            writesubtitles=True,
            writeautomaticsub=True,
            subtitleslangs=[target],
            outtmpl=str(Path(temp_dir) / "%(id)s.%(ext)s"),
        )
        try:
            try:
                with yt_dlp.YoutubeDL(options) as ydl:
                    ydl.process_ie_result(info, download=True)
            except yt_dlp.utils.DownloadError as exc:
                logger.warning(
                    "yt-dlp failed to download captions",
                    error=str(exc),
                )
                if is_youtube_bot_check(exc):
                    request_cookie_refresh()
                raise ExternalServiceError(
                    "Failed to download captions", service="youtube-captions"
                ) from exc
            vtt_files = sorted(Path(temp_dir).glob("*.vtt"))
            if vtt_files:
                chosen = _preferred_vtt_file(vtt_files)
                text = chosen.read_text(encoding="utf-8", errors="replace")
                return text, _caption_language(chosen.name)
        finally:
            release_temp_cookie(options)
            shutil.rmtree(temp_dir, ignore_errors=True)
    raise ExternalServiceError("No captions available", service="youtube-captions")


def _extract_video_info(youtube_url: str) -> dict:
    """Extract full video metadata (including caption tracks) with yt-dlp."""
    options = build_ydlp_options(skip_download=True)
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            return ydl.extract_info(youtube_url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        if is_youtube_bot_check(exc):
            request_cookie_refresh()
        raise
    finally:
        release_temp_cookie(options)


def _caption_targets(info: dict) -> list[str]:
    """
    Pick caption tracks to download, in preference order, regardless of language.

    Manual subtitles beat auto-captions (human-curated words), original-audio
    tracks (``xx-orig``) beat translated ones, and tracks in the video's
    ``original_language`` are preferred over other languages. Search only needs
    the words, so no language is ever excluded.
    """
    manual = list((info.get("subtitles") or {}).keys())
    auto = list((info.get("automatic_captions") or {}).keys())
    if not manual and not auto:
        raise ExternalServiceError("No captions available", service="youtube-captions")

    original = str(info.get("original_language") or "").lower().strip()

    def rank(track: str) -> tuple[int, int, int, str]:
        base_matches = 0 if (original and _language_base(track) == original) else 1
        orig = 0 if track.endswith("-orig") else (1 if "-" not in track else 2)
        group = 0 if track in manual else 1
        return (group, orig, base_matches, track)

    def reduce(tracks: list[str]) -> str:
        return min(tracks, key=rank)

    best = reduce(manual) if manual else reduce(auto)
    # Prefer the best of each group, then the best of the merge.
    if manual and auto:
        best = min([reduce(manual), reduce(auto)], key=rank)
    candidates = [best]
    base = _language_base(best)
    if base != best and base in (manual + auto):
        candidates.append(base)
    return candidates


def _preferred_vtt_file(files: list[Path]) -> Path:
    """Pick the best caption file: original-audio, then plain, then regional."""
    return min(files, key=lambda path: _caption_rank(_caption_locale(path.name)))


def _caption_locale(filename: str) -> str:
    """Return the raw locale portion of a caption filename (keeps ``-orig``)."""
    return Path(filename).stem.rsplit(".", 1)[-1]


def _caption_rank(locale: str) -> tuple[int, int, str]:
    """Rank a caption locale, ``-orig`` tracks (original audio) first."""
    if locale.endswith("-orig"):
        return (0, 0, locale)
    if "-" not in locale:
        return (0, 1, locale)
    return (1, 0, locale)


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
        from app.providers.assembly import AssemblyTranscriptProvider

        return AssemblyTranscriptProvider(settings.assembly_api_key)
    logger.warning(
        "Live transcription not configured; falling back to fake provider",
        live_external_calls=settings.jumpto_live_external_calls,
        has_api_key=bool(settings.assembly_api_key),
    )
    return FakeTranscriptProvider()
