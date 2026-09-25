"""YouTube caption downloader and VTT parsing for the transcript pipeline."""

import asyncio
import re
import shutil
import tempfile
from pathlib import Path

import yt_dlp

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
    TranscriptWordData,
    _close_word_times,
    _language_base,
)

logger = get_logger(__name__)


class YouTubeCaptionTranscriptService(TranscriptService):
    """YouTube caption downloader (works for any subtitle language)."""

    async def fetch(
        self,
        youtube_url: str,
        info: dict | None = None,
    ) -> TranscriptData | None:
        """Download and parse the best available caption track for a video.

        Returns ``None`` when the video has no usable caption tracks, so the
        caller can fall back to audio transcription without treating a normal
        caption-less video as an error.
        """
        caption = await asyncio.to_thread(_download_caption, youtube_url, info)
        if caption is None:
            return None
        vtt_text, language_code = caption
        return _parse_vtt(vtt_text, language_code)


def _download_caption(youtube_url: str, info: dict | None = None) -> tuple[str, str] | None:
    """
    Download the best available caption track and return its (text, language).

    yt-dlp handles YouTube client impersonation and retries so the caption
    endpoint is reached without the rate limiting that raw HTTP fetches hit.
    A single best track is downloaded to minimize caption requests.

    When ``info`` is provided it is replayed through ``process_ie_result`` so
    yt-dlp downloads captions without re-extracting the video metadata; the
    ``requested_subtitles`` list is recomputed from the cached caption tracks.

    Returns ``None`` when the video has no usable caption tracks — a normal
    outcome for many videos, not an error. Genuine download failures still
    raise ``ExternalServiceError``.
    """
    if info is None:
        info = _extract_video_info(youtube_url)
    targets = _caption_targets(info)
    if not targets:
        return None
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
    return None


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
    Pick caption tracks to download, preferring the video's own language.

    Tracks are first restricted to the video's ``original_language`` when the
    metadata reports it: translated subtitles must never pollute the search
    index, and a video whose original-language tracks are missing yields no
    caption targets (a soft miss that falls back to audio transcription). When
    ``original_language`` is absent, the reported ``language`` metadata (e.g.
    ``en-US``) is used as a hint so an English video does not silently pick a
    translated Arabic caption. If neither signal exists -- or the hinted
    language has no track -- the best available track is used rather than
    dropping captions entirely. Manual subtitles beat auto-captions, and
    ``xx-orig`` beats plain ``xx`` within the same language.
    """
    manual = list((info.get("subtitles") or {}).keys())
    auto = list((info.get("automatic_captions") or {}).keys())
    original = _caption_primary_language(info)
    if original:
        manual = _matching_tracks(manual, original)
        auto = _matching_tracks(auto, original)
        if not manual and not auto:
            return []
    elif info.get("language"):
        hint = _language_base(str(info.get("language")))
        preferred_manual = _matching_tracks(manual, hint)
        preferred_auto = _matching_tracks(auto, hint)
        if preferred_manual or preferred_auto:
            manual, auto = preferred_manual, preferred_auto
            original = hint
    if not manual and not auto:
        return []
    best = _best_track(manual, auto, original)
    targets = [best]
    base = _language_base(best)
    if base != best and base in (manual + auto):
        targets.append(base)
    return targets


def _caption_primary_language(info: dict) -> str:
    """Return the base language of the video's reported original language."""
    raw = str(info.get("original_language") or "").strip()
    return _language_base(raw) if raw else ""


def _matching_tracks(tracks: list[str], language: str) -> list[str]:
    """Return only caption tracks whose base language matches ``language``."""
    return [track for track in tracks if _language_base(track) == language]


def _best_track(manual: list[str], auto: list[str], primary: str) -> str:
    """Pick the best track: primary language, manual, ``-orig``, then name."""

    def rank(track: str) -> tuple[int, int, int, str]:
        base_matches = 0 if (primary and _language_base(track) == primary) else 1
        group = 0 if track in manual else 1
        orig = 0 if track.endswith("-orig") else (1 if "-" not in track else 2)
        return (base_matches, group, orig, track)

    if manual and auto:
        return min([min(manual, key=rank), min(auto, key=rank)], key=rank)
    tracks = manual if manual else auto
    return min(tracks, key=rank)


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
