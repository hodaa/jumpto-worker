"""Cut and concatenate original footage with ffmpeg.

Every clip is re-encoded with the same codec, resolution, and audio settings
before concatenation, so the concat demuxer can stream-copy the joined file
instead of failing on mismatched timebases or codec parameters.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger

logger = get_logger(__name__)

FFMPEG = "ffmpeg"
_SERVICE = "ffmpeg"
_ERROR_TAIL_CHARS = 500
_VIDEO_CODEC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
_AUDIO_CODEC = ["-c:a", "aac", "-b:a", "128k"]
_FASTSTART = ["-movflags", "+faststart"]
_CLIP_SUFFIX = ".mp4"
_LIST_FILENAME = "clips.txt"
_OUTPUT_NAME = "summary.mp4"


def render_summary(
    source: Path,
    clips: Sequence[tuple[float, float]],
    directory: Path,
) -> Path:
    """Cut each range from ``source``, concatenate them, and return the MP4."""
    if not clips:
        raise ExternalServiceError("No clips were selected for the summary", service=_SERVICE)
    clip_paths = [
        cut_clip(source, directory / f"clip-{index}{_CLIP_SUFFIX}", start, end)
        for index, (start, end) in enumerate(clips)
    ]
    output = directory / _OUTPUT_NAME
    concat_clips(clip_paths, output)
    return output


def cut_clip(source: Path, output: Path, start_seconds: float, end_seconds: float) -> Path:
    """Re-encode one accurate clip from ``source`` into ``output``.

    ``-ss`` before ``-i`` seeks quickly; because the clip is re-encoded, the
    cut itself is still frame-accurate.
    """
    duration = max(end_seconds - start_seconds, 0.1)
    args = [
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        *_VIDEO_CODEC,
        *_AUDIO_CODEC,
        *_FASTSTART,
        str(output),
    ]
    _run_ffmpeg(args, "Could not cut a summary clip")
    _require_output(output, "Summary clip")
    return output


def concat_clips(clips: Sequence[Path], output: Path) -> Path:
    """Concatenate identically encoded clips without re-encoding."""
    if not clips:
        raise ExternalServiceError("No summary clips to concatenate", service=_SERVICE)
    list_path = output.parent / _LIST_FILENAME
    list_path.write_text("".join(f"file '{_quote(path)}'\n" for path in clips))
    args = [
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        *_FASTSTART,
        str(output),
    ]
    _run_ffmpeg(args, "Could not concatenate summary clips")
    _require_output(output, "Summary video")
    return output


def _quote(path: Path) -> str:
    """Escape a path for the concat demuxer's single-quoted list syntax."""
    return str(path).replace("'", "'\\''")


def _require_output(path: Path, label: str) -> None:
    """Reject an ffmpeg run that claimed success but produced no file."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ExternalServiceError(f"{label} produced no file", service=_SERVICE)


def _run_ffmpeg(args: list[str], failure_message: str) -> None:
    """Run one ffmpeg invocation, turning a failure into a typed error."""
    command = [FFMPEG, "-hide_banner", "-loglevel", "error", *args]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("ffmpeg failed", stderr=result.stderr[-_ERROR_TAIL_CHARS:])
        raise ExternalServiceError(failure_message, service=_SERVICE)
