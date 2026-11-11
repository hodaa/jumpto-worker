"""Generate an original-footage summary video from a lecture transcript.

The AI only scores transcript segments; this service owns every timestamp.
Selected ranges are cut from one downloaded source file and concatenated, so
each cut is frame-accurate and the output streams start cleanly.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from app.client import BackendClient
from app.core.config import Settings, get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.integrations.blob import upload_summary_mp4
from app.integrations.summary_ai import OpenAISummaryAI, SegmentScore, SummaryAI
from app.integrations.video_edit import render_summary
from app.models import SummaryData, SummaryLineData
from app.services.downloads import fetch_video_file
from app.services.jobs import default_client_factory
from app.services.summary_segments import (
    SummarySegment,
    apply_scores,
    build_clips,
    segment_lines,
    select_segments,
    target_duration_seconds,
)

logger = get_logger(__name__)

FAILURE_SAFE_MESSAGE = "Could not generate the video summary. Please try again later."
SUMMARY_QUOTA_MESSAGE = "AI video summaries are temporarily unavailable. Please try again later."
_MAX_BATCH_CHARS = 12000
_MAX_BATCH_SEGMENTS = 40

AIFactory = Callable[[Settings], SummaryAI]
Downloader = Callable[..., Path]


def _default_ai(settings: Settings) -> SummaryAI:
    """Use OpenAI to score segments; no key means summaries cannot run."""
    return OpenAISummaryAI.from_settings(settings)


def _summary_data(payload: dict) -> SummaryData:
    """Build the typed request from the backend's internal response."""
    lines = [SummaryLineData(**line) for line in payload.get("transcript_lines", [])]
    return SummaryData(
        summary_id=payload["summary_id"],
        video_id=payload["video_id"],
        youtube_video_id=payload["youtube_video_id"],
        youtube_url=payload["youtube_url"],
        title=payload["title"],
        status=payload["status"],
        duration_seconds=payload.get("duration_seconds"),
        language=payload.get("language"),
        target_seconds=payload.get("target_seconds"),
        transcript_lines=lines,
    )


async def run_summary(
    summary_id: str,
    client_factory: Callable[..., BackendClient] | None = None,
    ai_factory: AIFactory | None = None,
    downloader: Downloader | None = None,
    renderer: Callable[..., Path] | None = None,
    uploader: Callable[[Path], str] | None = None,
) -> str:
    """Generate one summary video end to end and report its blob URL."""
    settings = get_settings()
    client = (client_factory or default_client_factory)(settings)
    request = _summary_data(await client.get_video_summary(summary_id))
    await client.advance_summary(summary_id)
    blob_url = await _generate_video(
        request,
        settings,
        ai_factory=ai_factory,
        downloader=downloader,
        renderer=renderer,
        uploader=uploader,
    )
    await client.mark_summary_ready(summary_id, blob_url)
    return blob_url


async def fail_summary(
    summary_id: str,
    message: str,
    client_factory: Callable[..., BackendClient] | None = None,
) -> None:
    """Report a failed summary request, best effort."""
    settings = get_settings()
    client = (client_factory or default_client_factory)(settings)
    try:
        await client.mark_summary_failed(summary_id, message)
    except Exception:
        logger.exception("Could not report video-summary failure", summary_id=summary_id)


async def _generate_video(
    request: SummaryData,
    settings: Settings,
    *,
    ai_factory: AIFactory | None,
    downloader: Downloader | None,
    renderer: Callable[..., Path] | None,
    uploader: Callable[[Path], str] | None,
) -> str:
    """Score segments, choose clips, render, and upload the MP4."""
    segments = segment_lines(
        request.transcript_lines,
        clip_seconds=settings.summary_clip_seconds,
        min_clip_seconds=settings.summary_min_clip_seconds,
        duration_seconds=request.duration_seconds,
    )
    if not segments:
        raise ExternalServiceError(
            "The transcript has no usable lines for a summary", service="summary"
        )

    ai = (ai_factory or _default_ai)(settings)
    if not ai.configured:
        raise ExternalServiceError("AI segment scoring is not configured", service="openai")
    scores = await _score_segments(ai, request, segments)
    scored = apply_scores(segments, {score.id: score.score for score in scores})
    target = target_duration_seconds(
        source_seconds=request.duration_seconds,
        requested_seconds=request.target_seconds,
        ratio=settings.summary_target_ratio,
        min_seconds=settings.summary_min_seconds,
        max_seconds=settings.summary_max_seconds,
    )
    selected = select_segments(scored, target)
    clips = build_clips(
        selected,
        source_seconds=request.duration_seconds,
        pad_seconds=settings.summary_pad_seconds,
    )
    logger.info(
        "Selected summary clips",
        summary_id=request.summary_id,
        segments=len(segments),
        selected=len(selected),
        target_seconds=target,
        clips=len(clips),
    )
    return await asyncio.to_thread(
        _render_and_upload,
        request,
        clips,
        settings,
        downloader or fetch_video_file,
        renderer or render_summary,
        uploader or upload_summary_mp4,
    )


async def _score_segments(
    ai: SummaryAI,
    request: SummaryData,
    segments: Sequence[SummarySegment],
) -> list[SegmentScore]:
    """Score all segments in prompt-sized batches."""
    scores: list[SegmentScore] = []
    for batch in _batches(segments):
        scores.extend(
            await ai.score_segments(
                title=request.title,
                language=request.language,
                segments=batch,
            )
        )
    if not scores:
        logger.warning(
            "Segment scoring returned no scores; selecting highest-ranked fallback",
            summary_id=request.summary_id,
        )
    return scores


def _batches(segments: Sequence[SummarySegment]) -> list[list[SummarySegment]]:
    """Pack segments into batches by prompt size and item count."""
    batches: list[list[SummarySegment]] = []
    current: list[SummarySegment] = []
    size = 0
    for segment in segments:
        segment_size = len(segment.text)
        if current and (
            len(current) >= _MAX_BATCH_SEGMENTS or size + segment_size > _MAX_BATCH_CHARS
        ):
            batches.append(current)
            current, size = [], 0
        current.append(segment)
        size += segment_size
    if current:
        batches.append(current)
    return batches


def _render_and_upload(
    request: SummaryData,
    clips: Sequence[tuple[float, float]],
    settings: Settings,
    downloader: Downloader,
    renderer: Callable[..., Path],
    uploader: Callable[[Path], str],
) -> str:
    """Download the source, render the cut, and upload it on a worker thread."""
    with tempfile.TemporaryDirectory(prefix=f"summary-{request.summary_id}-") as tmp:
        directory = Path(tmp)
        source = downloader(
            request.youtube_url,
            directory,
            height=settings.summary_height,
            settings=settings,
        )
        output = renderer(source, clips, directory)
        return uploader(output)
