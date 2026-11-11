"""Segment transcripts and choose the highest-value clips for a summary.

Segmentation is deterministic: transcript lines are grouped into roughly
equal-length windows, closing at natural pauses or once a window reaches the
preferred clip length. The AI then scores those segment indices; timestamps
never come from the model, so a bad score can never move a cut.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from app.models import SummaryLineData, SummarySegment

_WORDS_PER_SECOND = 2.6
_MIN_LINE_SECONDS = 0.8
_MIN_VALUE_SCORE = 3.0
_GAP_SECONDS = 2.0
_MERGE_GAP_SECONDS = 2.5


@dataclass(frozen=True)
class _Line:
    """A transcript line with a derived end position."""

    start: float
    end: float
    text: str


def segment_lines(
    lines: Sequence[SummaryLineData],
    *,
    clip_seconds: int = 75,
    min_clip_seconds: int = 20,
    gap_seconds: float = _GAP_SECONDS,
    duration_seconds: int | None = None,
) -> list[SummarySegment]:
    """Group transcript lines into scorable segments with real timestamps."""
    prepared = _prepare(lines, duration_seconds)
    if not prepared:
        return []

    windows: list[list[_Line]] = []
    current: list[_Line] = []
    for line in prepared:
        if current and (
            line.start - current[-1].end > gap_seconds
            or line.end - current[0].start >= clip_seconds
        ):
            windows.append(current)
            current = []
        current.append(line)
    if current:
        windows.append(current)

    merged = _merge_short(windows, min_clip_seconds, gap_seconds)
    return [
        SummarySegment(
            index=index,
            start_seconds=window[0].start,
            end_seconds=window[-1].end,
            text=" ".join(line.text for line in window),
        )
        for index, window in enumerate(merged)
    ]


def apply_scores(
    segments: Sequence[SummarySegment], scores: dict[int, float]
) -> list[SummarySegment]:
    """Attach clamped AI scores to segments, leaving unknown indices at zero."""
    return [
        replace(segment, score=_clamp_score(scores.get(segment.index, 0.0))) for segment in segments
    ]


def select_segments(
    segments: Sequence[SummarySegment], target_seconds: int
) -> list[SummarySegment]:
    """Choose segments that maximise educational value within the budget.

    A 0/1 knapsack over integer seconds; segments below the value baseline
    contribute nothing, so low-value filler is not added just to fill time.
    """
    if not segments or target_seconds <= 0:
        return []
    if sum(_seconds(segment) for segment in segments) <= target_seconds:
        return sorted(segments, key=lambda segment: segment.start_seconds)

    durations = [_seconds(segment) for segment in segments]
    values = [max(0.0, segment.score - _MIN_VALUE_SCORE) for segment in segments]
    capacity = int(target_seconds)

    best = [[0.0] * (capacity + 1) for _ in range(len(segments) + 1)]
    for row, (duration, value) in enumerate(zip(durations, values, strict=True), start=1):
        previous = best[row - 1]
        current = best[row]
        for budget in range(capacity + 1):
            if duration <= budget and previous[budget - duration] + value > previous[budget]:
                current[budget] = previous[budget - duration] + value
            else:
                current[budget] = previous[budget]

    selected: list[SummarySegment] = []
    budget = capacity
    for row in range(len(segments), 0, -1):
        if best[row][budget] > best[row - 1][budget] + 1e-9:
            selected.append(segments[row - 1])
            budget -= durations[row - 1]

    if not selected:
        return [max(segments, key=lambda segment: segment.score)]
    return sorted(selected, key=lambda segment: segment.start_seconds)


def target_duration_seconds(
    *,
    source_seconds: int | None,
    requested_seconds: int | None,
    ratio: float,
    min_seconds: int,
    max_seconds: int,
) -> int:
    """Resolve the summary budget from the request, ratio, and hard bounds."""
    source = max(int(source_seconds or 0), 0)
    if requested_seconds is not None:
        target = int(requested_seconds)
    elif source:
        target = int(round(source * ratio))
    else:
        target = max_seconds
    target = max(min_seconds, min(target, max_seconds))
    return min(target, source) if source else target


def build_clips(
    segments: Sequence[SummarySegment],
    *,
    source_seconds: int | None,
    pad_seconds: float,
) -> list[tuple[float, float]]:
    """Turn selected segments into padded, non-overlapping clip ranges."""
    source = float(source_seconds) if source_seconds else None
    clips: list[tuple[float, float]] = []
    for segment in sorted(segments, key=lambda item: item.start_seconds):
        start = max(0.0, segment.start_seconds - pad_seconds)
        end = segment.end_seconds + pad_seconds
        if source is not None:
            end = min(end, source)
        if end <= start:
            continue
        if clips and start <= clips[-1][1] + 0.05:
            previous_start, previous_end = clips[-1]
            clips[-1] = (previous_start, max(previous_end, end))
        else:
            clips.append((start, end))
    return clips


def _prepare(lines: Sequence[SummaryLineData], duration_seconds: int | None) -> list[_Line]:
    """Sort, drop blanks, and derive each line's end from the next start."""
    entries = sorted(
        ((float(line.seconds), line.text.strip()) for line in lines if line.text.strip()),
        key=lambda item: item[0],
    )
    prepared: list[_Line] = []
    for position, (start, text) in enumerate(entries):
        next_start = entries[position + 1][0] if position + 1 < len(entries) else None
        prepared.append(
            _Line(start=start, end=_line_end(start, text, next_start, duration_seconds), text=text)
        )
    return prepared


def _line_end(
    start: float, text: str, next_start: float | None, duration_seconds: int | None
) -> float:
    """Estimate a line's end, bounded by the next line and the video."""
    estimate = start + max(_MIN_LINE_SECONDS, len(text.split()) / _WORDS_PER_SECOND)
    if next_start is not None and next_start > start:
        estimate = min(estimate, next_start)
    end = max(estimate, start + _MIN_LINE_SECONDS)
    if duration_seconds:
        end = min(end, float(duration_seconds))
    return max(end, start + 0.1)


def _merge_short(
    windows: Sequence[list[_Line]], min_clip_seconds: int, gap_seconds: float
) -> list[list[_Line]]:
    """Merge a too-short window into its contiguous neighbour."""
    merged: list[list[_Line]] = []
    for window in windows:
        if merged and _window_seconds(window) < min_clip_seconds:
            previous = merged[-1]
            if window[0].start - previous[-1].end <= gap_seconds:
                merged[-1] = [*previous, *window]
                continue
        merged.append(window)
    if len(merged) > 1 and _window_seconds(merged[0]) < min_clip_seconds:
        first = merged.pop(0)
        if merged[0][0].start - first[-1].end <= gap_seconds:
            merged[0] = [*first, *merged[0]]
        else:
            merged.insert(0, first)
    return merged


def _window_seconds(window: Sequence[_Line]) -> float:
    return window[-1].end - window[0].start


def _seconds(segment: SummarySegment) -> int:
    return max(1, int(round(segment.duration_seconds)))


def _clamp_score(score: float) -> float:
    return max(0.0, min(float(score), 10.0))
