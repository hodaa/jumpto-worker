"""Unit tests for transcript segmentation and summary clip selection."""

from app.models import SummaryLineData
from app.services.summary_segments import (
    SummarySegment,
    apply_scores,
    build_clips,
    segment_lines,
    select_segments,
    target_duration_seconds,
)


def _line(seconds: float, text: str = "one two three four five") -> SummaryLineData:
    """Build one transcript line."""
    return SummaryLineData(seconds=seconds, text=text)


def _segment(index: int, start: float, end: float, score: float = 5.0) -> SummarySegment:
    """Build one scored segment."""
    return SummarySegment(
        index=index,
        start_seconds=start,
        end_seconds=end,
        text=f"segment {index}",
        score=score,
    )


class TestSegmentLines:
    """Segmentation is deterministic and derives ends from the transcript."""

    def test_groups_contiguous_lines_into_one_window(self) -> None:
        segments = segment_lines([_line(0), _line(3), _line(6)])

        assert len(segments) == 1
        assert segments[0].start_seconds == 0
        assert segments[0].end_seconds > 6
        assert segments[0].text.startswith("one two")

    def test_splits_on_long_pauses(self) -> None:
        segments = segment_lines([_line(0), _line(30)])

        assert len(segments) == 2
        assert [segment.index for segment in segments] == [0, 1]

    def test_closes_a_window_at_the_preferred_length(self) -> None:
        lines = [_line(seconds) for seconds in (0, 3, 6, 9, 12)]

        segments = segment_lines(lines, clip_seconds=10, min_clip_seconds=1)

        assert len(segments) > 1
        assert segments[1].start_seconds >= 9

    def test_merges_a_short_window_into_its_contiguous_neighbour(self) -> None:
        lines = [_line(0, "hi"), _line(0.5, " ".join(["word"] * 100))]

        segments = segment_lines(lines, clip_seconds=1, min_clip_seconds=20)

        assert len(segments) == 1

    def test_ignores_blank_lines_and_sorts_by_time(self) -> None:
        lines = [_line(4, "later"), _line(0, "  "), _line(2, "earlier")]

        segments = segment_lines(lines)

        assert [segment.text for segment in segments] == ["earlier later"]

    def test_clamps_the_last_line_to_the_video_duration(self) -> None:
        segments = segment_lines([_line(95, " ".join(["word"] * 100))], duration_seconds=100)

        assert segments[0].end_seconds == 100

    def test_empty_transcript_has_no_segments(self) -> None:
        assert segment_lines([]) == []


class TestApplyScores:
    """Scores are clamped and unknown ids are ignored."""

    def test_attaches_clamped_scores(self) -> None:
        scored = apply_scores([_segment(0, 0, 10), _segment(1, 10, 20)], {0: 12.5})

        assert [segment.score for segment in scored] == [10.0, 0.0]


class TestSelectSegments:
    """Selection maximises score inside the duration budget."""

    def test_keeps_everything_when_the_budget_is_large(self) -> None:
        segments = [_segment(0, 0, 10), _segment(1, 10, 20)]

        assert select_segments(segments, 100) == segments

    def test_prefers_high_value_segments(self) -> None:
        segments = [
            _segment(0, 0, 100, score=9),
            _segment(1, 100, 200, score=8),
            _segment(2, 200, 300, score=0),
        ]

        selected = select_segments(segments, 200)

        assert [segment.index for segment in selected] == [0, 1]

    def test_returns_the_best_segment_when_all_values_are_low(self) -> None:
        segments = [_segment(0, 0, 100, score=2), _segment(1, 100, 200, score=1)]

        assert [segment.index for segment in select_segments(segments, 100)] == [0]

    def test_preserves_chronological_order(self) -> None:
        segments = [
            _segment(0, 100, 200, score=9),
            _segment(1, 0, 100, score=9),
        ]

        assert [segment.index for segment in select_segments(segments, 1000)] == [1, 0]

    def test_zero_budget_selects_nothing(self) -> None:
        assert select_segments([_segment(0, 0, 10)], 0) == []


class TestTargetDuration:
    """The budget resolves from request, ratio, and hard bounds."""

    def test_uses_the_requested_duration(self) -> None:
        assert (
            target_duration_seconds(
                source_seconds=1200,
                requested_seconds=240,
                ratio=0.25,
                min_seconds=60,
                max_seconds=600,
            )
            == 240
        )

    def test_defaults_to_the_source_ratio(self) -> None:
        assert (
            target_duration_seconds(
                source_seconds=1200,
                requested_seconds=None,
                ratio=0.25,
                min_seconds=60,
                max_seconds=600,
            )
            == 300
        )

    def test_caps_at_the_maximum(self) -> None:
        assert (
            target_duration_seconds(
                source_seconds=7200,
                requested_seconds=None,
                ratio=0.25,
                min_seconds=60,
                max_seconds=600,
            )
            == 600
        )

    def test_never_exceeds_the_source(self) -> None:
        assert (
            target_duration_seconds(
                source_seconds=30,
                requested_seconds=None,
                ratio=0.25,
                min_seconds=60,
                max_seconds=600,
            )
            == 30
        )


class TestBuildClips:
    """Clip ranges are padded, clamped, and merged when adjacent."""

    def test_pads_and_clamps_to_the_source(self) -> None:
        clips = build_clips([_segment(0, 1, 10)], source_seconds=10, pad_seconds=0.5)

        assert clips == [(0.5, 10.0)]

    def test_merges_overlapping_clips(self) -> None:
        clips = build_clips(
            [_segment(0, 10, 20), _segment(1, 20.1, 30)],
            source_seconds=60,
            pad_seconds=0.5,
        )

        assert clips == [(9.5, 30.5)]

    def test_keeps_separate_clips_separate(self) -> None:
        clips = build_clips(
            [_segment(0, 0, 10), _segment(1, 20, 30)],
            source_seconds=60,
            pad_seconds=0.25,
        )

        assert clips == [(0.0, 10.25), (19.75, 30.25)]
