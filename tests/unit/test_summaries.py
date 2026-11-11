"""Unit tests for original-footage summary orchestration."""

from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.exceptions import ExternalServiceError
from app.integrations.summary_ai import SegmentScore
from app.services import summaries as summary_service
from app.services.summaries import (
    _batches,
    _summary_data,
    fail_summary,
    run_summary,
)
from app.services.summary_segments import SummarySegment


class FakeBackend:
    """In-memory backend client for one summary request."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.advanced: list[str] = []
        self.ready: list[tuple[str, str]] = []
        self.failed: list[tuple[str, str]] = []
        self.fail_ready = False

    async def get_video_summary(self, summary_id: str) -> dict:
        """Return the canned request."""
        return self.payload

    async def advance_summary(self, summary_id: str) -> None:
        """Record the processing callback."""
        self.advanced.append(summary_id)

    async def mark_summary_ready(self, summary_id: str, blob_url: str) -> None:
        """Record the ready callback."""
        self.ready.append((summary_id, blob_url))

    async def mark_summary_failed(self, summary_id: str, error: str) -> None:
        """Record the failure callback."""
        self.failed.append((summary_id, error))


class FakeAI:
    """Scripted segment scorer."""

    def __init__(self, *, configured: bool = True, score: float = 8.0) -> None:
        self.configured = configured
        self.score = score
        self.batches: list[list[SummarySegment]] = []

    async def score_segments(self, *, title, language, segments) -> list[SegmentScore]:
        """Score every supplied segment."""
        self.batches.append(list(segments))
        return [
            SegmentScore(id=segment.index, score=self.score, reason="core") for segment in segments
        ]


class FakeMedia:
    """Capture the download and render calls."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.download: dict = {}
        self.clips: list[tuple[float, float]] = []
        self.uploaded: Path | None = None

    def downloader(self, url: str, directory: Path, **kwargs) -> Path:
        """Record the download request and return a fake source file."""
        self.download = {"url": url, "directory": directory, **kwargs}
        path = directory / "source.mp4"
        path.write_bytes(b"source")
        return path

    def renderer(self, source: Path, clips, directory: Path) -> Path:
        """Record the clip ranges and return a fake cut."""
        self.clips = list(clips)
        path = directory / "summary.mp4"
        path.write_bytes(b"summary")
        return path

    def uploader(self, path: Path) -> str:
        """Record the upload and return a canned URL."""
        self.uploaded = path
        return "https://blob.example/summaries/summary.mp4"


def _line(seconds: int) -> dict:
    """Build one transcript line."""
    return {"seconds": seconds, "text": "one two three four five"}


def _payload(lines: list[dict] | None = None) -> dict:
    """Build an internal summary response."""
    return {
        "summary_id": "summary-1",
        "video_id": "video-1",
        "youtube_video_id": "abcde12345",
        "youtube_url": "https://www.youtube.com/watch?v=abcde12345",
        "title": "Linear Algebra",
        "status": "pending",
        "duration_seconds": 1200,
        "language": "en",
        "transcript_lines": (
            lines if lines is not None else [_line(index * 60) for index in range(5)]
        ),
    }


def _settings() -> Settings:
    """Build isolated worker settings for a summary run."""
    return Settings(_env_file=None, openai_api_key="sk-test")


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the service from reading the process environment."""
    monkeypatch.setattr(summary_service, "get_settings", _settings)


async def _run(payload: dict, ai: FakeAI | None = None, media: FakeMedia | None = None):
    """Run the service with in-memory collaborators."""
    backend = FakeBackend(payload)
    media = media or FakeMedia(Path("/tmp"))
    selected_ai = ai or FakeAI()
    url = await run_summary(
        "summary-1",
        client_factory=lambda settings: backend,
        ai_factory=lambda settings: selected_ai,
        downloader=media.downloader,
        renderer=media.renderer,
        uploader=media.uploader,
    )
    return url, backend, media, selected_ai


class TestSummaryData:
    """The backend response becomes a typed request."""

    def test_parses_transcript_lines(self) -> None:
        request = _summary_data(_payload())

        assert request.summary_id == "summary-1"
        assert request.duration_seconds == 1200
        assert len(request.transcript_lines) == 5
        assert request.transcript_lines[1].seconds == 60


class TestRunSummary:
    """The service scores, cuts, uploads, and reports ready."""

    @pytest.mark.asyncio
    async def test_runs_the_pipeline_end_to_end(self, tmp_path) -> None:
        media = FakeMedia(tmp_path)

        url, backend, media, ai = await _run(_payload(), media=media)

        assert url == "https://blob.example/summaries/summary.mp4"
        assert backend.advanced == ["summary-1"]
        assert backend.ready == [("summary-1", url)]
        assert media.download["url"].endswith("abcde12345")
        assert media.download["height"] == 720
        assert media.download["settings"].summary_height == 720
        assert media.uploaded is not None
        assert media.uploaded.name == "summary.mp4"
        assert len(ai.batches) == 1
        assert [segment.index for segment in ai.batches[0]] == [0, 1, 2, 3, 4]
        assert len(media.clips) == 5

    @pytest.mark.asyncio
    async def test_requests_scoring_in_batches(self, tmp_path) -> None:
        payload = _payload([_line(index * 60) for index in range(45)])
        media = FakeMedia(tmp_path)

        _, _, _, ai = await _run(payload, media=media)

        assert [len(batch) for batch in ai.batches] == [40, 5]

    @pytest.mark.asyncio
    async def test_fails_when_the_transcript_is_empty(self, tmp_path) -> None:
        backend = FakeBackend(_payload([]))
        media = FakeMedia(tmp_path)
        ai = FakeAI()

        with pytest.raises(ExternalServiceError) as caught:
            await run_summary(
                "summary-1",
                client_factory=lambda settings: backend,
                ai_factory=lambda settings: ai,
                downloader=media.downloader,
                renderer=media.renderer,
                uploader=media.uploader,
            )

        assert "no usable lines" in caught.value.message
        assert backend.advanced == ["summary-1"]
        assert backend.ready == []
        assert ai.batches == []

    @pytest.mark.asyncio
    async def test_fails_loudly_when_scoring_is_not_configured(self, tmp_path) -> None:
        backend = FakeBackend(_payload())
        media = FakeMedia(tmp_path)

        with pytest.raises(ExternalServiceError) as caught:
            await run_summary(
                "summary-1",
                client_factory=lambda settings: backend,
                ai_factory=lambda settings: FakeAI(configured=False),
                downloader=media.downloader,
                renderer=media.renderer,
                uploader=media.uploader,
            )

        assert "not configured" in caught.value.message
        assert media.download == {}
        assert media.clips == []


class TestBatches:
    """Prompt batches respect size and item limits."""

    def test_splits_on_character_budget(self) -> None:
        segments = [
            SummarySegment(index=index, start_seconds=0, end_seconds=1, text="x" * 7000)
            for index in range(3)
        ]

        assert [len(batch) for batch in _batches(segments)] == [1, 1, 1]

    def test_splits_on_item_count(self) -> None:
        segments = [
            SummarySegment(index=index, start_seconds=0, end_seconds=1, text="x")
            for index in range(41)
        ]

        assert [len(batch) for batch in _batches(segments)] == [40, 1]


class TestFailSummary:
    """Failure reporting is best effort."""

    @pytest.mark.asyncio
    async def test_reports_the_failure(self) -> None:
        backend = FakeBackend(_payload())

        await fail_summary("summary-1", "failed", client_factory=lambda settings: backend)

        assert backend.failed == [("summary-1", "failed")]

    @pytest.mark.asyncio
    async def test_swallows_backend_errors(self) -> None:
        class BrokenBackend(FakeBackend):
            async def mark_summary_failed(self, summary_id: str, error: str) -> None:
                raise RuntimeError("backend down")

        await fail_summary(
            "summary-1", "failed", client_factory=lambda settings: BrokenBackend(_payload())
        )
