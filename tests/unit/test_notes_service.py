"""Unit tests for the lecture-notes orchestration service."""

import pytest

from app.core.config import Settings
from app.core.exceptions import ExternalServiceError
from app.integrations.equation_review import RECONSTRUCTED_MARKER, EquationReview
from app.integrations.openai import OpenAINotesAI
from app.models import NoteBlock, NotesLineData
from app.services import notes as notes_service
from app.services.notes import (
    _chunks,
    _default_ai,
    _fallback_blocks,
    _notes_data,
    _notes_title,
    _renumber_topics,
    _without_model_title,
    fail_notes,
    is_quota_error,
    run_notes,
)


class FakeBackend:
    """In-memory backend client for one notes request."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.advanced: list[str] = []
        self.ready: list[tuple[str, str]] = []
        self.failed: list[tuple[str, str]] = []

    async def get_video_notes(self, note_id: str) -> dict:
        """Return the canned request."""
        return self.payload

    async def advance_notes(self, note_id: str) -> None:
        """Record the processing callback."""
        self.advanced.append(note_id)

    async def mark_notes_ready(self, note_id: str, blob_url: str) -> None:
        """Record the ready callback."""
        self.ready.append((note_id, blob_url))

    async def mark_notes_failed(self, note_id: str, error: str) -> None:
        """Record the failure callback."""
        self.failed.append((note_id, error))


class FakeAI:
    """Scripted transcript-cleanup client."""

    def __init__(
        self,
        *,
        configured: bool = True,
        blocks: list[NoteBlock] | None = None,
        reviews: list[EquationReview] | None = None,
        fail_cleanup: bool = False,
        fail_quota: bool = False,
        fail_review: bool = False,
        overview: str = "",
        fail_overview: bool = False,
    ) -> None:
        self.configured = configured
        self.blocks = blocks or [NoteBlock(kind="paragraph", text="cleaned")]
        self.reviews = reviews
        self.fail_cleanup = fail_cleanup
        self.fail_quota = fail_quota
        self.fail_review = fail_review
        self.overview_text = overview
        self.fail_overview = fail_overview
        self.cleaned = 0
        self.reviewed = 0

    async def overview(self, **kwargs) -> str:
        """Return the scripted overview or fail it."""
        if self.fail_overview:
            raise ExternalServiceError("overview down", service="openai")
        return self.overview_text

    async def clean_chunk(self, **kwargs) -> list[NoteBlock]:
        """Return the scripted blocks or fail one window."""
        self.cleaned += 1
        if self.fail_quota:
            raise ExternalServiceError(
                "quota",
                service="openai",
                details={"quota_exhausted": True},
            )
        if self.fail_cleanup:
            raise ExternalServiceError("cleanup down", service="openai")
        return self.blocks

    async def review_equations(
        self, transcript: str, equations: list[NoteBlock]
    ) -> list[EquationReview]:
        """Return the scripted verdicts or accept every equation."""
        self.reviewed += 1
        if self.fail_review:
            raise ExternalServiceError("review down", service="openai")
        if self.reviews is not None:
            return self.reviews
        return [EquationReview(index=index, verdict="supported") for index in range(len(equations))]


class FakeRenderer:
    """Capture the blocks and notice passed to the renderer."""

    def __init__(self) -> None:
        self.title = ""
        self.blocks: list[NoteBlock] = []
        self.overview = ""
        self.notice = ""
        self.is_rtl = False

    def render(self, request, content, notice="") -> bytes:
        """Record the render input."""
        self.title = content.title
        self.blocks = list(content.blocks)
        self.overview = content.overview
        self.notice = notice
        self.is_rtl = request.is_rtl
        return b"%PDF-fake"


def _payload(lines: list[dict] | None = None) -> dict:
    """Build an internal notes response."""
    return {
        "note_id": "note-1",
        "video_id": "video-1",
        "youtube_video_id": "abcde12345",
        "youtube_url": "https://www.youtube.com/watch?v=abcde12345",
        "title": "Linear Algebra",
        "status": "pending",
        "duration_seconds": 1200,
        "language": "en",
        "is_rtl": False,
        "transcript_lines": lines
        or [
            {"seconds": 0.0, "text": "Welcome"},
            {"seconds": 500.0, "text": "Next"},
            {"seconds": 650.0, "text": "Third"},
        ],
    }


class TestProviderSelection:
    """OpenAI is the only notes AI provider."""

    @pytest.mark.asyncio
    async def test_uses_openai_when_configured(self) -> None:
        settings = Settings(_env_file=None, openai_api_key="sk-test")

        assert isinstance(_default_ai(settings), OpenAINotesAI)

    @pytest.mark.asyncio
    async def test_without_a_key_openai_is_unconfigured(self) -> None:
        settings = Settings(_env_file=None, openai_api_key="")

        ai = _default_ai(settings)

        assert isinstance(ai, OpenAINotesAI)
        assert not ai.configured


class TestRunNotes:
    """The end-to-end service queues, generates, uploads, and reports."""

    @pytest.mark.asyncio
    async def test_generates_and_reports_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = FakeBackend(_payload())
        ai = FakeAI()
        renderer = FakeRenderer()
        monkeypatch.setattr(
            notes_service, "upload_pdf", lambda document, name: "https://blob/notes.pdf"
        )

        url = await run_notes(
            "note-1",
            client_factory=lambda settings: backend,
            ai_factory=lambda settings: ai,
            renderer_factory=lambda: renderer,
        )

        assert url == "https://blob/notes.pdf"
        assert backend.advanced == ["note-1"]
        assert backend.ready == [("note-1", "https://blob/notes.pdf")]
        assert renderer.title == "Summary of Linear Algebra"
        assert renderer.blocks[0].text == "cleaned"
        assert "AI-generated summary" in renderer.notice
        assert ai.cleaned >= 1
        assert ai.reviewed == 0

    @pytest.mark.asyncio
    async def test_uses_arabic_notice_for_rtl_notes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = _payload()
        payload.update(title="محاضرة", language="ar", is_rtl=True)
        backend = FakeBackend(payload)
        renderer = FakeRenderer()
        monkeypatch.setattr(
            notes_service, "upload_pdf", lambda document, name: "https://blob/notes.pdf"
        )

        await run_notes(
            "note-1",
            client_factory=lambda settings: backend,
            ai_factory=lambda settings: FakeAI(),
            renderer_factory=lambda: renderer,
        )

        assert renderer.is_rtl is True
        assert renderer.title == "ملخص محاضرة"
        assert "ملخص من إنشاء الذكاء الاصطناعي" in renderer.notice
        assert "AI-generated summary" not in renderer.notice

    @pytest.mark.asyncio
    async def test_uses_transcript_only_without_an_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = FakeBackend(_payload())
        ai = FakeAI(configured=False)
        renderer = FakeRenderer()
        monkeypatch.setattr(
            notes_service, "upload_pdf", lambda document, name: "https://blob/notes.pdf"
        )

        await run_notes(
            "note-1",
            client_factory=lambda settings: backend,
            ai_factory=lambda settings: ai,
            renderer_factory=lambda: renderer,
        )

        assert ai.cleaned == 0
        assert renderer.title == "Linear Algebra"
        assert "Transcript only" in renderer.notice
        assert [block.kind for block in renderer.blocks] == ["paragraph"]

    @pytest.mark.asyncio
    async def test_falls_back_to_transcript_when_cleanup_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = FakeBackend(_payload())
        renderer = FakeRenderer()
        monkeypatch.setattr(
            notes_service, "upload_pdf", lambda document, name: "https://blob/notes.pdf"
        )

        await run_notes(
            "note-1",
            client_factory=lambda settings: backend,
            ai_factory=lambda settings: FakeAI(fail_cleanup=True),
            renderer_factory=lambda: renderer,
        )

        assert renderer.title == "Linear Algebra"
        assert "Welcome" in renderer.blocks[0].text
        assert "Transcript only" in renderer.notice
        assert backend.ready

    @pytest.mark.asyncio
    async def test_renders_equations_generated_from_transcript(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = FakeBackend(_payload())
        ai = FakeAI(
            blocks=[
                NoteBlock(kind="paragraph", text="The PMF is:"),
                NoteBlock(kind="equation", latex="p^x q^{1-x}"),
            ]
        )
        renderer = FakeRenderer()
        monkeypatch.setattr(
            notes_service, "upload_pdf", lambda document, name: "https://blob/notes.pdf"
        )

        await run_notes(
            "note-1",
            client_factory=lambda settings: backend,
            ai_factory=lambda settings: ai,
            renderer_factory=lambda: renderer,
        )

        assert ai.cleaned >= 1
        assert any(block.kind == "equation" for block in renderer.blocks)
        assert renderer.title == "Summary of Linear Algebra"
        assert "AI-generated summary" in renderer.notice

    @pytest.mark.asyncio
    async def test_summarizes_a_lecture_without_equations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = FakeBackend(
            _payload([{"seconds": 0.0, "text": "Manage your time by prioritizing."}])
        )
        ai = FakeAI(
            blocks=[
                NoteBlock(kind="heading", text="Prioritization"),
                NoteBlock(kind="paragraph", text="Urgent work is not always important."),
                NoteBlock(kind="bullet", text="Schedule deep work first."),
            ],
            overview="A lecture about managing time and priorities.",
        )
        renderer = FakeRenderer()
        monkeypatch.setattr(
            notes_service, "upload_pdf", lambda document, name: "https://blob/notes.pdf"
        )

        await run_notes(
            "note-1",
            client_factory=lambda settings: backend,
            ai_factory=lambda settings: ai,
            renderer_factory=lambda: renderer,
        )

        assert renderer.title == "Summary of Linear Algebra"
        assert renderer.overview == "A lecture about managing time and priorities."
        assert not any(block.kind == "equation" for block in renderer.blocks)
        assert [block.kind for block in renderer.blocks] == ["heading", "paragraph", "bullet"]
        assert ai.reviewed == 0

    @pytest.mark.asyncio
    async def test_applies_a_reviewer_correction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = FakeBackend(_payload())
        ai = FakeAI(
            blocks=[NoteBlock(kind="equation", latex="p^x q")],
            reviews=[EquationReview(index=0, verdict="corrected", corrected_latex="p^{x} q^{1-x}")],
        )
        renderer = FakeRenderer()
        monkeypatch.setattr(
            notes_service, "upload_pdf", lambda document, name: "https://blob/notes.pdf"
        )

        await run_notes(
            "note-1",
            client_factory=lambda settings: backend,
            ai_factory=lambda settings: ai,
            renderer_factory=lambda: renderer,
        )

        equation = next(block for block in renderer.blocks if block.kind == "equation")
        assert equation.latex == "p^{x} q^{1-x}"
        assert ai.reviewed >= 1
        assert not any(block.text == RECONSTRUCTED_MARKER for block in renderer.blocks)

    @pytest.mark.asyncio
    async def test_keeps_an_equation_the_reviewer_rejects(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = FakeBackend(_payload())
        ai = FakeAI(
            blocks=[
                NoteBlock(kind="paragraph", text="Formula:"),
                NoteBlock(kind="equation", latex="P(X=k)"),
            ],
            reviews=[EquationReview(index=0, verdict="unsupported")],
        )
        renderer = FakeRenderer()
        monkeypatch.setattr(
            notes_service, "upload_pdf", lambda document, name: "https://blob/notes.pdf"
        )

        await run_notes(
            "note-1",
            client_factory=lambda settings: backend,
            ai_factory=lambda settings: ai,
            renderer_factory=lambda: renderer,
        )

        equation = next(block for block in renderer.blocks if block.kind == "equation")
        assert equation.latex == "P(X=k)"
        assert any(block.text == RECONSTRUCTED_MARKER for block in renderer.blocks)

    @pytest.mark.asyncio
    async def test_keeps_an_uncertain_equation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = FakeBackend(_payload())
        ai = FakeAI(
            blocks=[NoteBlock(kind="equation", latex="x^2")],
            reviews=[EquationReview(index=0, verdict="uncertain")],
        )
        renderer = FakeRenderer()
        monkeypatch.setattr(
            notes_service, "upload_pdf", lambda document, name: "https://blob/notes.pdf"
        )

        await run_notes(
            "note-1",
            client_factory=lambda settings: backend,
            ai_factory=lambda settings: ai,
            renderer_factory=lambda: renderer,
        )

        assert any(block.kind == "equation" for block in renderer.blocks)
        assert any(block.text == RECONSTRUCTED_MARKER for block in renderer.blocks)

    @pytest.mark.asyncio
    async def test_keeps_equations_when_the_review_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = FakeBackend(_payload())
        ai = FakeAI(
            blocks=[NoteBlock(kind="equation", latex="x^2")],
            fail_review=True,
        )
        renderer = FakeRenderer()
        monkeypatch.setattr(
            notes_service, "upload_pdf", lambda document, name: "https://blob/notes.pdf"
        )

        await run_notes(
            "note-1",
            client_factory=lambda settings: backend,
            ai_factory=lambda settings: ai,
            renderer_factory=lambda: renderer,
        )

        assert any(block.kind == "equation" for block in renderer.blocks)
        assert not any(block.text == RECONSTRUCTED_MARKER for block in renderer.blocks)
        assert backend.ready

    @pytest.mark.asyncio
    async def test_quota_failure_does_not_upload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = FakeBackend(_payload())
        renderer = FakeRenderer()
        monkeypatch.setattr(
            notes_service,
            "upload_pdf",
            lambda document, name: pytest.fail("should not upload without AI notes"),
        )

        with pytest.raises(ExternalServiceError) as caught:
            await run_notes(
                "note-1",
                client_factory=lambda settings: backend,
                ai_factory=lambda settings: FakeAI(fail_quota=True),
                renderer_factory=lambda: renderer,
            )

        assert is_quota_error(caught.value)
        assert backend.advanced == ["note-1"]
        assert backend.ready == []

    @pytest.mark.asyncio
    async def test_failure_reports_safe_message(self) -> None:
        backend = FakeBackend(_payload())

        async def boom(note_id, message, client_factory=None):
            backend.failed.append((note_id, message))

        await fail_notes("note-1", "safe", client_factory=lambda settings: backend)

        assert backend.failed == [("note-1", "safe")]

    @pytest.mark.asyncio
    async def test_failure_report_never_raises(self) -> None:
        class Broken:
            async def mark_notes_failed(self, note_id, error):
                raise RuntimeError("backend down")

        await fail_notes("note-1", "safe", client_factory=lambda settings: Broken())


class TestHelpers:
    """Chunking and fallback behavior."""

    def test_builds_typed_request(self) -> None:
        request = _notes_data(_payload())

        assert request.note_id == "note-1"
        assert request.transcript_lines[0] == NotesLineData(seconds=0.0, text="Welcome")

    def test_chunks_by_window(self) -> None:
        lines = [
            NotesLineData(seconds=0.0, text="a"),
            NotesLineData(seconds=500.0, text="b"),
            NotesLineData(seconds=650.0, text="c"),
            NotesLineData(seconds=1300.0, text="d"),
        ]

        chunks = list(_chunks(lines, 600))

        assert [chunk[0] for chunk in chunks] == [0, 600, 1200]
        assert [line.text for line in chunks[0][2]] == ["a", "b"]

    def test_empty_transcript_has_no_chunks(self) -> None:
        assert list(_chunks([], 600)) == []

    def test_fallback_groups_lines_without_timestamps(self) -> None:
        blocks = _fallback_blocks(
            [NotesLineData(seconds=0, text="alpha"), NotesLineData(seconds=1, text="beta")]
        )

        assert blocks[0].text == "alpha beta"
        assert "0" not in blocks[0].text

    def test_empty_fallback_has_a_message(self) -> None:
        assert _fallback_blocks([])[0].text == "No transcript is available."

    def test_notes_title_uses_the_lecture_title(self) -> None:
        request = _notes_data(_payload())

        assert _notes_title(request) == "Summary of Linear Algebra"

    def test_notes_title_is_arabic_for_rtl(self) -> None:
        payload = _payload()
        payload.update(title="محاضرة", is_rtl=True)

        assert _notes_title(_notes_data(payload)) == "ملخص محاضرة"

    def test_drops_a_model_supplied_title(self) -> None:
        blocks = [
            NoteBlock(kind="heading", text="Summary of Lecture 2"),
            NoteBlock(kind="heading", text="1. Bernoulli distribution"),
        ]

        assert _without_model_title(blocks) == [blocks[1]]

    def test_renumbers_topics_and_drops_repeated_headings(self) -> None:
        blocks = [
            NoteBlock(kind="heading", text="Bernoulli distribution"),
            NoteBlock(kind="equation", latex="p"),
            NoteBlock(kind="heading", text="1. Bernoulli distribution"),
            NoteBlock(kind="equation", latex="np"),
            NoteBlock(kind="heading", text="Binomial distribution"),
            NoteBlock(kind="heading", text="A. Failures before success"),
        ]

        result = _renumber_topics(blocks)

        assert [block.text for block in result if block.kind == "heading"] == [
            "1. Bernoulli distribution",
            "2. Binomial distribution",
            "A. Failures before success",
        ]


class TestLectureOverview:
    """A brief overview is added above the equations."""

    @pytest.mark.asyncio
    async def test_passes_the_overview_to_the_renderer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        renderer = FakeRenderer()
        monkeypatch.setattr(
            notes_service, "upload_pdf", lambda document, name: "https://blob/notes.pdf"
        )

        await run_notes(
            "note-1",
            client_factory=lambda settings: FakeBackend(_payload()),
            ai_factory=lambda settings: FakeAI(overview="ملخص المحاضرة"),
            renderer_factory=lambda: renderer,
        )

        assert renderer.overview == "ملخص المحاضرة"
        assert renderer.blocks

    @pytest.mark.asyncio
    async def test_fails_open_when_the_overview_call_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        renderer = FakeRenderer()
        monkeypatch.setattr(
            notes_service, "upload_pdf", lambda document, name: "https://blob/notes.pdf"
        )

        url = await run_notes(
            "note-1",
            client_factory=lambda settings: FakeBackend(_payload()),
            ai_factory=lambda settings: FakeAI(fail_overview=True),
            renderer_factory=lambda: renderer,
        )

        assert url == "https://blob/notes.pdf"
        assert renderer.overview == ""
        assert renderer.blocks
