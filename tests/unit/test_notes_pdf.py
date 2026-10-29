"""Unit tests for the lecture-notes PDF renderer."""

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics

from app.integrations.notes_pdf import (
    LectureNotesPdfRenderer,
    _arabic_paragraph,
    _create_styles,
    _equation_to_image,
    _normalize_latex,
    _overview_lines,
    _rtl_paragraph,
    render_notes_pdf,
)
from app.models import NoteBlock
from app.services.notes import _NotesContent


@pytest.fixture
def content() -> _NotesContent:
    return _NotesContent(
        title="ملخص المحاضرة",
        blocks=[
            NoteBlock(kind="equation", latex=r"P(X=k)=\binom{n}{k}p^k (1-p)^{n-k}"),
            NoteBlock(kind="equation", latex=r"E[X]=np"),
        ],
        extracted=True,
    )


class TestArabicParagraph:
    """Text is shaped for ReportLab's LTR drawing model."""

    def test_escapes_markup_characters(self) -> None:
        assert "&lt;" in _arabic_paragraph("a < b")
        assert "&gt;" in _arabic_paragraph("a > b")
        assert "&amp;" in _arabic_paragraph("a & b")

    def test_turns_newlines_into_breaks(self) -> None:
        assert "<br/>" in _arabic_paragraph("line one\nline two")

    def test_returns_empty_string_for_empty_input(self) -> None:
        assert _arabic_paragraph("") == ""


class TestLatexNormalization:
    """Provider LaTeX is unwrapped before mathtext sees it."""

    def test_strips_display_dollars(self) -> None:
        assert _normalize_latex("$$x^2$$") == "x^2"

    def test_strips_bracket_delimiters(self) -> None:
        assert _normalize_latex(r"\[x^2\]") == "x^2"

    def test_strips_inline_paren_delimiters(self) -> None:
        assert _normalize_latex(r"\(x^2\)") == "x^2"

    def test_strips_single_dollars(self) -> None:
        assert _normalize_latex("$x^2$") == "x^2"

    def test_leaves_plain_latex(self) -> None:
        assert _normalize_latex("x^2") == "x^2"


class TestEquationImage:
    """An equation is rasterised to an in-memory PNG."""

    def test_renders_png_bytes(self) -> None:
        image = _equation_to_image(r"\frac{a}{b}")
        data = image.getvalue()

        assert data.startswith(b"\x89PNG")
        assert len(data) > 100

    def test_invalid_latex_falls_back_to_text_image(self) -> None:
        image = _equation_to_image(r"\thisCommandDoesNotExist{x}")
        data = image.getvalue()

        assert data.startswith(b"\x89PNG")


class TestRenderNotesPdf:
    """The renderer produces a structured notes document."""

    def test_renders_a_pdf_when_equations_exist(self, content: _NotesContent) -> None:
        document = render_notes_pdf(object(), content, notice="AI-cleaned")

        assert document.startswith(b"%PDF")
        assert len(document) > 1000

    def test_renders_a_pdf_when_no_equations_exist(self) -> None:
        content = _NotesContent(title="محاضرة", blocks=[], extracted=False)

        document = render_notes_pdf(object(), content)

        assert document.startswith(b"%PDF")

    def test_renders_non_equation_blocks(self) -> None:
        content = _NotesContent(
            title="Summary of Time Management",
            blocks=[
                NoteBlock(kind="heading", text="Prioritization"),
                NoteBlock(kind="paragraph", text="Urgent work is not always important."),
                NoteBlock(kind="bullet", text="Schedule deep work first."),
            ],
            extracted=True,
        )
        document = render_notes_pdf(object(), content)

        assert document.startswith(b"%PDF")
        assert len(document) > 1000


class TestLectureNotesPdfRenderer:
    """The service-facing renderer delegates to the module function."""

    def test_renders_to_pdf(self, content: _NotesContent) -> None:
        document = LectureNotesPdfRenderer().render(object(), content)

        assert document.startswith(b"%PDF")
        assert len(document) > 1000


class TestOverviewLines:
    """Overview prose is grouped into paragraphs and bullets."""

    def test_empty_overview_yields_nothing(self) -> None:
        assert _overview_lines("   ") == []

    def test_single_newlines_join_into_one_paragraph(self) -> None:
        assert _overview_lines("first line\nsecond line") == [(False, "first line second line")]

    def test_blank_lines_separate_paragraphs(self) -> None:
        assert _overview_lines("one\n\ntwo") == [(False, "one"), (False, "two")]

    def test_bullets_are_split_and_trimmed(self) -> None:
        assert _overview_lines("intro\n• first\n- second\n* third") == [
            (False, "intro"),
            (True, "first"),
            (True, "second"),
            (True, "third"),
        ]


class TestOverviewRendering:
    """The overview is drawn above the detailed notes."""

    def test_renders_a_pdf_with_an_overview(self) -> None:
        content = _NotesContent(
            title="Lecture",
            blocks=[NoteBlock(kind="equation", latex="x^2")],
            overview="This lecture covers limits.",
            extracted=True,
        )

        pdf = render_notes_pdf(object(), content)

        assert pdf.startswith(b"%PDF")
        assert len(pdf) > 1000


class TestRtlWrapping:
    """RTL paragraphs are wrapped in logical order, then shaped per line."""

    def test_long_text_keeps_logical_line_order(self) -> None:
        style = _create_styles(rtl=True)["NotesParagraph"]
        text = " ".join(["word"] * 200)

        lines = _rtl_paragraph(text, style).split("<br/>")

        assert len(lines) > 1
        assert lines[0].startswith("word")
        assert text.startswith(lines[0].split()[0])

    def test_each_rendered_line_fits_the_available_width(self) -> None:
        style = _create_styles(rtl=True)["NotesParagraph"]
        text = " ".join(["كلمة"] * 120)

        lines = _rtl_paragraph(text, style).split("<br/>")

        for line in lines:
            assert pdfmetrics.stringWidth(line, style.fontName, style.fontSize) <= (A4[0] - 4 * cm)
