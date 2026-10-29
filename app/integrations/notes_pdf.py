"""PDF renderer for structured lecture notes using ReportLab."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import TYPE_CHECKING

import arabic_reshaper
import matplotlib

from app.core.logging import get_logger

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer

if TYPE_CHECKING:
    from app.models import NoteBlock, NotesData
    from app.services.notes import _NotesContent

logger = get_logger(__name__)

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
ARABIC_FONT_PATH = FONT_DIR / "Amiri-Regular.ttf"
FONT_NAME = "Amiri"

try:
    if ARABIC_FONT_PATH.exists():
        pdfmetrics.registerFont(TTFont(FONT_NAME, str(ARABIC_FONT_PATH)))
    else:
        logger.warning("Arabic font not found", path=str(ARABIC_FONT_PATH))
        FONT_NAME = "Helvetica"
except Exception:
    logger.warning("Could not register Arabic font", exc_info=True)
    FONT_NAME = "Helvetica"


def _shape_arabic(text: str) -> str:
    if not text:
        return ""
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text


def _escape_markup(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _arabic_paragraph(text: str) -> str:
    shaped = _shape_arabic(text)
    escaped = _escape_markup(shaped)
    return escaped.replace("\n", "<br/>")


def _rtl_paragraph(text: str, style: ParagraphStyle) -> str:
    """Wrap logical-order text, then shape each line.

    ReportLab wraps the string it is given. Shaping a whole paragraph first and
    letting ReportLab wrap the visual-order result reverses multi-line
    paragraphs, so lines must be measured and shaped one at a time.
    """
    available = (A4[0] - 4 * cm - style.leftIndent) * 0.97
    rendered: list[str] = []
    for logical_line in text.split("\n"):
        words = logical_line.split()
        if not words:
            continue
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word])
            width = pdfmetrics.stringWidth(_shape_arabic(candidate), style.fontName, style.fontSize)
            if current and width > available:
                rendered.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            rendered.append(" ".join(current))
    shaped = [_escape_markup(_shape_arabic(line)) for line in rendered]
    return "<br/>".join(shaped)


def _paragraph(text: str, style: ParagraphStyle, rtl: bool) -> str:
    return _rtl_paragraph(text, style) if rtl else _arabic_paragraph(text)


def _normalize_latex(latex: str) -> str:
    latex = str(latex).strip()
    latex = re.sub(r"^\$\$(.*?)\$\$$", r"\1", latex, flags=re.S)
    latex = re.sub(r"^\\\[(.*?)\\\]$", r"\1", latex, flags=re.S)
    latex = re.sub(r"^\\\((.*?)\\\)$", r"\1", latex, flags=re.S)
    latex = latex.strip()
    if len(latex) >= 2 and latex.startswith("$") and latex.endswith("$"):
        latex = latex[1:-1].strip()
    return latex


def _equation_to_image(latex: str) -> io.BytesIO:
    latex = _normalize_latex(latex)
    fig = plt.figure(figsize=(10, 1.2))
    fig.patch.set_alpha(0)
    try:
        fig.text(0.5, 0.5, f"${latex}$", fontsize=17, ha="center", va="center")
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=200,
            bbox_inches="tight",
            transparent=True,
            pad_inches=0.18,
        )
        buffer.seek(0)
        plt.close(fig)
        return buffer
    except Exception as exc:
        plt.close(fig)
        logger.warning("Could not render equation", error=str(exc))
        fallback_fig = plt.figure(figsize=(10, 1.5))
        fallback_fig.patch.set_alpha(0)
        fallback_fig.text(0.02, 0.5, latex[:300], fontsize=11, ha="left", va="center", wrap=True)
        buffer = io.BytesIO()
        fallback_fig.savefig(
            buffer,
            format="png",
            dpi=180,
            bbox_inches="tight",
            transparent=True,
            pad_inches=0.18,
        )
        buffer.seek(0)
        plt.close(fallback_fig)
        return buffer


def _create_styles(rtl: bool = False):
    styles = getSampleStyleSheet()
    body_alignment = TA_RIGHT if rtl else TA_LEFT
    for name in [
        "NotesTitle",
        "NotesNotice",
        "NotesSection",
        "NotesParagraph",
        "NotesBullet",
        "NotesEquationNum",
        "NotesDescription",
        "NotesNote",
    ]:
        if name in styles:
            del styles[name]
    styles.add(
        ParagraphStyle(
            name="NotesTitle",
            fontName=FONT_NAME,
            fontSize=20,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17365D"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="NotesNotice",
            fontName=FONT_NAME,
            fontSize=9,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#666666"),
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="NotesSection",
            fontName=FONT_NAME,
            fontSize=14,
            leading=24,
            alignment=body_alignment,
            textColor=colors.HexColor("#17365D"),
            spaceBefore=4,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="NotesParagraph",
            fontName=FONT_NAME,
            fontSize=11.5,
            leading=19,
            alignment=body_alignment,
            textColor=colors.HexColor("#333333"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="NotesBullet",
            fontName=FONT_NAME,
            fontSize=11.5,
            leading=19,
            alignment=body_alignment,
            textColor=colors.HexColor("#333333"),
            leftIndent=0.6 * cm,
            bulletIndent=0.2 * cm,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="NotesEquationNum",
            fontName=FONT_NAME,
            fontSize=12,
            leading=20,
            alignment=body_alignment,
            textColor=colors.HexColor("#17365D"),
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="NotesDescription",
            fontName=FONT_NAME,
            fontSize=11.5,
            leading=19,
            alignment=body_alignment,
            textColor=colors.HexColor("#333333"),
            spaceBefore=12,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="NotesNote",
            fontName=FONT_NAME,
            fontSize=9.5,
            leading=16,
            alignment=body_alignment,
            textColor=colors.HexColor("#666666"),
            spaceBefore=0,
            spaceAfter=6,
        )
    )
    return styles


def _overview_lines(overview: str) -> list[tuple[bool, str]]:
    """Group overview text into paragraphs and bullet items.

    Blank lines separate paragraphs; a single newline inside a paragraph is
    treated as a space, because models wrap prose arbitrarily.
    """
    items: list[tuple[bool, str]] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            items.append((False, " ".join(paragraph)))
            paragraph.clear()

    for raw_line in (overview or "").replace("\r\n", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line[0] in "•-*":
            flush()
            items.append((True, line.lstrip("•-* ").strip()))
        else:
            paragraph.append(line)
    flush()
    return items


def _append_overview(story, styles, overview: str, rtl: bool) -> None:
    """Render the brief lecture overview above the detailed notes."""
    items = _overview_lines(overview)
    if not items:
        return
    heading = "نظرة عامة" if rtl else "Overview"
    section_style = styles["NotesSection"]
    story.append(Paragraph(_paragraph(heading, section_style, rtl), section_style))
    for is_bullet, text in items:
        name = "NotesBullet" if is_bullet else "NotesParagraph"
        style = styles[name]
        body = f"• {text}" if is_bullet else text
        story.append(Paragraph(_paragraph(body, style, rtl), style))
    story.append(Spacer(1, 0.4 * cm))


def _append_blocks(story, styles, blocks: list[NoteBlock], rtl: bool) -> None:
    """Render headings, prose, bullets, and equations in model order."""
    equation_number = 0
    for block in blocks:
        if block.kind == "heading":
            style = styles["NotesSection"]
            story.append(Paragraph(_paragraph(block.text, style, rtl), style))
            continue
        if block.kind == "bullet":
            style = styles["NotesBullet"]
            story.append(Paragraph(_paragraph(f"• {block.text}", style, rtl), style))
            continue
        if block.kind != "equation":
            style = styles["NotesParagraph"]
            story.append(Paragraph(_paragraph(block.text, style, rtl), style))
            continue

        equation_number += 1
        block_items = []
        description = block.text.strip()
        label = (
            f"{equation_number}. {description}"
            if description
            else f"{equation_number}. {'معادلة' if rtl else 'Equation'}"
        )
        description_style = styles["NotesDescription"]
        block_items.append(Paragraph(_paragraph(label, description_style, rtl), description_style))
        note = block.note.strip()
        if note:
            note_style = styles["NotesNote"]
            block_items.append(Paragraph(_paragraph(note, note_style, rtl), note_style))
        try:
            equation_image = _equation_to_image(block.latex)
            image = Image(equation_image)
            max_width = A4[0] - 4 * cm
            max_height = 5 * cm
            if image.imageWidth and image.imageHeight:
                scale = min(max_width / image.imageWidth, max_height / image.imageHeight, 1.0)
                image.drawWidth = image.imageWidth * scale
                image.drawHeight = image.imageHeight * scale
            image.hAlign = "CENTER"
            block_items.append(image)
        except Exception:
            logger.warning("Failed to render equation image", exc_info=True)
        block_items.append(Spacer(1, 0.5 * cm))
        story.append(KeepTogether(block_items))


def render_notes_pdf(request: NotesData, content: _NotesContent, notice: str = "") -> bytes:
    rtl = bool(getattr(request, "is_rtl", False))
    styles = _create_styles(rtl)
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title=content.title,
    )
    story = []

    title_style = styles["NotesTitle"]
    story.append(Paragraph(_paragraph(content.title, title_style, rtl), title_style))
    if notice:
        notice_style = styles["NotesNotice"]
        story.append(Paragraph(_paragraph(notice, notice_style, rtl), notice_style))
    story.append(Spacer(1, 0.3 * cm))
    _append_overview(story, styles, content.overview, rtl)

    if not content.blocks:
        if not content.overview:
            empty_style = styles["NotesNotice"]
            message = (
                "لا يوجد محتوى متاح لهذه المحاضرة."
                if rtl
                else "No content is available for this lecture."
            )
            story.append(Paragraph(_paragraph(message, empty_style, rtl), empty_style))
        document.build(story)
        return buffer.getvalue()

    _append_blocks(story, styles, content.blocks, rtl)
    document.build(story)
    return buffer.getvalue()


class LectureNotesPdfRenderer:
    """Renderer used by the notes service."""

    def render(self, request: NotesData, content: _NotesContent, notice: str = "") -> bytes:
        return render_notes_pdf(request, content, notice)
