"""Generate an AI-written lecture-notes PDF from a video's transcript."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from app.client import BackendClient
from app.core.config import Settings, get_settings
from app.core.exceptions import DomainError, ExternalServiceError
from app.core.logging import get_logger
from app.integrations.blob import upload_pdf
from app.integrations.equation_review import apply_reviews
from app.integrations.notes_ai import NotesAI
from app.integrations.notes_pdf import LectureNotesPdfRenderer
from app.integrations.openai import OpenAINotesAI
from app.models import NoteBlock, NotesData, NotesLineData
from app.services.jobs import default_client_factory

logger = get_logger(__name__)

FAILURE_SAFE_MESSAGE = "Could not generate the lecture notes. Please try again later."
NOTES_QUOTA_MESSAGE = "AI notes are temporarily unavailable. Please try again later."
_TRANSCRIPT_ONLY_NOTICE_EN = (
    "Transcript only. AI notes are not configured yet, so structured notes and "
    "equations are unavailable."
)
_TRANSCRIPT_ONLY_NOTICE_AR = (
    "النص التفريغي فقط. لم يتم إعداد ملاحظات الذكاء الاصطناعي بعد، لذا الملاحظات "
    "المنظمة والمعادلات غير متاحة."
)
_AI_NOTICE_EN = "AI-generated summary. Verify against the lecture."
_AI_NOTICE_AR = "ملخص من إنشاء الذكاء الاصطناعي. تحقّق منه بمطابقته مع المحاضرة."
_NOTES_TITLE_EN = "Summary"
_NOTES_TITLE_AR = "ملخص"
_TITLE_HEADING_PREFIXES = ("summary", "lecture notes", "ملخص", "ملاحظات")
_TOPIC_NUMBER = re.compile(r"^\d+\.\s+")
_OPTION_HEADING = re.compile(r"^[^\d\s]\.\s+")
_FALLBACK_PARAGRAPH_CHARS = 700
_MAX_TRANSCRIPT_CHARS = 12000
_MAX_OVERVIEW_CHARS = 12000

AIFactory = Callable[[Settings], NotesAI]


@dataclass(frozen=True)
class _NotesContent:
    """The document title, its blocks, and whether AI extraction ran."""

    title: str
    blocks: list[NoteBlock]
    extracted: bool
    overview: str = ""


def _default_ai(settings: Settings) -> NotesAI:
    """Use OpenAI for transcript cleanup; an empty key disables AI notes."""
    return OpenAINotesAI.from_settings(settings)


def _format_timestamp(seconds: float) -> str:
    """Format a position as HH:MM:SS for transcript alignment."""
    total = max(int(seconds), 0)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def is_quota_error(exc: Exception) -> bool:
    """Return True when the AI failed because its quota is exhausted."""
    return isinstance(exc, ExternalServiceError) and bool(exc.details.get("quota_exhausted"))


def _notes_data(payload: dict) -> NotesData:
    """Build the typed request from the backend's internal response."""
    lines = [NotesLineData(**line) for line in payload.get("transcript_lines", [])]
    return NotesData(
        note_id=payload["note_id"],
        video_id=payload["video_id"],
        youtube_video_id=payload["youtube_video_id"],
        youtube_url=payload["youtube_url"],
        title=payload["title"],
        status=payload["status"],
        duration_seconds=payload.get("duration_seconds"),
        language=payload.get("language"),
        is_rtl=bool(payload.get("is_rtl")),
        transcript_lines=lines,
    )


async def run_notes(
    note_id: str,
    client_factory: Callable[..., BackendClient] | None = None,
    ai_factory: AIFactory | None = None,
    renderer_factory: Callable[[], LectureNotesPdfRenderer] | None = None,
) -> str:
    """Generate one lecture-notes PDF end to end and report its blob URL."""
    settings = get_settings()
    client = (client_factory or default_client_factory)(settings)
    request = _notes_data(await client.get_video_notes(note_id))
    await client.advance_notes(note_id)
    blob_url = await _generate_document(request, settings, ai_factory, renderer_factory)
    await client.mark_notes_ready(note_id, blob_url)
    return blob_url


async def _generate_document(
    request: NotesData,
    settings: Settings,
    ai_factory: AIFactory | None,
    renderer_factory: Callable[[], LectureNotesPdfRenderer] | None,
) -> str:
    """Build the PDF and upload it, returning its blob URL."""
    ai = (ai_factory or _default_ai)(settings)
    content = await _blocks_for(request, settings, ai)
    document = await asyncio.to_thread(
        _render,
        request,
        content,
        _notice(content.extracted, request.is_rtl),
        renderer_factory,
    )
    return await asyncio.to_thread(upload_pdf, document, name=f"{request.note_id}.pdf")


async def fail_notes(
    note_id: str,
    message: str,
    client_factory: Callable[..., BackendClient] | None = None,
) -> None:
    """Report a failed lecture-notes request, best effort."""
    settings = get_settings()
    client = (client_factory or default_client_factory)(settings)
    try:
        await client.mark_notes_failed(note_id, message)
    except Exception:
        logger.exception("Could not report lecture-notes failure", note_id=note_id)


async def _blocks_for(
    request: NotesData,
    settings: Settings,
    ai: NotesAI,
) -> _NotesContent:
    """Build the structured summary for the requested lecture."""
    if not ai.configured:
        return _fallback_content(request)
    blocks = await _clean_chunks(ai, request, settings.notes_chunk_seconds)
    if not blocks:
        return _fallback_content(request)
    overview = await _overview(ai, request)
    topics = _renumber_topics(_without_model_title(blocks))
    return _NotesContent(
        title=_notes_title(request),
        blocks=topics,
        extracted=True,
        overview=overview,
    )


async def _overview(ai: NotesAI, request: NotesData) -> str:
    """Ask for a brief lecture overview, failing open to no overview."""
    transcript = " ".join(line.text.strip() for line in request.transcript_lines)
    transcript = transcript[:_MAX_OVERVIEW_CHARS]
    if not transcript:
        return ""
    try:
        return await ai.overview(
            title=request.title,
            language=request.language,
            transcript=transcript,
        )
    except DomainError as exc:
        logger.warning("Lecture overview failed; continuing without it", error=str(exc))
        return ""


def _fallback_content(request: NotesData) -> _NotesContent:
    """Return the raw transcript under the lecture's own title."""
    return _NotesContent(
        title=request.title,
        blocks=_fallback_blocks(request.transcript_lines),
        extracted=False,
    )


def _notes_title(request: NotesData) -> str:
    """Build the document title from the lecture title and text direction."""
    base = _NOTES_TITLE_AR if request.is_rtl else _NOTES_TITLE_EN
    title = request.title.strip()
    if not title:
        return base
    return f"{base} {title}" if request.is_rtl else f"{base} of {title}"


def _without_model_title(blocks: Sequence[NoteBlock]) -> list[NoteBlock]:
    """Drop a model-supplied title so the system title is the only one."""
    first = blocks[0] if blocks else None
    if (
        first is not None
        and first.kind == "heading"
        and first.text.casefold().startswith(_TITLE_HEADING_PREFIXES)
    ):
        return list(blocks[1:])
    return list(blocks)


def _renumber_topics(blocks: Sequence[NoteBlock]) -> list[NoteBlock]:
    """Number topic headings sequentially and drop repeated chunk headings."""
    result: list[NoteBlock] = []
    topic = ""
    number = 0
    for block in blocks:
        if block.kind != "heading" or _OPTION_HEADING.match(block.text):
            result.append(block)
            continue
        title = _TOPIC_NUMBER.sub("", block.text).strip()
        if not title or title == topic:
            continue
        number += 1
        topic = title
        result.append(NoteBlock(kind="heading", text=f"{number}. {title}"))
    return result


async def _clean_chunks(
    ai: NotesAI,
    request: NotesData,
    chunk_seconds: int,
) -> list[NoteBlock] | None:
    """Clean each window, returning None when every call failed."""
    blocks: list[NoteBlock] = []
    succeeded = False
    for start, end, lines in _chunks(request.transcript_lines, chunk_seconds):
        chunk = await _clean_chunk(ai, request, (start, end, lines))
        if chunk is None:
            blocks.extend(_dedupe_headings(blocks, _fallback_blocks(lines)))
        else:
            succeeded = True
            blocks.extend(_dedupe_headings(blocks, chunk))
    return blocks if succeeded else None


def _notice(extracted: bool, is_rtl: bool) -> str:
    """Choose the PDF notice in the document's language."""
    if extracted:
        return _AI_NOTICE_AR if is_rtl else _AI_NOTICE_EN
    return _TRANSCRIPT_ONLY_NOTICE_AR if is_rtl else _TRANSCRIPT_ONLY_NOTICE_EN


def _transcript_text(lines: Sequence[NotesLineData]) -> str:
    """Render one transcript window without exposing it past the prompt cap."""
    return "\n".join(f"[{_format_timestamp(line.seconds)}] {line.text}" for line in lines)[
        :_MAX_TRANSCRIPT_CHARS
    ]


def _log_cleanup_failure(start: int) -> None:
    """Record a failed cleanup call before using the raw transcript."""
    logger.warning("Lecture-notes cleanup failed; using transcript", start=start)


async def _clean_chunk(
    ai: NotesAI,
    request: NotesData,
    window: tuple[int, int, Sequence[NotesLineData]],
) -> list[NoteBlock] | None:
    """Clean one transcript window, returning None if the call fails."""
    start, end, lines = window
    transcript = _transcript_text(lines)
    try:
        blocks = await ai.clean_chunk(
            title=request.title,
            language=request.language,
            transcript=transcript,
            start_seconds=start,
            end_seconds=end,
        )
    except ExternalServiceError as exc:
        if is_quota_error(exc):
            logger.warning("Lecture-notes quota exhausted", start=start)
            raise
        _log_cleanup_failure(start)
        return None
    return await _review_equations(ai, blocks, transcript, start)


async def _review_equations(
    ai: NotesAI,
    blocks: Sequence[NoteBlock],
    transcript: str,
    start: int,
) -> list[NoteBlock]:
    """Verify equations against the transcript, failing open on AI errors."""
    equations = [block for block in blocks if block.kind == "equation"]
    if not equations:
        return list(blocks)
    try:
        reviews = await ai.review_equations(transcript, equations)
    except DomainError as exc:
        logger.warning(
            "Equation review failed; keeping unreviewed equations",
            start=start,
            error=str(exc),
        )
        return list(blocks)
    return apply_reviews(blocks, reviews)


def _chunks(
    lines: Sequence[NotesLineData], chunk_seconds: int
) -> Iterator[tuple[int, int, list[NotesLineData]]]:
    """Split transcript lines into consecutive time windows."""
    if not lines:
        return
    start = int(lines[0].seconds // chunk_seconds) * chunk_seconds
    current: list[NotesLineData] = []
    for line in lines:
        bucket = int(line.seconds // chunk_seconds) * chunk_seconds
        if current and bucket != start:
            yield start, start + chunk_seconds, current
            start, current = bucket, []
        current.append(line)
    if current:
        yield start, start + chunk_seconds, current


def _dedupe_headings(
    existing: Sequence[NoteBlock], incoming: Sequence[NoteBlock]
) -> list[NoteBlock]:
    """Drop a heading that repeats the previous block's heading."""
    result = list(incoming)
    same_heading = (
        existing
        and existing[-1].kind == "heading"
        and result
        and result[0].kind == "heading"
        and existing[-1].text.strip() == result[0].text.strip()
    )
    return result[1:] if same_heading else result


def _fallback_blocks(lines: Sequence[NotesLineData]) -> list[NoteBlock]:
    """Group transcript lines into readable paragraphs with no timestamps."""
    blocks: list[NoteBlock] = []
    current: list[str] = []
    for line in lines:
        current.append(line.text.strip())
        if sum(len(part) + 1 for part in current) >= _FALLBACK_PARAGRAPH_CHARS:
            blocks.append(NoteBlock(kind="paragraph", text=" ".join(current)))
            current = []
    if current:
        blocks.append(NoteBlock(kind="paragraph", text=" ".join(current)))
    return blocks or [NoteBlock(kind="paragraph", text="No transcript is available.")]


def _render(
    request: NotesData,
    content: _NotesContent,
    notice: str,
    renderer_factory: Callable[[], LectureNotesPdfRenderer] | None,
) -> bytes:
    """Render the PDF on a worker thread."""
    renderer = (renderer_factory or LectureNotesPdfRenderer)()
    return renderer.render(request, content, notice)
