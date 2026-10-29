"""Shared prompt and response handling for lecture-notes AI providers."""

from __future__ import annotations

import json
import random
import re
from typing import Any, Protocol

import httpx

from app.core.exceptions import ExternalServiceError
from app.integrations.equation_review import EquationReview
from app.integrations.latex import normalize_math
from app.models import NoteBlock

VALID_BLOCK_KINDS = frozenset({"heading", "paragraph", "equation", "bullet"})
_HEADING_TRAILING = ".:;،؛"
_ORPHAN_LABEL = re.compile(r"^\(\d+\)$")
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
MAX_RETRIES = 3
RETRY_BACKOFF_BASE_SECONDS = 2.0
RETRY_BACKOFF_MAX_SECONDS = 30.0
_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

NOTES_PROMPT = r"""You turn a lecture transcript into concise, well-structured study notes.

Return JSON only in this exact format:
{"blocks":[{"type":"heading","text":"...","note":"","latex":"","evidence":""}]}

Allowed block types: "heading", "paragraph", "bullet", "equation".

Rules:
- Write in the lecture's language (Arabic when the transcript is Arabic).
- Organize the lecture into 2-6 short topic sections. Begin each section with a heading, then explain it with paragraphs and bullets.
- Be faithful and concise: capture the main ideas, definitions, examples, and caveats the lecturer stresses. Never invent facts.
- Equations are optional. Add an equation block only when the lecturer explicitly states or clearly describes a formula. If the lecture has no formulas, return no equation blocks; never invent one.
- For heading, paragraph, and bullet blocks, set "note", "latex", and "evidence" to empty strings.
- For equation blocks, write "text" as a one- or two-sentence description in the lecture's language (Arabic when the transcript is Arabic) and "note" as an optional short clarification in the same language (empty string if none).
- Put the LaTeX in "latex" with no $, $$, \[, \], markdown fences, or labels.
- Write piecewise definitions as separate equations. Never use \begin, \end, cases, aligned, or any LaTeX environment.
- Repair obvious ASR errors only when unambiguous.
- Include closest transcript phrase in "evidence" (use empty string if none).
- Use valid LaTeX for matplotlib. Allowed: \frac, \sqrt, \sum, \int, \binom, \Gamma, \operatorname, \mathrm, \text, \times, \cdot, \leq, \geq, \neq, \approx, \to, \infty, \partial, \nabla, \quad, \left, \right. Always brace \frac (\frac{1}{2}). Use \leq/\geq not \le/\ge.
- Never use \big, \Big, \bigg, \Bigg, \displaystyle, or Arabic text inside an equation.
- English/standard math symbols inside equations.
- Do not add a document title; start directly with the first topic heading.
- Return at least one block. If the transcript has no usable content, return one paragraph saying so."""


class NotesAI(Protocol):
    """Provider-neutral interface used by the notes pipeline."""

    @property
    def configured(self) -> bool:
        """Whether an API key is available for cleanup."""

    async def overview(self, *, title: str, language: str | None, transcript: str) -> str:
        """Return a brief lecture overview in the transcript's language."""

    async def clean_chunk(
        self,
        *,
        title: str,
        language: str | None,
        transcript: str,
        start_seconds: int,
        end_seconds: int,
    ) -> list[NoteBlock]:
        """Clean one transcript window into note blocks."""

    async def review_equations(
        self, transcript: str, equations: list[NoteBlock]
    ) -> list[EquationReview]:
        """Verify one chunk's equations against its transcript."""


def format_timestamp(seconds: int) -> str:
    """Format a position as HH:MM:SS for prompts."""
    hours, rest = divmod(max(int(seconds), 0), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def notes_prompt(
    title: str,
    language: str | None,
    transcript: str,
    start_seconds: int,
    end_seconds: int,
) -> str:
    """Build the cleanup prompt for one transcript window."""
    window = f"{format_timestamp(start_seconds)}-{format_timestamp(end_seconds)}"
    return (
        f"{NOTES_PROMPT}\n\n"
        f"Lecture title: {title}\n"
        f"Language: {language or 'unknown'}\n"
        f"Window: {window}\n\n"
        f"Transcript:\n{transcript}"
    )


OVERVIEW_PROMPT = r"""You write a brief overview of a lecture from its transcript.

Return JSON only in this exact format:
{"overview":"..."}

Rules:
- Write in the lecture's language (Arabic when the transcript is Arabic).
- Keep it brief: one short paragraph of two to four sentences, or one short paragraph plus up to five bullet lines starting with "• ".
- Cover only the main topics, key definitions, and any caveats the lecturer stresses. Match the transcript; never invent facts.
- Do not add a heading or title; the document supplies one.
- Detailed notes follow this overview. Do not repeat them line by line and do not include LaTeX."""


def overview_prompt(title: str, language: str | None, transcript: str) -> str:
    """Build the overview prompt for a lecture transcript."""
    return (
        f"{OVERVIEW_PROMPT}\n\n"
        f"Lecture title: {title}\n"
        f"Language: {language or 'unknown'}\n\n"
        f"Transcript:\n{transcript}"
    )


_VALID_JSON_ESCAPES = frozenset('"\\/bfnrt')
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _escape_stray_backslashes(text: str) -> str:
    """Escape backslashes the model left as bare LaTeX in a JSON string.

    LaTeX commands such as ``\\leq`` or ``\\underline`` are invalid JSON
    escapes. Doubling only those backslashes lets the document parse while
    leaving genuine escapes (``\\n``, ``\\t``, ``\\u1234``) to the normal path.
    """
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char != "\\":
            out.append(char)
            index += 1
            continue
        following = text[index + 1] if index + 1 < length else ""
        if following in _VALID_JSON_ESCAPES:
            out.append(text[index : index + 2])
            index += 2
            continue
        if (
            following == "u"
            and index + 6 <= length
            and all(digit in _HEX_DIGITS for digit in text[index + 2 : index + 6])
        ):
            out.append(text[index : index + 6])
            index += 6
            continue
        out.append("\\\\")
        index += 1
    return "".join(out)


def parse_json(text: str, service: str) -> dict[str, Any]:
    """Parse a model response that may be wrapped in a JSON code fence."""
    cleaned = _JSON_FENCE.sub("", text.strip())
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            payload = json.loads(_escape_stray_backslashes(cleaned))
        except json.JSONDecodeError as exc:
            raise ExternalServiceError(
                "Lecture notes AI returned invalid JSON", service=service
            ) from exc
    if not isinstance(payload, dict):
        raise ExternalServiceError(
            "Lecture notes AI returned an unexpected payload", service=service
        )
    return payload


def retry_delay(attempt: int, response: httpx.Response | None = None) -> float:
    """Return bounded exponential backoff, honouring Retry-After when present."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), RETRY_BACKOFF_MAX_SECONDS)
            except ValueError:
                pass
    cap = min(RETRY_BACKOFF_BASE_SECONDS * (2**attempt), RETRY_BACKOFF_MAX_SECONDS)
    return random.uniform(cap / 2, cap)


def _heading_text(text: str) -> str:
    """Collapse whitespace and drop a trailing sentence mark from a heading."""
    return " ".join(text.split()).rstrip(_HEADING_TRAILING).strip()


def blocks_from_payload(payload: dict[str, Any]) -> list[NoteBlock]:
    """Validate and normalize model blocks into renderer blocks."""
    blocks: list[NoteBlock] = []
    for item in payload.get("blocks", []):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type", "")).strip()
        if kind not in VALID_BLOCK_KINDS:
            continue
        text = str(item.get("text", "")).strip()
        latex = normalize_math(str(item.get("latex", ""))).strip()
        evidence = str(item.get("evidence", "")).strip()
        note = str(item.get("note", "")).strip()
        if kind == "equation":
            if latex:
                blocks.append(
                    NoteBlock(
                        kind="equation",
                        text=text,
                        latex=latex,
                        evidence=evidence,
                        note=note,
                    )
                )
            continue
        if _ORPHAN_LABEL.match(text):
            continue
        if kind == "heading":
            text = _heading_text(text)
        elif kind == "bullet":
            text = re.sub(r"^[•\-*]\s*", "", text).strip()
        if text:
            blocks.append(NoteBlock(kind=kind, text=text))
    return blocks
