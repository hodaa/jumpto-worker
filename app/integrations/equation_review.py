"""Second-pass review for equations generated from a lecture transcript."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.integrations.latex import normalize_math
from app.models import NoteBlock

VALID_VERDICTS = frozenset({"supported", "corrected", "unsupported", "uncertain"})
RECONSTRUCTED_MARKER = "[Reconstructed: verify against lecture]"
_UNCERTAIN_VERDICTS = frozenset({"uncertain", "unsupported"})

EQUATION_REVIEW_PROMPT = r"""You are a mathematical reviewer for lecture notes.
The transcript is the source of intent, but mathematical correctness takes
precedence over noisy speech-to-text. If the transcript is garbled, use
"uncertain" instead of "correcting" a valid equation to match it.

Check for:
- missing or extra factors, signs, exponents, limits, subscripts, and indices;
- whether an integral or series converges;
- whether the stated domain, parameter constraints, and assumptions are valid
  (for example 0 <= p <= 1, lambda > 0, x >= 0);
- whether PMF, PDF, CDF, survival function, and MGF are used correctly;
- simple numerical substitutions and arithmetic;
- whether a distribution convention matches the transcript. For example, the
  geometric distribution has E[X] = 1/p when counting trials up to the first
  success, but E[X] = (1-p)/p when counting failures before the first success.
  If the transcript does not settle the convention, use "uncertain"; never
  choose one silently;
- broken ASR notation such as "rac", "imes", dropped backslashes, or split
  subscripts;
- whether the equation matches what the lecturer actually said.

Return one review for every equation index. Use these verdicts:
- "supported": the equation is correct and supported by the transcript.
- "corrected": the equation is wrong or incomplete; put the full corrected
  single-line LaTeX in corrected_latex.
- "unsupported": the equation is wrong and cannot be corrected confidently.
- "uncertain": the transcript does not contain enough evidence to decide.

Do not invent a correction. If the transcript is ambiguous or garbled, use
"uncertain". Use matplotlib mathtext-compatible LaTeX in corrected_latex:
write \leq and \geq, never \le or \ge, and brace both \frac arguments.
A "corrected" equation must be complete and self-contained, and must preserve
the PMF/PDF/CDF/survival meaning and its domain. Return JSON only."""


@dataclass(frozen=True)
class EquationReview:
    """One verifier verdict for a candidate equation."""

    index: int
    verdict: str
    corrected_latex: str = ""
    reason: str = ""


def equation_review_prompt(transcript: str, equations: Sequence[NoteBlock]) -> str:
    """Build the verification prompt for a chunk's equations."""
    listing = "\n\n".join(_equation_entry(index, block) for index, block in enumerate(equations))
    return f"{EQUATION_REVIEW_PROMPT}\n\nTranscript:\n{transcript}\n\nEquations:\n{listing}"


def _equation_entry(index: int, block: NoteBlock) -> str:
    """Format one candidate equation and its transcript evidence."""
    entry = f"[{index}] {block.latex}"
    return f"{entry}\nEvidence: {block.evidence}" if block.evidence else entry


def reviews_from_payload(payload: dict[str, Any], count: int) -> list[EquationReview]:
    """Parse valid verifier verdicts, ignoring malformed entries."""
    reviews: list[EquationReview] = []
    for item in payload.get("reviews", []):
        review = _review_from_item(item, count)
        if review is not None:
            reviews.append(review)
    return reviews


def _review_from_item(item: Any, count: int) -> EquationReview | None:
    """Normalize one verifier entry when its index and verdict are valid."""
    if not isinstance(item, dict):
        return None
    try:
        index = int(item.get("index"))
    except (TypeError, ValueError):
        return None
    verdict = str(item.get("verdict", "")).strip().lower()
    if not 0 <= index < count or verdict not in VALID_VERDICTS:
        return None
    return EquationReview(
        index=index,
        verdict=verdict,
        corrected_latex=normalize_math(str(item.get("corrected_latex", ""))).strip(),
        reason=str(item.get("reason", "")).strip(),
    )


def apply_reviews(
    blocks: Sequence[NoteBlock], reviews: Sequence[EquationReview]
) -> list[NoteBlock]:
    """Keep every equation, applying verifier corrections where offered."""
    by_index = {review.index: review for review in reviews}
    result: list[NoteBlock] = []
    equation_index = 0
    for position, block in enumerate(blocks):
        if block.kind != "equation":
            result.append(block)
            continue
        review = by_index.get(equation_index)
        equation_index += 1
        replacement = _reviewed_block(block, review)
        if replacement is not None:
            result.append(replacement)
            if review is not None and review.verdict in _UNCERTAIN_VERDICTS:
                _append_marker(result, blocks, position)
    return result


def _append_marker(blocks: list[NoteBlock], source: Sequence[NoteBlock], position: int) -> None:
    """Mark a reconstructed equation without duplicating an existing marker."""
    following = source[position + 1] if position + 1 < len(source) else None
    previous = source[position - 1] if position else None
    marked_next_to_equation = (
        following is not None and following.text == RECONSTRUCTED_MARKER
    ) or (previous is not None and previous.text == RECONSTRUCTED_MARKER)
    if marked_next_to_equation or (blocks and blocks[-1].text == RECONSTRUCTED_MARKER):
        return
    blocks.append(NoteBlock(kind="paragraph", text=RECONSTRUCTED_MARKER))


def _reviewed_block(block: NoteBlock, review: EquationReview | None) -> NoteBlock | None:
    """Return the equation, applying a correction when the reviewer offers one."""
    if review is not None and review.verdict == "corrected" and review.corrected_latex:
        return NoteBlock(
            kind="equation",
            text=block.text,
            latex=normalize_math(review.corrected_latex),
            evidence=block.evidence,
            note=block.note,
        )
    return block
