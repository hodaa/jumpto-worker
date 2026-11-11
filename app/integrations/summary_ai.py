"""Score transcript segments for an original-footage summary.

The model never sees or returns timestamps that drive cuts. It receives
stable segment indices and transcript text, and returns a 0-10 educational
value score per index; the pipeline maps those indices back to the transcript
timings it already owns.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.core.config import Settings
from app.integrations.notes_ai import format_timestamp
from app.integrations.openai import OpenAINotesAI, _completion_body
from app.models import SummarySegment

_SUMMARY_SYSTEM = "You score lecture transcript sections. Return JSON only."
_MAX_SEGMENT_CHARS = 900

SUMMARY_SCORING_PROMPT = r"""You rank sections of a lecture transcript for a short recap video.

Return JSON only in this exact format:
{"scores":[{"id":0,"score":8.5,"reason":"..."}]}

Rules:
- Score every id you are given, exactly once. Never invent ids.
- score is 0-10: 10 = essential definition, core explanation, formula, or
  exam-relevant worked example; 5 = useful supporting explanation; 0 = intro,
  admin, greeting, repetition, filler, or off-topic chatter.
- Judge educational value for a student revising the lecture, not
  entertainment value.
- Be decisive: use the full range and avoid giving every section the same
  score.
- Write "reason" in the lecture's language, in one short sentence.
- The caller selects clips by these scores; it does not use any text you
  write as a timestamp."""


@dataclass(frozen=True)
class SegmentScore:
    """One model verdict for a segment index."""

    id: int
    score: float
    reason: str = ""


class SummaryAI(Protocol):
    """Provider-neutral interface for scoring summary segments."""

    @property
    def configured(self) -> bool:
        """Whether an API key is available for scoring."""

    async def score_segments(
        self,
        *,
        title: str,
        language: str | None,
        segments: Sequence[SummarySegment],
    ) -> list[SegmentScore]:
        """Score every supplied segment index."""


SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "score", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}


def summary_scoring_prompt(
    title: str,
    language: str | None,
    segments: Sequence[SummarySegment],
) -> str:
    """Build the scoring prompt for one batch of segments."""
    rendered = "\n\n".join(_render_segment(segment) for segment in segments)
    return (
        f"{SUMMARY_SCORING_PROMPT}\n\n"
        f"Lecture title: {title}\n"
        f"Language: {language or 'unknown'}\n\n"
        f"Segments:\n{rendered}"
    )


def scores_from_payload(payload: dict[str, Any], valid_ids: set[int]) -> list[SegmentScore]:
    """Keep well-formed scores for known ids, clamping and de-duplicating."""
    result: dict[int, SegmentScore] = {}
    for item in payload.get("scores") or []:
        if not isinstance(item, dict):
            continue
        try:
            segment_id = int(item["id"])
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if segment_id not in valid_ids or segment_id in result:
            continue
        result[segment_id] = SegmentScore(
            id=segment_id,
            score=max(0.0, min(score, 10.0)),
            reason=str(item.get("reason") or "").strip(),
        )
    return [result[segment_id] for segment_id in sorted(result)]


class OpenAISummaryAI(OpenAINotesAI):
    """Segment scoring through the same OpenAI client as lecture notes."""

    @classmethod
    def from_settings(
        cls, settings: Settings, client: httpx.AsyncClient | None = None
    ) -> OpenAISummaryAI:
        """Build the scorer from worker settings."""
        return cls(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_api_url,
            client=client,
        )

    async def score_segments(
        self,
        *,
        title: str,
        language: str | None,
        segments: Sequence[SummarySegment],
    ) -> list[SegmentScore]:
        """Score one batch of segments with strict JSON output."""
        if not segments:
            return []
        prompt = summary_scoring_prompt(title, language, segments)
        payload = await self._post(
            _completion_body(
                self._model, prompt, SUMMARY_SCHEMA, "summary_scoring", _SUMMARY_SYSTEM
            )
        )
        return scores_from_payload(payload, {segment.index for segment in segments})


def _render_segment(segment: SummarySegment) -> str:
    """Render one segment with index, timestamps, and trimmed transcript."""
    text = segment.text.strip()
    if len(text) > _MAX_SEGMENT_CHARS:
        text = f"{text[:_MAX_SEGMENT_CHARS]}…"
    window = (
        f"{format_timestamp(int(segment.start_seconds))}-"
        f"{format_timestamp(int(segment.end_seconds))}"
    )
    return f"[{segment.index}] {window}\n{text}"
