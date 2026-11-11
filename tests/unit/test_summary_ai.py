"""Unit tests for AI segment scoring."""

import json

import httpx
import pytest

from app.integrations.summary_ai import (
    SUMMARY_SCHEMA,
    OpenAISummaryAI,
    scores_from_payload,
    summary_scoring_prompt,
)
from app.services.summary_segments import SummarySegment


def _segment(index: int, text: str = "a definition") -> SummarySegment:
    """Build one unscored segment."""
    return SummarySegment(
        index=index,
        start_seconds=index * 60,
        end_seconds=index * 60 + 60,
        text=text,
    )


def _response(payload: dict) -> httpx.Response:
    """Build a successful Chat Completions response carrying JSON text."""
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": json.dumps(payload)}}]},
    )


def _ai(handler) -> OpenAISummaryAI:
    """Build a scorer over an in-memory transport."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAISummaryAI(
        api_key="secret-key",
        model="gpt-test",
        base_url="https://openai.test/v1",
        client=client,
    )


class TestPrompt:
    """The prompt exposes indices and times, never asking for timestamps."""

    def test_renders_each_segment_with_its_index_and_window(self) -> None:
        prompt = summary_scoring_prompt("Algebra", "en", [_segment(3, "solve for x")])

        assert "Lecture title: Algebra" in prompt
        assert "Language: en" in prompt
        assert "[3] 00:03:00-00:04:00" in prompt
        assert "solve for x" in prompt
        assert "Never invent ids" in prompt

    def test_uses_unknown_language_when_missing(self) -> None:
        prompt = summary_scoring_prompt("Algebra", None, [_segment(0)])

        assert "Language: unknown" in prompt


class TestScoresFromPayload:
    """Parsing keeps only well-formed scores for known ids."""

    def test_keeps_known_ids_and_clamps_their_score(self) -> None:
        scores = scores_from_payload(
            {
                "scores": [
                    {"id": 0, "score": 12, "reason": " core "},
                    {"id": 1, "score": -3, "reason": "intro"},
                ]
            },
            {0, 1, 2},
        )

        assert [(score.id, score.score) for score in scores] == [(0, 10.0), (1, 0.0)]
        assert scores[0].reason == "core"

    def test_drops_unknown_duplicate_and_malformed_entries(self) -> None:
        scores = scores_from_payload(
            {
                "scores": [
                    {"id": 9, "score": 8, "reason": "unknown"},
                    {"id": 0, "score": 7, "reason": "first"},
                    {"id": 0, "score": 2, "reason": "duplicate"},
                    {"id": "bad", "score": 5},
                    {"score": 4},
                    "not-a-dict",
                ]
            },
            {0},
        )

        assert [(score.id, score.score, score.reason) for score in scores] == [(0, 7.0, "first")]

    def test_missing_scores_key_yields_nothing(self) -> None:
        assert scores_from_payload({}, {0}) == []


class TestSchema:
    """The response schema is strict and requires every declared key."""

    def test_requires_every_score_field(self) -> None:
        item = SUMMARY_SCHEMA["properties"]["scores"]["items"]

        assert item["additionalProperties"] is False
        assert item["required"] == ["id", "score", "reason"]
        assert set(item["required"]) == set(item["properties"])


class TestScoreSegments:
    """The scorer posts a strict schema and validates returned ids."""

    @pytest.mark.asyncio
    async def test_posts_the_scoring_schema_and_keeps_valid_scores(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return _response(
                {
                    "scores": [
                        {"id": 0, "score": 9, "reason": "definition"},
                        {"id": 7, "score": 10, "reason": "invented"},
                    ]
                }
            )

        scores = await _ai(handler).score_segments(
            title="Algebra",
            language="en",
            segments=[_segment(0)],
        )

        assert seen["response_format"]["json_schema"]["name"] == "summary_scoring"
        assert seen["response_format"]["json_schema"]["strict"] is True
        assert "You rank sections" in seen["messages"][1]["content"]
        assert [(score.id, score.score) for score in scores] == [(0, 9.0)]

    @pytest.mark.asyncio
    async def test_empty_batch_makes_no_request(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no request expected")

        assert await _ai(handler).score_segments(title="Algebra", language="en", segments=[]) == []
