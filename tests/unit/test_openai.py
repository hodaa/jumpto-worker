"""Unit tests for the OpenAI transcript-cleanup client."""

import json

import httpx
import pytest

from app.core.exceptions import ExternalServiceError, PermanentExternalServiceError
from app.integrations import openai as openai_client
from app.integrations.equation_review import EquationReview
from app.integrations.openai import OpenAINotesAI
from app.models import NoteBlock


def _response(payload: dict) -> httpx.Response:
    """Build a successful Chat Completions response carrying JSON text."""
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": json.dumps(payload)}}]},
    )


def _ai(handler) -> OpenAINotesAI:
    """Build a client over an in-memory transport."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAINotesAI(
        api_key="secret-key",
        model="gpt-test",
        base_url="https://openai.test/v1",
        client=client,
    )


def _clean(ai: OpenAINotesAI, transcript: str = "hello"):
    """Run one cleanup call with the test defaults."""
    return ai.clean_chunk(
        title="Lecture",
        language="en",
        transcript=transcript,
        start_seconds=0,
        end_seconds=600,
    )


class TestRequest:
    """The client asks for strict JSON and keeps the key out of the URL."""

    @pytest.mark.asyncio
    async def test_sends_a_strict_json_schema_and_the_prompt(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("Authorization", "")
            seen.update(json.loads(request.content))
            return _response({"blocks": []})

        await _clean(_ai(handler))

        assert seen["url"] == "https://openai.test/v1/chat/completions"
        assert seen["auth"] == "Bearer secret-key"
        assert seen["model"] == "gpt-test"
        assert seen["response_format"]["json_schema"]["strict"] is True
        assert seen["response_format"]["json_schema"]["name"] == "lecture_notes"
        prompt = seen["messages"][1]["content"]
        assert "structured study notes" in prompt
        assert "Equations are optional" in prompt
        assert seen["response_format"]["json_schema"]["schema"]["properties"]["blocks"]["items"][
            "properties"
        ]["type"]["enum"] == ["heading", "paragraph", "bullet", "equation"]


class TestCleanChunk:
    """Cleanup keeps only well-formed blocks."""

    @pytest.mark.asyncio
    async def test_filters_invalid_blocks(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _response(
                {
                    "blocks": [
                        {"type": "heading", "text": "Intro", "latex": ""},
                        {"type": "equation", "latex": r"x^2", "text": ""},
                        {"type": "equation", "text": "", "latex": ""},
                        {"type": "nonsense", "text": "drop", "latex": ""},
                    ]
                }
            )

        blocks = await _clean(_ai(handler))

        assert [block.kind for block in blocks] == ["heading", "equation"]
        assert blocks[1].latex == "x^2"


class TestReviewEquations:
    """The review call uses its own strict schema and parses verdicts."""

    @pytest.mark.asyncio
    async def test_requests_a_review_and_parses_verdicts(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return _response(
                {
                    "reviews": [
                        {
                            "index": 0,
                            "verdict": "corrected",
                            "corrected_latex": "p^{x}",
                            "reason": "sign",
                        }
                    ]
                }
            )

        reviews = await _ai(handler).review_equations(
            "the variance is p times one minus p",
            [NoteBlock(kind="equation", latex="p^x", evidence="variance is p times")],
        )

        assert reviews == [
            EquationReview(
                index=0,
                verdict="corrected",
                corrected_latex="p^{x}",
                reason="sign",
            )
        ]
        assert seen["response_format"]["json_schema"]["name"] == "equation_review"
        prompt = seen["messages"][1]["content"]
        assert "[0] p^x" in prompt
        assert "Evidence: variance is p times" in prompt

    @pytest.mark.asyncio
    async def test_skips_the_request_without_equations(self) -> None:
        called = False

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return _response({"reviews": []})

        assert await _ai(handler).review_equations("transcript", []) == []
        assert called is False


class TestRetriesAndFailures:
    """Transient errors retry; exhausted quotas and credentials do not."""

    @pytest.mark.asyncio
    async def test_retries_a_transient_429(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return _response({"blocks": [{"type": "paragraph", "text": "ok"}]})

        async def no_sleep(seconds: float) -> None:
            return None

        monkeypatch.setattr(openai_client.asyncio, "sleep", no_sleep)
        blocks = await _clean(_ai(handler))

        assert calls == 2
        assert blocks[0].text == "ok"

    @pytest.mark.asyncio
    async def test_stops_on_an_exhausted_quota(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "You exceeded your current quota",
                        "code": "insufficient_quota",
                    }
                },
            )

        with pytest.raises(ExternalServiceError) as caught:
            await _clean(_ai(handler))

        assert calls == 1
        assert caught.value.details["quota_exhausted"] is True

    @pytest.mark.asyncio
    async def test_rejects_credentials_without_retrying(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(401, json={"error": "invalid key"})

        with pytest.raises(PermanentExternalServiceError):
            await _clean(_ai(handler))

        assert calls == 1

    @pytest.mark.asyncio
    async def test_rejects_invalid_json(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})

        with pytest.raises(ExternalServiceError):
            await _clean(_ai(handler))

    @pytest.mark.asyncio
    async def test_rejects_missing_choices(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": []})

        with pytest.raises(ExternalServiceError):
            await _clean(_ai(handler))


class TestOverview:
    """The overview call uses its own strict schema and returns the text."""

    @pytest.mark.asyncio
    async def test_requests_and_parses_the_overview(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return _response({"overview": "ملخص المحاضرة"})

        ai = _ai(handler)

        overview = await ai.overview(title="محاضرة", language="ar", transcript="نص")

        assert overview == "ملخص المحاضرة"
        schema = seen["response_format"]["json_schema"]
        assert schema["name"] == "lecture_overview"
        assert schema["schema"]["required"] == ["overview"]
        assert "brief overview" in seen["messages"][1]["content"]
