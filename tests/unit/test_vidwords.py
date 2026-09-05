"""Unit tests for the VidWords transcript provider."""

import httpx
import pytest

from app.core.exceptions import ExternalServiceError
from app.providers.vidwords import VidWordsPermanentError, VidWordsTranscriptProvider

WATCH_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"

_SENTINEL = object()


def _provider(handler) -> VidWordsTranscriptProvider:
    """Build a provider wired to an httpx mock transport."""
    return VidWordsTranscriptProvider(
        api_key="test-token",
        base_url="https://vidwords.test",
        lang="en",
        transport=httpx.MockTransport(handler),
    )


def _json_response(body: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status, json=body, request=httpx.Request("POST", "https://vidwords.test/api/transcripts")
    )


def _success_item() -> dict:
    return {
        "id": "jNQXAC9IVRw",
        "title": "Me at the zoo",
        "author": "jawed",
        "language": "English",
        "languageCode": "en-US",
        "isGenerated": False,
        "text": "All right, so here we are in front of the, uh, elephants.",
        "segments": [
            {"text": "All right, so here", "start": 0.0, "duration": 4.2},
            {"text": "we are in front of the elephants.", "start": 4.2, "duration": 3.6},
        ],
    }


class TestFetchSuccess:
    """Tests for a successful VidWords transcript fetch."""

    @pytest.mark.asyncio
    async def test_builds_result_and_converts_segments_to_words(self) -> None:
        provider = _provider(lambda request: _json_response({"results": [_success_item()]}))

        result = await provider.fetch(WATCH_URL)

        assert result is not None
        assert result.title == "Me at the zoo"
        assert result.author == "jawed"
        assert result.is_generated is False
        assert result.duration_seconds == 8  # ceil(4.2 + 3.6)
        assert result.transcript.language == "en"
        assert result.transcript.text == "All right, so here we are in front of the, uh, elephants."
        # Words in the first cue share cue start; second cue starts at 4.2.
        assert result.transcript.words[0].start_time == 0.0
        assert result.transcript.words[4].start_time == 4.2
        # End times are closed from the next word's start.
        assert result.transcript.words[0].end_time == result.transcript.words[1].start_time

    @pytest.mark.asyncio
    async def test_sends_expected_payload_and_auth(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("Authorization")
            captured["body"] = request.content
            return _json_response({"results": [_success_item()]})

        provider = _provider(handler)
        await provider.fetch(WATCH_URL)

        assert captured["auth"] == "Basic test-token"
        import json

        assert json.loads(captured["body"]) == {"ids": [WATCH_URL], "lang": "en"}


class TestFetchErrorBranches:
    """Tests for the VidWords error/miss branches."""

    @pytest.mark.asyncio
    async def test_no_transcript_returns_none(self) -> None:
        item = {"id": "xyz", "error": "no_transcript", "message": "Video has no captions"}
        provider = _provider(lambda request: _json_response({"results": [item]}))

        assert await provider.fetch(WATCH_URL) is None

    @pytest.mark.asyncio
    async def test_transcripts_disabled_returns_none(self) -> None:
        item = {"id": "xyz", "error": "transcripts_disabled", "message": ""}
        provider = _provider(lambda request: _json_response({"results": [item]}))

        assert await provider.fetch(WATCH_URL) is None

    @pytest.mark.asyncio
    async def test_video_unavailable_is_permanent(self) -> None:
        item = {
            "id": "aaaaaaaaaaa",
            "error": "video_unavailable",
            "message": "Video is unavailable",
        }
        provider = _provider(lambda request: _json_response({"results": [item]}))

        with pytest.raises(VidWordsPermanentError):
            await provider.fetch(WATCH_URL)

    @pytest.mark.asyncio
    async def test_fetch_failed_is_transient(self) -> None:
        item = {"id": "xyz", "error": "fetch_failed", "message": "Could not reach YouTube"}
        provider = _provider(lambda request: _json_response({"results": [item]}))

        with pytest.raises(ExternalServiceError) as excinfo:
            await provider.fetch(WATCH_URL)
        assert not isinstance(excinfo.value, VidWordsPermanentError)

    @pytest.mark.asyncio
    async def test_unauthorized_is_permanent(self) -> None:
        provider = _provider(
            lambda request: _json_response(
                {"error": "unauthorized", "message": "Bad key"}, status=401
            )
        )

        with pytest.raises(VidWordsPermanentError):
            await provider.fetch(WATCH_URL)

    @pytest.mark.asyncio
    async def test_rate_limited_is_transient(self) -> None:
        provider = _provider(
            lambda request: _json_response({"error": "rate_limited", "message": ""}, status=429)
        )

        with pytest.raises(ExternalServiceError) as excinfo:
            await provider.fetch(WATCH_URL)
        assert not isinstance(excinfo.value, VidWordsPermanentError)

    @pytest.mark.asyncio
    async def test_no_results_raises(self) -> None:
        provider = _provider(lambda request: _json_response({"results": []}))

        with pytest.raises(ExternalServiceError):
            await provider.fetch(WATCH_URL)

    @pytest.mark.asyncio
    async def test_network_error_is_external_service_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        provider = _provider(handler)

        with pytest.raises(ExternalServiceError):
            await provider.fetch(WATCH_URL)
