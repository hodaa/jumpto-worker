"""Unit tests for the TranscriptFetch transcript provider."""

import httpx
import pytest

from app.core.exceptions import ExternalServiceError
from app.providers.transcript import TranscriptJobPending
from app.providers.transcriptfetch import (
    TranscriptFetchPermanentError,
    TranscriptFetchTranscriptProvider,
)

WATCH_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
VIDEO_ID = "jNQXAC9IVRw"


def _provider(handler) -> TranscriptFetchTranscriptProvider:
    """Build a provider wired to an httpx mock transport."""
    return TranscriptFetchTranscriptProvider(
        api_key="test-token",
        base_url="https://transcriptfetch.test",
        lang="en",
        transport=httpx.MockTransport(handler),
    )


def _ok_response(body: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={"ok": True, "request_id": "req_1", "data": body, "usage": {}},
        request=httpx.Request("POST", "https://transcriptfetch.test/api/v2/transcripts/video"),
    )


def _transcript_data() -> dict:
    return {
        "kind": "transcript",
        "video_id": VIDEO_ID,
        "title": "Me at the zoo",
        "source": "captions",
        "segments": [
            {"start": 0, "duration": 3.5, "text": "All right, so here"},
            {"start": 3.5, "duration": 2.0, "text": "we are in front of the elephants."},
        ],
    }


class TestFetchSuccess:
    """Tests for a successful synchronous TranscriptFetch fetch."""

    @pytest.mark.asyncio
    async def test_builds_result_and_converts_segments_to_words(self) -> None:
        result = await _provider(lambda request: _ok_response(_transcript_data())).fetch(
            WATCH_URL, VIDEO_ID
        )

        assert result is not None
        assert result.title == "Me at the zoo"
        assert result.duration_seconds == 6
        assert result.transcript.language == "en"
        assert result.transcript.text == "All right, so here we are in front of the elephants."
        assert result.transcript.words[0].start_time == 0.0
        assert result.transcript.words[4].start_time == 3.5
        assert result.is_generated is False

    @pytest.mark.asyncio
    async def test_sends_bearer_auth_and_auto_mode(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("Authorization")
            captured["json"] = request.content
            return _ok_response(_transcript_data())

        await _provider(handler).fetch(WATCH_URL, VIDEO_ID)

        assert captured["auth"] == "Bearer test-token"
        import json

        body = json.loads(captured["json"])
        assert body["video"] == WATCH_URL
        assert body["mode"] == "auto"
        assert body["timestamps"] is True

    @pytest.mark.asyncio
    async def test_audio_source_marks_generated(self) -> None:
        data = {**_transcript_data(), "source": "audio", "text": "Some audio transcript"}
        result = await _provider(lambda request: _ok_response(data)).fetch(WATCH_URL, VIDEO_ID)

        assert result is not None
        assert result.is_generated is True
        assert result.transcript.text == "Some audio transcript"


class TestFetchMissAndErrors:
    """Tests for TranscriptFetch misses and error branches."""

    @pytest.mark.asyncio
    async def test_no_captions_is_a_soft_miss(self) -> None:
        body = {"ok": False, "error": {"code": "no_captions", "message": "No caption track"}}
        provider = _provider(
            lambda request: httpx.Response(
                200, json=body, request=httpx.Request("POST", "https://transcriptfetch.test")
            )
        )

        assert await provider.fetch(WATCH_URL, VIDEO_ID) is None

    @pytest.mark.asyncio
    async def test_unauthorized_is_permanent(self) -> None:
        body = {"ok": False, "error": {"code": "unauthorized", "message": "Bad key"}}
        provider = _provider(
            lambda request: httpx.Response(
                401, json=body, request=httpx.Request("POST", "https://transcriptfetch.test")
            )
        )

        with pytest.raises(TranscriptFetchPermanentError):
            await provider.fetch(WATCH_URL, VIDEO_ID)

    @pytest.mark.asyncio
    async def test_rate_limited_is_transient(self) -> None:
        body = {"ok": False, "error": {"code": "rate_limited", "message": "Slow down"}}
        provider = _provider(
            lambda request: httpx.Response(
                429, json=body, request=httpx.Request("POST", "https://transcriptfetch.test")
            )
        )

        with pytest.raises(ExternalServiceError):
            await provider.fetch(WATCH_URL, VIDEO_ID)

    @pytest.mark.asyncio
    async def test_202_raises_non_resumable_pending(self) -> None:
        body = {
            "ok": True,
            "request_id": "req_1",
            "status": "processing",
            "job_id": "asr_123",
            "poll_url": "/api/v2/transcripts/jobs/asr_123",
        }
        provider = _provider(
            lambda request: httpx.Response(
                202, json=body, request=httpx.Request("POST", "https://transcriptfetch.test")
            )
        )

        with pytest.raises(TranscriptJobPending) as excinfo:
            await provider.fetch(WATCH_URL, VIDEO_ID)

        assert excinfo.value.resume_token == "asr_123"
        assert excinfo.value.provider == "transcriptfetch"
        assert excinfo.value.resumable is False

    @pytest.mark.asyncio
    async def test_resume_token_is_refused_not_resumed(self) -> None:
        provider = _provider(
            lambda request: httpx.Response(
                202, request=httpx.Request("POST", "https://transcriptfetch.test")
            )
        )

        with pytest.raises(TranscriptJobPending) as excinfo:
            await provider.fetch(WATCH_URL, VIDEO_ID, resume_token="asr_123")

        assert excinfo.value.provider == "transcriptfetch"
        assert excinfo.value.resumable is False
        assert excinfo.value.resume_token == "asr_123"

    @pytest.mark.asyncio
    async def test_network_error_is_external_service_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        provider = _provider(handler)

        with pytest.raises(ExternalServiceError):
            await provider.fetch(WATCH_URL, VIDEO_ID)
