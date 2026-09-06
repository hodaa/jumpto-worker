"""Unit tests for the Supadata transcript provider."""

import httpx
import pytest

from app.core.exceptions import ExternalServiceError
from app.providers.supadata import (
    SupadataPermanentError,
    SupadataTranscriptProvider,
)
from app.providers.transcript import TranscriptJobPending

WATCH_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
VIDEO_ID = "jNQXAC9IVRw"


def _provider(handler) -> SupadataTranscriptProvider:
    """Build a provider wired to an httpx mock transport."""
    return SupadataTranscriptProvider(
        api_key="test-token",
        base_url="https://api.supadata.test",
        lang="en",
        transport=httpx.MockTransport(handler),
    )


def _resp(body: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status, json=body, request=httpx.Request("GET", "https://api.supadata.test")
    )


def _transcript_body() -> dict:
    return {
        "content": [
            {"text": "All right, so here", "offset": 0, "duration": 4200},
            {"text": "we are in front of the elephants.", "offset": 4200, "duration": 3600},
        ],
        "lang": "en",
        "availableLangs": ["en", "es"],
    }


def _video_body() -> dict:
    return {
        "id": VIDEO_ID,
        "title": "Me at the zoo",
        "duration": 19,
        "channel": {"id": "UCxDAujTVTCRy3Fhs", "name": "jawed"},
    }


class TestFetchSuccess:
    """Tests for a successful Supadata fetch."""

    @pytest.mark.asyncio
    async def test_builds_result_and_converts_chunks_to_words(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return (
                _resp(_video_body())
                if request.url.path.endswith("/youtube/video")
                else _resp(_transcript_body())
            )

        result = await _provider(handler).fetch(WATCH_URL, VIDEO_ID)

        assert result is not None
        assert result.title == "Me at the zoo"
        assert result.author == "jawed"
        assert result.duration_seconds == 19  # from metadata, not chunks
        assert result.transcript.language == "en"
        assert result.transcript.text == "All right, so here we are in front of the elephants."
        # Chunk offsets are ms, converted to seconds.
        assert result.transcript.words[0].start_time == 0.0
        assert result.transcript.words[4].start_time == 4.2
        assert result.transcript.words[0].end_time == result.transcript.words[1].start_time

    @pytest.mark.asyncio
    async def test_falls_back_to_chunk_end_when_metadata_missing_duration(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/youtube/video"):
                body = {**_video_body(), "duration": None}
            else:
                body = _transcript_body()
            return _resp(body)

        result = await _provider(handler).fetch(WATCH_URL, VIDEO_ID)

        assert result.duration_seconds == 8  # ceil((4200 + 3600) / 1000)

    @pytest.mark.asyncio
    async def test_sends_auth_header(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/transcript"):
                captured["auth"] = request.headers.get("x-api-key")
                captured["path"] = request.url.path
                captured["params"] = dict(request.url.params)
            return (
                _resp(_video_body())
                if request.url.path.endswith("/youtube/video")
                else _resp(_transcript_body())
            )

        await _provider(handler).fetch(WATCH_URL, VIDEO_ID)

        assert captured["auth"] == "test-token"
        assert captured["path"] == "/transcript"
        assert captured["params"] == {"url": WATCH_URL, "lang": "en", "mode": "auto"}


class TestFetchErrorBranches:
    """Tests for the Supadata error/miss branches."""

    @pytest.mark.asyncio
    async def test_transcript_unavailable_returns_none(self) -> None:
        body = {"error": "transcript-unavailable", "message": "No transcript available"}
        provider = _provider(lambda request: _resp(body, status=206))

        assert await provider.fetch(WATCH_URL, VIDEO_ID) is None

    @pytest.mark.asyncio
    async def test_unauthorized_is_permanent(self) -> None:
        body = {"error": "unauthorized", "message": "Bad key"}
        provider = _provider(lambda request: _resp(body, status=401))

        with pytest.raises(SupadataPermanentError):
            await provider.fetch(WATCH_URL, VIDEO_ID)

    @pytest.mark.asyncio
    async def test_rate_limited_is_transient(self) -> None:
        body = {"error": "limit-exceeded", "message": "Slow down"}
        provider = _provider(lambda request: _resp(body, status=429))

        with pytest.raises(ExternalServiceError) as excinfo:
            await provider.fetch(WATCH_URL, VIDEO_ID)
        assert not isinstance(excinfo.value, SupadataPermanentError)

    @pytest.mark.asyncio
    async def test_not_found_is_permanent(self) -> None:
        body = {"error": "not-found", "message": "Video not found"}
        provider = _provider(lambda request: _resp(body, status=404))

        with pytest.raises(SupadataPermanentError):
            await provider.fetch(WATCH_URL, VIDEO_ID)

    @pytest.mark.asyncio
    async def test_metadata_error_propagates_after_transcript_ok(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/youtube/video"):
                return _resp({"error": "limit-exceeded", "message": "Slow down"}, status=429)
            return _resp(_transcript_body())

        provider = _provider(handler)

        with pytest.raises(ExternalServiceError):
            await provider.fetch(WATCH_URL, VIDEO_ID)

    @pytest.mark.asyncio
    async def test_network_error_is_external_service_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        provider = _provider(handler)

        with pytest.raises(ExternalServiceError):
            await provider.fetch(WATCH_URL, VIDEO_ID)


class TestFetchAsyncJob:
    """Tests for the async job flow (202 -> task retries, single call/attempt)."""

    @pytest.mark.asyncio
    async def test_202_queues_job_and_raises_pending(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/youtube/video"):
                return _resp(_video_body())
            return _resp({"jobId": "job-123"}, status=202)

        provider = _provider(handler)

        with pytest.raises(TranscriptJobPending) as excinfo:
            await provider.fetch(WATCH_URL, VIDEO_ID)
        assert excinfo.value.resume_token == "job-123"
        assert excinfo.value.provider == "supadata"
        assert excinfo.value.resumable is True

    @pytest.mark.asyncio
    async def test_resume_completed_job_returns_result_in_one_call(self) -> None:
        polled: dict = {"times": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/youtube/video"):
                return _resp(_video_body())
            polled["times"] += 1
            assert request.url.path == "/transcript/job-123"
            return _resp(
                {
                    "status": "completed",
                    "content": _transcript_body()["content"],
                    "lang": "en",
                    "availableLangs": ["en"],
                }
            )

        result = await _provider(handler).fetch(WATCH_URL, VIDEO_ID, resume_token="job-123")

        assert polled["times"] == 1
        assert result is not None
        assert result.transcript.text == "All right, so here we are in front of the elephants."
        assert result.is_generated is True

    @pytest.mark.asyncio
    async def test_resume_processing_job_raises_pending_after_one_call(self) -> None:
        polled: dict = {"times": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            polled["times"] += 1
            return _resp({"status": "active"})

        provider = _provider(handler)

        with pytest.raises(TranscriptJobPending) as excinfo:
            await provider.fetch(WATCH_URL, VIDEO_ID, resume_token="job-123")
        assert excinfo.value.resume_token == "job-123"
        assert excinfo.value.provider == "supadata"
        assert excinfo.value.resumable is True
        assert polled["times"] == 1

    @pytest.mark.asyncio
    async def test_resume_failed_job_is_permanent(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _resp({"status": "failed", "error": "boom"})

        with pytest.raises(SupadataPermanentError):
            await _provider(handler).fetch(WATCH_URL, VIDEO_ID, resume_token="job-123")

    @pytest.mark.asyncio
    async def test_missing_job_id_is_permanent(self) -> None:
        provider = _provider(lambda request: _resp({"status": "active"}, status=202))

        with pytest.raises(SupadataPermanentError):
            await provider.fetch(WATCH_URL, VIDEO_ID)


class TestEmptyContent:
    """Tests for responses that carry no usable transcript."""

    @pytest.mark.asyncio
    async def test_empty_content_is_a_miss(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/youtube/video"):
                return _resp(_video_body())
            return _resp({"content": [], "lang": "en", "availableLangs": []})

        result = await _provider(handler).fetch(WATCH_URL, VIDEO_ID)

        assert result is None
