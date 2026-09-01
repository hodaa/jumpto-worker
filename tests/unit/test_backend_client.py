"""Unit tests for the BackendClient."""

from unittest.mock import AsyncMock, Mock

import pytest

from app.client.backend import BackendClient
from app.core.exceptions import BackendCommunicationError
from app.models import JobData, TranscriptSubmission, TranscriptWordData

_BASE = "http://backend.test:8000"
_API_KEY = "test-key"


def _client_context(response: Mock) -> AsyncMock:
    """Build an httpx.AsyncClient mock that returns a given response."""
    client = AsyncMock()
    client.request.return_value = response
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def _json_response(status_code: int, data: dict | None = None) -> Mock:
    """Build a mock response with optional JSON payload."""
    response = Mock()
    response.status_code = status_code
    response.json.return_value = data or {}
    response.content = str(data or {}).encode()
    return response


@pytest.mark.asyncio
async def test_get_job_returns_job_data(monkeypatch) -> None:
    response = _json_response(
        200,
        {
            "job_id": "job-1",
            "video_id": "video-1",
            "youtube_video_id": "abcde12345",
            "youtube_url": "https://www.youtube.com/watch?v=abcde12345",
            "status": "pending",
        },
    )
    client_context = _client_context(response)
    monkeypatch.setattr("app.client.backend.httpx.AsyncClient", lambda **kw: client_context)

    client = BackendClient(_BASE, _API_KEY)
    job = await client.get_job("job-1")

    assert isinstance(job, JobData)
    assert job.youtube_video_id == "abcde12345"
    headers = client_context.request.await_args.kwargs["headers"]
    assert headers["X-Internal-API-Key"] == _API_KEY


@pytest.mark.asyncio
async def test_advance_job_sends_post(monkeypatch) -> None:
    response = _json_response(200, {"status": "processing"})
    client_context = _client_context(response)
    monkeypatch.setattr("app.client.backend.httpx.AsyncClient", lambda **kw: client_context)

    client = BackendClient(_BASE, _API_KEY)
    await client.advance_job("job-1")

    assert client_context.request.await_args.args[0] == "POST"


@pytest.mark.asyncio
async def test_report_progress_sends_progress_payload(monkeypatch) -> None:
    response = _json_response(200, {"status": "processing"})
    client_context = _client_context(response)
    monkeypatch.setattr("app.client.backend.httpx.AsyncClient", lambda **kw: client_context)

    client = BackendClient(_BASE, _API_KEY)
    await client.report_progress("job-1", 70)

    assert client_context.request.await_args.args[0] == "POST"
    assert client_context.request.await_args.args[1].endswith("/progress")
    assert client_context.request.await_args.kwargs["json"] == {"progress": 70}


@pytest.mark.asyncio
async def test_store_transcript_sends_words(monkeypatch) -> None:
    response = _json_response(200, {"status": "pending"})
    client_context = _client_context(response)
    monkeypatch.setattr("app.client.backend.httpx.AsyncClient", lambda **kw: client_context)

    submission = TranscriptSubmission(
        title="My Video",
        duration_seconds=240,
        language="en",
        transcript_text="hello world",
        words=[TranscriptWordData(word_index=0, word="hello", start_time=0.0, end_time=0.5)],
    )

    client = BackendClient(_BASE, _API_KEY)
    await client.store_transcript("job-1", submission)

    payload = client_context.request.await_args.kwargs["json"]
    assert payload["transcript_text"] == "hello world"
    assert payload["words"][0]["word_index"] == 0


@pytest.mark.asyncio
async def test_complete_job(monkeypatch) -> None:
    response = _json_response(200, {"status": "completed"})
    client_context = _client_context(response)
    monkeypatch.setattr("app.client.backend.httpx.AsyncClient", lambda **kw: client_context)

    client = BackendClient(_BASE, _API_KEY)
    await client.complete_job("job-1")

    assert client_context.request.await_args.args[1].endswith("/complete")


@pytest.mark.asyncio
async def test_fail_job_sends_error(monkeypatch) -> None:
    response = _json_response(200, {"status": "failed"})
    client_context = _client_context(response)
    monkeypatch.setattr("app.client.backend.httpx.AsyncClient", lambda **kw: client_context)

    client = BackendClient(_BASE, _API_KEY)
    await client.fail_job("job-1", "Something went wrong")

    payload = client_context.request.await_args.kwargs["json"]
    assert payload["error"] == "Something went wrong"


@pytest.mark.asyncio
async def test_get_job_raises_on_non_2xx(monkeypatch) -> None:
    response = _json_response(404)
    client_context = _client_context(response)
    monkeypatch.setattr("app.client.backend.httpx.AsyncClient", lambda **kw: client_context)

    client = BackendClient(_BASE, _API_KEY)
    with pytest.raises(BackendCommunicationError):
        await client.get_job("job-1")


@pytest.mark.asyncio
async def test_request_retries_on_5xx(monkeypatch) -> None:
    first = _json_response(500)
    second = _json_response(
        200,
        {
            "job_id": "job-1",
            "video_id": "video-1",
            "youtube_video_id": "abcde12345",
            "youtube_url": "https://www.youtube.com/watch?v=abcde12345",
            "status": "pending",
        },
    )
    client_context = _client_context(first)
    client_context.request.side_effect = [first, second]
    monkeypatch.setattr("app.client.backend.httpx.AsyncClient", lambda **kw: client_context)

    client = BackendClient(_BASE, _API_KEY)
    await client.get_job("job-1")

    assert client_context.request.await_count == 2
