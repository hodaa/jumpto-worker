"""Backend API client for the JumpTo worker."""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from app.client.http import get_shared_http_client
from app.core.exceptions import BackendCommunicationError
from app.core.logging import get_logger
from app.models import JobData, TranscriptSubmission

logger = get_logger(__name__)

_REQUEST_TIMEOUT_SECONDS = 30
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE_SECONDS = 0.25
_RETRY_BACKOFF_MAX_SECONDS = 4.0
_INTERNAL_API_KEY_HEADER = "X-Internal-API-Key"


def _retry_delay(attempt: int, response: httpx.Response | None = None) -> float:
    """Return bounded exponential backoff with optional Retry-After support."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if isinstance(retry_after, str):
            try:
                return min(float(retry_after), _RETRY_BACKOFF_MAX_SECONDS)
            except ValueError:
                pass
    cap = min(_RETRY_BACKOFF_BASE_SECONDS * (2**attempt), _RETRY_BACKOFF_MAX_SECONDS)
    return random.uniform(cap / 2, cap)


class BackendClient:
    """HTTP client for the backend's internal worker API."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = get_shared_http_client(timeout=_REQUEST_TIMEOUT_SECONDS)

    async def close(self) -> None:
        """Keep the process-level connection pool alive for later tasks."""
        # The client belongs to the worker-process pool, not this task. It is
        # closed when the persistent worker event loop shuts down.
        return None

    async def get_job(self, job_id: str) -> JobData:
        """Fetch job and video data for a job id."""
        data = await self._request("GET", f"/internal/jobs/{job_id}")
        return JobData(**data)

    async def advance_job(self, job_id: str) -> None:
        """Mark a job as processing."""
        await self._request("POST", f"/internal/jobs/{job_id}/advance")

    async def store_transcript(self, job_id: str, submission: TranscriptSubmission) -> None:
        """Store a transcript (words + metadata) for a job."""
        payload: dict[str, Any] = {
            "title": submission.title,
            "duration_seconds": submission.duration_seconds,
            "language": submission.language,
            "transcript_text": submission.transcript_text,
            "provider": submission.provider,
            "words": [
                {
                    "word_index": word.word_index,
                    "word": word.word,
                    "start_time": word.start_time,
                    "end_time": word.end_time,
                }
                for word in submission.words
            ],
        }
        await self._request("POST", f"/internal/jobs/{job_id}/transcript", json=payload)

    async def complete_job(self, job_id: str) -> None:
        """Mark a job as completed."""
        await self._request("POST", f"/internal/jobs/{job_id}/complete")

    async def fail_job(self, job_id: str, error: str) -> None:
        """Mark a job as failed with a user-safe error message."""
        await self._request("POST", f"/internal/jobs/{job_id}/fail", json={"error": error})

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send an authenticated request to the internal API, with retries."""
        url = f"{self.base_url}{path}"
        headers = {_INTERNAL_API_KEY_HEADER: self.api_key}
        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.request(method, url, headers=headers, json=json)
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning(
                    "Backend request failed (network)",
                    method=method,
                    path=path,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt + 1 < _MAX_RETRIES:
                    await asyncio.sleep(_retry_delay(attempt))
                    continue
                raise BackendCommunicationError(
                    f"Failed to reach backend at {path}"
                ) from last_error
            if response.status_code >= 500:
                last_error = BackendCommunicationError(
                    f"Backend returned {response.status_code} for {path}"
                )
                logger.warning(
                    "Backend request failed (server error)",
                    method=method,
                    path=path,
                    status_code=response.status_code,
                    attempt=attempt + 1,
                )
                if attempt + 1 < _MAX_RETRIES:
                    await asyncio.sleep(_retry_delay(attempt, response))
                    continue
                raise last_error
            if response.status_code >= 400:
                raise BackendCommunicationError(
                    f"Backend rejected request {method} {path}: {response.status_code}"
                )
            data = response.content
            if not data:
                return {}
            return response.json()
        raise BackendCommunicationError(f"Backend request failed for {path}")  # pragma: no cover
