"""Backend API client for the JumpTo worker."""

from __future__ import annotations

from typing import Any

import httpx

from app.core.exceptions import BackendCommunicationError
from app.core.logging import get_logger
from app.models import JobData, TranscriptSubmission

logger = get_logger(__name__)

_REQUEST_TIMEOUT_SECONDS = 30
_MAX_RETRIES = 3
_INTERNAL_API_KEY_HEADER = "X-Internal-API-Key"


class BackendClient:
    """HTTP client for the backend's internal worker API."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def get_job(self, job_id: str) -> JobData:
        """Fetch job and video data for a job id."""
        data = await self._request("GET", f"/internal/jobs/{job_id}")
        return JobData(**data)

    async def advance_job(self, job_id: str) -> None:
        """Mark a job as processing."""
        await self._request("POST", f"/internal/jobs/{job_id}/advance")

    async def report_progress(self, job_id: str, progress: int) -> None:
        """Report intermediate progress for a job."""
        await self._request(
            "POST", f"/internal/jobs/{job_id}/progress", json={"progress": progress}
        )

    async def store_transcript(self, job_id: str, submission: TranscriptSubmission) -> None:
        """Store a transcript (words + metadata) for a job."""
        payload: dict[str, Any] = {
            "title": submission.title,
            "duration_seconds": submission.duration_seconds,
            "language": submission.language,
            "transcript_text": submission.transcript_text,
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
                async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                    response = await client.request(method, url, headers=headers, json=json)
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
