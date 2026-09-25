"""Backend job lifecycle for the transcription pipeline.

``JobService`` holds the backend client for a single job: connect, load
(advancing pending jobs to ``processing``), submit (store transcript +
complete), best-effort fail, and release. The pipeline and the Celery task
depend on this abstraction rather than reaching for ``BackendClient``
directly. The underlying httpx connection is shared process-wide and closed by
the worker event-loop teardown, not by the per-job service.
"""

from collections.abc import Callable

from app.client import BackendClient
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models import JobData

logger = get_logger(__name__)


def default_client_factory(settings) -> BackendClient:
    """Build a backend client from worker settings."""
    return BackendClient(settings.backend_url, settings.internal_api_key)


class JobService:
    """Backend client lifecycle for a single transcription job."""

    def __init__(
        self,
        settings,
        client_factory: Callable[..., BackendClient] | None = None,
    ) -> None:
        self._settings = settings
        self._client_factory = client_factory or default_client_factory
        self._client: BackendClient | None = None

    def connect(self) -> "JobService":
        """Construct the backend client (idempotent)."""
        if self._client is None:
            self._client = self._client_factory(self._settings)
        return self

    async def __aenter__(self) -> "JobService":
        return self.connect()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def load(self, job_id: str) -> JobData:
        """Fetch a job and advance it from ``pending`` to ``processing``."""
        client = self._require_client()
        job = await client.get_job(job_id)
        if job.status == "pending":
            await client.advance_job(job_id)
        return job

    async def submit(self, job_id: str, submission) -> None:
        """Submit the transcript and mark the job completed."""
        client = self._require_client()
        await client.store_transcript(job_id, submission)
        await client.complete_job(job_id)

    async def fail(self, job_id: str, message: str) -> None:
        """Best-effort mark a job as failed through an existing client."""
        client = self._require_client()
        try:
            await client.fail_job(job_id, message)
        except Exception:
            logger.exception("Failed to mark job as failed", job_id=job_id)

    async def close(self) -> None:
        """Release this service's handle on the pooled backend client.

        The httpx client is owned by the process event loop
        (``get_shared_http_client``) and closed at worker shutdown, not here.
        """
        self._client = None

    def _require_client(self) -> BackendClient:
        if self._client is None:
            raise RuntimeError("JobService is not connected; call connect() first")
        return self._client


async def fail_job(
    job_id: str,
    message: str,
    client_factory: Callable[..., BackendClient] | None = None,
) -> None:
    """Best-effort mark a job as failed in the backend outside a pipeline run."""
    settings = get_settings()
    service = JobService(settings, client_factory=client_factory).connect()
    try:
        await service.fail(job_id, message)
    finally:
        await service.close()
