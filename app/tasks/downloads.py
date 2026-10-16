"""Celery task producing one offloaded video download.

The orchestration lives in ``app.services.downloads``; this module is the thin
Celery-facing shell: on failure it reports a user-safe message back to the
backend (the UI reads it from the download status) and re-raises so the task
itself is also recorded as failed.
"""

from collections.abc import Callable

from app.client import BackendClient
from app.core.logging import get_logger
from app.services.downloads import FAILURE_SAFE_MESSAGE, fail_download, run_download
from app.services.event_loop import run_async
from app.tasks.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task
def download_video(
    download_id: str,
    client_factory: Callable[..., BackendClient] | None = None,
) -> dict:
    """Produce one offloaded video download and report its blob URL."""
    try:
        run_async(run_download(download_id, client_factory=client_factory))
    except Exception:
        logger.exception("Video download failed", download_id=download_id)
        run_async(fail_download(download_id, FAILURE_SAFE_MESSAGE, client_factory=client_factory))
        raise
    return {"status": "ready"}
