"""Celery task probing one video's quality ladder for the backend cache.

Best-effort by design: the backend only queues a probe when it has no cached
ladder (or the cached one is stale), serves the cached value meanwhile, and
simply leaves the cache as it was when a probe fails - there is no job or
download to fail and nothing user-visible to report, so the task swallows its
own errors after logging them.
"""

from collections.abc import Callable

from app.client import BackendClient
from app.core.logging import get_logger
from app.services.downloads import probe_and_store_formats
from app.services.event_loop import run_async
from app.tasks.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task
def probe_video_formats(
    youtube_video_id: str,
    youtube_url: str,
    client_factory: Callable[..., BackendClient] | None = None,
) -> dict:
    """Probe a video's H.264 ladder and cache it in the backend."""
    try:
        heights = run_async(
            probe_and_store_formats(youtube_video_id, youtube_url, client_factory=client_factory)
        )
    except Exception:
        logger.exception("Format probe failed", youtube_video_id=youtube_video_id)
        return {"status": "failed"}
    return {"status": "completed", "heights": heights}
