"""Celery task shell for original-footage video summaries."""

from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.services.event_loop import run_async
from app.services.summaries import (
    FAILURE_SAFE_MESSAGE,
    SUMMARY_QUOTA_MESSAGE,
    fail_summary,
    run_summary,
)
from app.tasks.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(bind=True, name="app.tasks.summaries.generate_video_summary")
def generate_video_summary(self, summary_id: str) -> dict[str, str]:
    """Generate an original-footage summary for a queued request."""

    async def _run() -> str:
        return await run_summary(summary_id)

    try:
        run_async(_run())
    except ExternalServiceError as exc:
        if exc.details.get("quota_exhausted"):
            logger.warning("Video summary quota exhausted", summary_id=summary_id)
            run_async(fail_summary(summary_id, SUMMARY_QUOTA_MESSAGE))
            raise
        logger.warning("Video summary failed", summary_id=summary_id, exc_info=True)
        run_async(fail_summary(summary_id, FAILURE_SAFE_MESSAGE))
        raise
    except Exception:
        logger.warning("Video summary failed", summary_id=summary_id, exc_info=True)
        run_async(fail_summary(summary_id, FAILURE_SAFE_MESSAGE))
        raise
    return {"status": "ready"}
