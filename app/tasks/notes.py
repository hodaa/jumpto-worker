"""Celery task shell for lecture notes."""

from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.services.event_loop import run_async
from app.services.notes import (
    FAILURE_SAFE_MESSAGE,
    NOTES_QUOTA_MESSAGE,
    fail_notes,
    run_notes,
)
from app.tasks.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(bind=True, name="app.tasks.notes.generate_lecture_notes")
def generate_lecture_notes(self, note_id: str) -> dict[str, str]:
    """Generate lecture notes for a queued request."""

    async def _run() -> str:
        return await run_notes(note_id)

    try:
        run_async(_run())
    except ExternalServiceError as exc:
        if exc.details.get("quota_exhausted"):
            logger.warning("Lecture notes quota exhausted", note_id=note_id)
            run_async(fail_notes(note_id, NOTES_QUOTA_MESSAGE))
            raise
        logger.warning("Lecture notes failed", note_id=note_id, exc_info=True)
        run_async(fail_notes(note_id, FAILURE_SAFE_MESSAGE))
        raise
    except Exception:
        logger.warning("Lecture notes failed", note_id=note_id, exc_info=True)
        run_async(fail_notes(note_id, FAILURE_SAFE_MESSAGE))
        raise
    return {"status": "ready"}
