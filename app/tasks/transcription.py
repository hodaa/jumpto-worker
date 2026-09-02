"""Celery tasks for video transcription."""

import asyncio

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.client.backend import BackendClient
from app.models import JobData
from app.providers.media import get_media_info
from app.tasks.celery_app import celery_app  # ✅ CORRECT IMPORT

logger = get_logger(__name__)

# Progress percentages
PROGRESS_FETCH = 20
PROGRESS_PARSE = 50
PROGRESS_STORE = 80


# ✅ CELERY TASK DECORATOR
@celery_app.task(bind=True, max_retries=3)
def download_and_transcribe(self, job_id: str) -> dict:
    """
    ✅ Celery task entry point.
    Runs the async transcription pipeline.
    
    Args:
        job_id: ID of the job to process
    
    Returns:
        Dictionary with completion status and video_id
    """
    try:
        logger.info("Starting task", job_id=job_id)
        result = asyncio.run(run_pipeline(job_id))
        logger.info("Task completed successfully", job_id=job_id)
        return result
        
    except Exception as exc:
        logger.error("Task failed", job_id=job_id, exc_info=True)
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60, max_retries=3)


async def run_pipeline(job_id: str) -> dict:
    """Run the transcription pipeline for a job against the backend API."""
    settings = get_settings()
    client = BackendClient(settings.backend_url, settings.internal_api_key)

    async def report(progress: int) -> None:
        """Report progress to the backend, ignoring reporting failures."""
        try:
            await client.report_progress(job_id, progress)
        except Exception:
            logger.exception("Failed to report progress", job_id=job_id, progress=progress)

    try:
        job = await client.get_job(job_id)
        if job.status != "pending":
            logger.info("Skipping non-pending job", job_id=job_id, status=job.status)
            return {"status": job.status}

        await client.advance_job(job_id)
        
        submission = await asyncio.wait_for(
            _perform_transcription(job, report),
            timeout=settings.job_timeout_seconds,
        )
        
        await report(PROGRESS_STORE)
        await client.store_transcript(job_id, submission)
        await client.complete_job(job_id)
        
    except asyncio.TimeoutError:
        error = "Transcription timed out"
        logger.error(error, job_id=job_id)
        try:
            await client.fail_job(job_id, error)
        except Exception:
            logger.exception("Failed to mark job as failed", job_id=job_id)
        raise
    
    except ExternalServiceError as exc:
        error = str(exc)
        logger.error("Transcription service error", job_id=job_id, error=error)
        try:
            await client.fail_job(job_id, error)
        except Exception:
            logger.exception("Failed to mark job as failed", job_id=job_id)
        raise
    
    except Exception as exc:
        error = _user_safe_message(exc)
        logger.exception("Transcription pipeline failed", job_id=job_id, error=error)
        try:
            await client.fail_job(job_id, error)
        except Exception:
            logger.exception("Failed to mark job as failed", job_id=job_id)
        raise

    logger.info("Pipeline completed", job_id=job_id)
    return {"status": "completed", "video_id": job.youtube_video_id}


async def _perform_transcription(
    job: JobData,
    report: callable,
) -> dict:
    """
    ✅ Perform transcription using Assembly.ai YouTube support.
    NO yt-dlp, NO bot detection, NO local audio download!
    """
    logger.info(
        "Starting transcription",
        job_id=job.job_id,
        youtube_url=job.youtube_url,
    )
    
    try:
        # ✅ Step 1: Fetch media info and transcript from Assembly.ai
        await report(PROGRESS_FETCH)
        
        logger.info("Fetching media from Assembly.ai", job_id=job.job_id)
        
        media = await get_media_info(
            job.youtube_video_id,
            job.youtube_url
        )
        
        logger.info(
            "Media fetched successfully",
            job_id=job.job_id,
            language=media["language"],
            word_count=len(media["words"])
        )
        
        # ✅ Step 2: Parse transcript data
        await report(PROGRESS_PARSE)
        
        submission = {
            "video_id": job.youtube_video_id,
            "youtube_url": job.youtube_url,
            "language": media["language"],
            "text": media["text"],
            "words": [
                {
                    "text": word.word,
                    "start": int(word.start_time * 1000),  # Milliseconds
                    "end": int(word.end_time * 1000),      # Milliseconds
                }
                for word in media["words"]
            ],
        }
        
        logger.info(
            "Transcription completed successfully",
            job_id=job.job_id,
            word_count=len(submission["words"])
        )
        
        return submission
    
    except asyncio.TimeoutError:
        logger.error("Assembly.ai transcription timed out", job_id=job.job_id)
        raise ExternalServiceError(
            "Transcription timed out - video too long or Assembly.ai unavailable",
            service="assembly"
        )
    
    except ExternalServiceError as e:
        logger.error(
            "Assembly.ai service error",
            job_id=job.job_id,
            error=str(e),
            service=e.service
        )
        raise
    
    except Exception as e:
        logger.error(
            "Unexpected transcription error",
            job_id=job.job_id,
            error=str(e),
            exc_info=True
        )
        raise ExternalServiceError(
            f"Transcription failed: {str(e)}",
            service="transcription"
        ) from e


def _user_safe_message(exc: Exception) -> str:
    """Convert exception to user-safe error message."""
    if isinstance(exc, ExternalServiceError):
        return str(exc)
    
    exc_type = type(exc).__name__
    
    if "timeout" in str(exc).lower():
        return "Transcription took too long. Please try a shorter video."
    
    if "not found" in str(exc).lower():
        return "Video not found or no longer available."
    
    if "private" in str(exc).lower() or "restricted" in str(exc).lower():
        return "This video is not accessible."
    
    return f"Transcription failed ({exc_type}). Please try again."