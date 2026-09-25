"""Transcription pipeline orchestration.

Owns the end-to-end flow: resolve the single configured provider from settings,
fetch a transcript (resuming pending async jobs when the provider supports it),
build the submission, and report back through the backend's ``JobService``.
Provider strategy details, webhook URL building, and submission payload mapping
live in their own service modules; this module composes them.
"""

import asyncio
from collections.abc import Callable

from app.client import BackendClient
from app.core.config import get_settings
from app.core.exceptions import (
    ExternalServiceError,
    NoSpeechDetectedError,
    PermanentExternalServiceError,
)
from app.core.logging import get_logger
from app.core.timeouts import pipeline_timeout_seconds
from app.models import TranscriptSubmission
from app.providers import TranscriptJobPending
from app.providers.registry import resolve_provider
from app.services.jobs import JobService
from app.services.submissions import build_result_submission
from app.services.webhooks import build_assembly_webhook_url

logger = get_logger(__name__)

_USER_SAFE_FAILURE = "Transcription failed. Please try again later."
EXTERNAL_FAILURE = "Could not fetch the transcript for this video. Please try again later."
_TIMEOUT_SAFE_MESSAGE = "Transcription timed out. Please try again later."
NO_SPEECH_SAFE_MESSAGE = "No speech detected in this video."


async def run_pipeline(
    job_id: str,
    resume_token: str = "",
    resume_provider: str = "",
    client_factory: Callable[..., BackendClient] | None = None,
) -> dict:
    """Run the transcription pipeline for a job against the backend API.

    ``resume_token``/``resume_provider`` carry a pending cloud async job across
    Celery retries: when present the job is already ``processing`` and the
    pipeline resumes by checking that job once (routed to the originating
    provider) instead of requeueing a new transcription.
    """
    settings = get_settings()
    async with JobService(settings, client_factory=client_factory) as jobs:
        # Video context is populated once the job loads; the failure handler
        # includes it even when transcription itself raises. If the load fails
        # there is no video to report and the dict stays empty.
        video_context: dict[str, str] = {}
        try:
            job = await jobs.load(job_id)
            video_context = {
                "video_id": job.video_id,
                "youtube_video_id": job.youtube_video_id,
                "youtube_url": job.youtube_url,
            }
            logger.info(
                "Transcription job started",
                job_id=job_id,
                **video_context,
            )
            submission = await asyncio.wait_for(
                perform_transcription(job, resume_token, resume_provider),
                timeout=pipeline_timeout_seconds(settings),
            )
            await jobs.submit(job_id, submission)
        except TranscriptJobPending:
            logger.info(
                "Cloud provider job still processing; task will decide the outcome",
                job_id=job_id,
                resume_token=resume_token,
                resume_provider=resume_provider,
            )
            raise
        except NoSpeechDetectedError:
            logger.info(
                "Job completed without transcript (no speech detected)",
                job_id=job_id,
                **video_context,
            )
            await jobs.complete(job_id, NO_SPEECH_SAFE_MESSAGE)
            return {"status": "completed", "video_id": job.video_id}
        except Exception as exc:
            error = user_safe_message(exc)
            logger.exception(
                "Transcription pipeline failed",
                job_id=job_id,
                error=error,
                **video_context,
            )
            await jobs.fail(job_id, error)
            raise

    logger.info(
        "Pipeline completed",
        job_id=job_id,
        **video_context,
    )
    return {"status": "completed", "video_id": job.video_id}


async def perform_transcription(
    job, resume_token: str = "", resume_provider: str = ""
) -> TranscriptSubmission:
    settings = get_settings()
    provider = resolve_provider(settings)
    webhook_url = build_assembly_webhook_url(settings, job.job_id, provider.name)
    result = await try_provider(provider, job, resume_token, resume_provider, webhook_url)
    if result is not None:
        logger.info(
            "Transcript provider used",
            provider=provider.name,
            youtube_url=job.youtube_url,
        )
        if not result.transcript.text.strip():
            logger.warning(
                "Transcript contained no speech",
                provider=provider.name,
                youtube_url=job.youtube_url,
            )
            raise NoSpeechDetectedError("Transcript contained no speech")
        return build_result_submission(result, provider.name)

    raise ExternalServiceError(
        "Could not fetch the transcript for this video",
        service="transcription",
    )


async def try_provider(
    provider,
    job,
    resume_token: str = "",
    resume_provider: str = "",
    webhook_url: str = "",
):
    """Fetch via a single provider strategy; a soft miss returns ``None``.

    A still-processing async job (``TranscriptJobPending``) is re-raised so the
    task can decide whether to retry it, end it (a completion webhook was
    armed), or fail the job; a transient ``ExternalServiceError`` is logged and
    returned as ``None`` so the caller can fail the job with a user-safe
    message.

    The completion callback URL is only forwarded to strategies that declare
    ``supports_webhook`` (interface segregation: the pipeline checks the
    provider's declared capability, never a hard-coded name list).
    """
    resume = (
        resume_token
        if (resume_token and provider.supports_resume and provider.name == resume_provider)
        else ""
    )
    try:
        kwargs: dict = {"resume_token": resume}
        if webhook_url and not resume and getattr(provider, "supports_webhook", False):
            kwargs["webhook_url"] = webhook_url
        return await provider.fetch(job.youtube_url, job.youtube_video_id, **kwargs)
    except TranscriptJobPending:
        raise
    except PermanentExternalServiceError:
        logger.exception(
            "Permanent cloud transcript provider failure",
            provider=provider.name,
            youtube_url=job.youtube_url,
        )
        raise
    except ExternalServiceError:
        logger.exception(
            "Transcript provider failed; falling back",
            provider=provider.name,
            youtube_url=job.youtube_url,
        )
        return None


def user_safe_message(exc: Exception) -> str:
    """Map an exception to a user-safe failure message."""
    if isinstance(exc, NoSpeechDetectedError):
        return NO_SPEECH_SAFE_MESSAGE
    if isinstance(exc, ExternalServiceError | PermanentExternalServiceError):
        return EXTERNAL_FAILURE
    if isinstance(exc, TimeoutError):
        return _TIMEOUT_SAFE_MESSAGE
    return _USER_SAFE_FAILURE
