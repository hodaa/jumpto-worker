"""Centralized job timeout and retry watchdog policy.

A single transcription job is bounded by a ladder of nested watchdogs. Their
mutual ordering is load-bearing, so it is defined in exactly one place:

1. Pipeline timeout — ``job_timeout_seconds``.
   ``asyncio.wait_for`` in :func:`app.services.pipeline.run_pipeline`. This is
   the logical per-job budget for transcript fetching and must be the first
   watchdog to fire: the pipeline still owns the async stack, so it can fail
   the job with a user-safe "timed out" message before anything else can
   interrupt it.

2. Celery soft limit — ``job_timeout_seconds + soft margin``.
   A backstop for the pipeline timeout: a thread running inside
   ``asyncio.to_thread`` that blocks past the budget is not cancellable by
   ``wait_for`` alone, and the soft limit is what actually interrupts it. The
   +5s margin ensures the pipeline timeout normally wins first and the soft
   limit stays a failsafe, not the primary trigger.

3. Celery hard limit — ``job_timeout_seconds + task_timeout_grace_seconds``.
   The process-level kill. The grace (default 30s) leaves room for
   soft-limit cleanup before the worker process is terminated.

Invariant enforced by construction: pipeline < soft < hard.

The per-provider retry loops (media metadata retry, caption target loops) and
the task-level async-job retry budget (``download_and_transcribe``
``max_retries``/backoff) all run *inside* these bounds and are bounded by
them, never the other way round.
"""

from app.core.config import Settings

# Would-be-separate knob kept next to the ladder it belongs to: by how many
# seconds the Celery soft limit trails the pipeline timeout.
_SOFT_LIMIT_MARGIN_SECONDS = 5


def pipeline_timeout_seconds(settings: Settings) -> int:
    """Return the logical per-job budget for one transcription attempt."""
    return settings.job_timeout_seconds


def task_soft_time_limit_seconds(settings: Settings) -> int:
    """Return the Celery soft limit for a transcription task."""
    return settings.job_timeout_seconds + _SOFT_LIMIT_MARGIN_SECONDS


def task_hard_time_limit_seconds(settings: Settings) -> int:
    """Return the Celery hard limit (process kill) for a transcription task."""
    return settings.job_timeout_seconds + settings.task_timeout_grace_seconds
