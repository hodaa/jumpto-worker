"""Transcript domain types shared across providers.

Holders of transcript data, word timings, and the async-job-pending signal.
These are pure data; provider and pipeline code depends on them.
"""

from dataclasses import dataclass
from typing import Any

from app.core.exceptions import ExternalServiceError


class TranscriptJobPending(ExternalServiceError):
    """A cloud provider escalated transcription to an async job not yet done.

    Raised by providers that queue work on their side (e.g. Supadata AI
    generation). ``resume_token`` lets a later attempt resume the same job
    instead of queuing a fresh transcription; the orchestrator routes it back
    to the originating provider via ``provider``. ``resumable`` separates jobs
    the caller may wait on (retry with backoff, carrying the resume token)
    from providers running a strict one-call policy (e.g. TranscriptFetch),
    where the job must be failed rather than polled or retried.

    ``webhook`` marks a job whose completion is expected to arrive via an
    externally wired webhook (the submit already armed it); the pipeline must
    end the task cleanly instead of poller-retrying, since Assembly waits for
    the async transcription that outlasts the local retry budget.
    """

    def __init__(
        self,
        message: str = "Transcription service is still processing",
        *,
        provider: str,
        resume_token: str = "",
        resumable: bool = True,
        webhook: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.resume_token = resume_token
        self.resumable = resumable
        self.webhook = webhook
        super().__init__(
            message,
            service=provider,
            details={**(details or {}), "resume_token": resume_token},
        )


@dataclass
class TranscriptWordData:
    """Single word with timing information."""

    word: str
    start_time: float
    end_time: float


@dataclass
class TranscriptData:
    """Full transcript with per-word timestamps."""

    language: str
    text: str
    words: list[TranscriptWordData]


def _close_word_times(words: list[TranscriptWordData]) -> None:
    """Set each word's end_time from the next word's start (or a fallback gap)."""
    for index in range(len(words) - 1):
        words[index].end_time = words[index + 1].start_time
    if words:
        words[-1].end_time = words[-1].start_time + 0.3


def _language_base(code: str) -> str:
    """Reduce a locale/region language code to its base portion."""
    return code.split("-")[0].split("_")[0].lower()
