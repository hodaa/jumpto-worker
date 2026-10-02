"""Job-scoped log context: what identifies a job in its own log records.

Owns one responsibility — the *shape* of that identity and its lifecycle. Where
it comes from and when it is cleared are other modules' business: Celery's
task lifecycle in ``app.tasks.celery_app``, the render format in
``app.core.logging``. Because :func:`bind`, :func:`current` and ``CONTEXT_KEYS``
all derive from :class:`JobContext`, adding an identifier is a single line on
the dataclass — no call site, signal or processor learns the new field's name.

Only *identity* belongs here: which job, video and attempt a record belongs to.
Per-event data (``error``, ``progress``, ``resume_token``, ``word_count``)
stays an explicit keyword on the ``logger.*`` call, because binding it globally
would stamp a stale value onto every unrelated record.
"""

from dataclasses import asdict, dataclass, fields, replace

from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    get_contextvars,
)


@dataclass(frozen=True)
class JobContext:
    """The identity attached to every log record emitted while a job runs."""

    job_id: str = ""
    retry_attempt: int = 0
    video_id: str = ""
    youtube_video_id: str = ""
    youtube_url: str = ""

    def merged(self, **updates: str) -> "JobContext":
        """Return a copy with ``updates`` applied, ignoring empty values.

        Identity is discovered in two phases: the job id is known when Celery
        starts the task, the video id only once the backend has answered, so the
        second phase merges into the first rather than replacing it. Frozen, so
        a context outliving one task can never be mutated underneath another.
        """
        return replace(self, **{k: v for k, v in updates.items() if v})


# Derived from the dataclass, never hand-maintained: an identity field exists in
# the rendered context the moment it exists on JobContext.
CONTEXT_KEYS = tuple(f.name for f in fields(JobContext))


def bind(context: JobContext) -> None:
    """Attach ``context``'s populated fields to subsequent log records.

    Bound once per Celery task so records emitted deep in the pipeline —
    provider, audio, HTTP client — carry the job they belong to instead of each
    re-deriving it. Empty fields are left unbound rather than bound as empty
    strings, so a failure that never learned the video does not match a
    ``video_id:`` filter in Sentry.
    """
    bind_contextvars(**{k: v for k, v in asdict(context).items() if v != ""})


def current() -> JobContext:
    """Return the bound context; fields that were never bound keep their default."""
    bound = get_contextvars()
    return JobContext(**{f.name: bound.get(f.name, f.default) for f in fields(JobContext)})


def clear() -> None:
    """Drop the job context.

    Celery prefork children are reused across tasks, so without this a stale
    ``job_id`` from a finished task would be attached to the next one.

    structlog's ``clear_contextvars()`` clears every structlog-owned contextvar
    rather than a named subset, which is correct here: this worker binds
    contextvars from nowhere else. Plain ``contextvars`` are untouched, since
    only structlog's prefixed ones are cleared.
    """
    clear_contextvars()
