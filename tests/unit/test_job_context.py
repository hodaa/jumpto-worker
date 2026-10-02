"""Unit tests for the job-scoped log context.

These fields exist so a Sentry error can be traced back to one job, one video
and one attempt without re-running anything. Their absence is silent: every log
line still renders, it just stops saying which job it belongs to. So these tests
pin the behaviour rather than the implementation.
"""

import contextvars
from dataclasses import make_dataclass
from types import SimpleNamespace

import pytest
import structlog
from structlog.contextvars import get_contextvars

from app.core import logging as worker_logging
from app.core.job_context import CONTEXT_KEYS, JobContext, bind, clear, current

# A contextvar this project does not own, to prove clearing is scoped.
_FOREIGN = contextvars.ContextVar("plain_contextvar", default=None)


@pytest.fixture(autouse=True)
def _clean_context():
    """Keep each test independent of whatever the previous one bound."""
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()


def _rendered_record(**event) -> dict:
    """Merge bound context into an event exactly as the worker renders one."""
    return structlog.contextvars.merge_contextvars(None, "info", event)


class TestJobContextShape:
    """JobContext is the single definition of what identifies a job."""

    def test_defaults_are_unpopulated(self) -> None:
        assert JobContext() == JobContext("", 0, "", "", "")

    def test_context_keys_are_derived_from_the_dataclass(self) -> None:
        """Keys must not be a second, hand-maintained list of the fields."""
        from dataclasses import fields

        assert set(CONTEXT_KEYS) == {f.name for f in fields(JobContext)}

    def test_merged_applies_updates(self) -> None:
        merged = JobContext(job_id="job-1").merged(video_id="video-9", youtube_url="u")

        assert merged == JobContext(job_id="job-1", video_id="video-9", youtube_url="u")

    def test_merged_keeps_existing_identity(self) -> None:
        """The video phase must not drop what the Celery phase bound."""
        merged = JobContext(job_id="job-1", retry_attempt=2).merged(video_id="video-9")

        assert merged.job_id == "job-1"
        assert merged.retry_attempt == 2

    def test_merged_ignores_empty_updates(self) -> None:
        """An unknown video must not overwrite known identity with a blank."""
        merged = JobContext(job_id="job-1").merged(video_id="", youtube_url="")

        assert merged == JobContext(job_id="job-1")

    def test_merged_does_not_mutate_the_original(self) -> None:
        """Frozen, so one task's context cannot change under a concurrent one."""
        original = JobContext(job_id="job-1")

        original.merged(video_id="video-9")

        assert original.video_id == ""


class TestBind:
    """Binding is what ``merge_contextvars`` later renders."""

    def test_binds_identity_and_attempt(self) -> None:
        bind(JobContext(job_id="job-1", retry_attempt=3))

        record = _rendered_record(event="boom")

        assert record["job_id"] == "job-1"
        assert record["retry_attempt"] == 3

    def test_retry_attempt_defaults_to_first_attempt(self) -> None:
        bind(JobContext(job_id="job-1"))

        assert get_contextvars()["retry_attempt"] == 0

    def test_video_identity_is_bound_by_a_later_phase(self) -> None:
        bind(JobContext(job_id="job-1"))
        bind(current().merged(video_id="video-9", youtube_video_id="yt-9"))

        record = _rendered_record(event="boom")

        assert record["video_id"] == "video-9"
        assert record["youtube_video_id"] == "yt-9"
        assert record["job_id"] == "job-1"

    def test_unpopulated_fields_stay_absent(self) -> None:
        """Absent, not empty: an empty string would match a Sentry filter."""
        bind(JobContext(job_id="job-1"))

        assert "video_id" not in _rendered_record(event="boom")

    def test_every_owned_key_is_renderable(self) -> None:
        """Guards against a key being bound under a name the renderer drops."""
        bind(JobContext(**{key: f"v-{key}" for key in CONTEXT_KEYS}))

        assert set(CONTEXT_KEYS) <= set(_rendered_record(event="boom"))


class TestCurrent:
    """``current`` reads back what is bound, defaulting what is not."""

    def test_round_trips_a_bound_context(self) -> None:
        bound = JobContext(job_id="job-1", retry_attempt=2, video_id="video-9")

        bind(bound)

        assert current() == bound

    def test_returns_defaults_when_nothing_is_bound(self) -> None:
        assert current() == JobContext()


class TestClear:
    """Prefork children are reused, so a stale job id must not leak."""

    def test_removes_every_owned_key(self) -> None:
        bind(JobContext(job_id="job-1", retry_attempt=2, video_id="video-9"))

        clear()

        assert not set(CONTEXT_KEYS) & set(get_contextvars())

    def test_leaves_non_structlog_contextvars_untouched(self) -> None:
        """structlog clears its own prefixed vars only, so ours survive."""
        token = _FOREIGN.set("keep-me")
        try:
            bind(JobContext(job_id="job-1"))

            clear()

            assert _FOREIGN.get() == "keep-me"
        finally:
            _FOREIGN.reset(token)


class TestExtensibility:
    """The reason for the shape: a new identifier must be plumbing-only."""

    def test_a_new_field_needs_no_change_to_the_binder(self) -> None:
        """This is the property the dataclass exists to provide.

        ``bind`` derives its keys from the instance, so an identifier added
        later is carried without touching ``bind``, ``current``, ``clear`` or any
        call site. If this ever fails, the binder has started enumerating
        fields and extending the context has become a modification.
        """
        Extended = make_dataclass(
            "ExtendedJobContext",
            [("channel_id", str, "")],
            bases=(JobContext,),
            frozen=True,
        )

        bind(Extended(job_id="job-1", channel_id="c-1"))

        assert _rendered_record(event="boom")["channel_id"] == "c-1"


class TestHostnameIsARecordAttribute:
    """Hostname is process-level, not job identity, so it is a processor."""

    def test_processor_adds_the_hostname(self) -> None:
        record = worker_logging.add_hostname(None, "info", {"event": "x"})

        assert record["hostname"]

    def test_hostname_is_resolved_once_at_import(self) -> None:
        assert worker_logging._HOSTNAME

    def test_configure_logging_installs_the_processor(self) -> None:
        worker_logging.configure_logging()
        try:
            processors = structlog.get_config()["processors"]
        finally:
            worker_logging.configure_logging()

        assert worker_logging.add_hostname in processors

    def test_hostname_renders_without_any_job_bound(self) -> None:
        """Broker reconnects happen outside a job and still need the worker id."""
        worker_logging.configure_logging()
        try:
            chain = structlog.get_config()["processors"]
        finally:
            worker_logging.configure_logging()

        renderers = (structlog.dev.ConsoleRenderer, structlog.processors.JSONRenderer)
        event_dict: dict = {"event": "broker reconnect"}
        for processor in chain:
            if isinstance(processor, renderers):
                break
            event_dict = processor(SimpleNamespace(name="test"), "info", event_dict)

        assert event_dict["hostname"] == worker_logging._HOSTNAME
