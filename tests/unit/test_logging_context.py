"""Unit tests for the Celery signal wiring of the job-scoped log context.

The context itself is covered in ``test_job_context``; what matters here is that
Celery's lifecycle puts it in place and takes it away again, so a task's records
carry its job and the next task on the same reused prefork child does not
inherit it.
"""

from types import SimpleNamespace

import pytest
import sentry_sdk
import structlog
from celery.signals import task_postrun, task_prerun
from structlog.contextvars import get_contextvars

from app.core.job_context import CONTEXT_KEYS
from app.tasks.celery_app import celery_app


@pytest.fixture(autouse=True)
def _clean_context():
    """Keep each test independent of whatever the previous one bound."""
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()


@pytest.fixture
def sentry_tags(monkeypatch) -> dict:
    """Capture Sentry tags without needing a live DSN."""
    tags: dict = {}
    monkeypatch.setattr(sentry_sdk, "set_tag", lambda k, v: tags.__setitem__(k, v))
    # Scope uses __slots__, so the method is patched on the class, not the
    # instance — this is the exact call ``_untag_job_id`` makes.
    monkeypatch.setattr(
        type(sentry_sdk.get_isolation_scope()),
        "remove_tag",
        lambda self, key: tags.pop(key, None),
    )
    return tags


def _rendered_record(**event) -> dict:
    """Merge bound context into an event exactly as the worker renders one."""
    return structlog.contextvars.merge_contextvars(None, "info", event)


def _task(args: tuple = (), retries: int = 0) -> SimpleNamespace:
    """Build a stand-in for a Celery task invocation."""
    return SimpleNamespace(request=SimpleNamespace(args=args, retries=retries))


class TestTaskPrerun:
    """The task starts with its own identity, nothing inherited."""

    def test_binds_job_and_attempt(self) -> None:
        task_prerun.send(sender=celery_app, task=_task(args=("job-42",), retries=2))

        record = _rendered_record(event="boom")

        assert record["job_id"] == "job-42"
        assert record["retry_attempt"] == 2

    def test_task_without_a_request_binds_defaults(self) -> None:
        task_prerun.send(sender=celery_app, task=None)

        assert get_contextvars()["retry_attempt"] == 0

    def test_unrecognised_message_binds_no_job_id(self) -> None:
        """Absent rather than blank, so it cannot match a ``job_id:`` filter."""
        task_prerun.send(sender=celery_app, task=_task())

        assert "job_id" not in get_contextvars()

    def test_overrides_a_context_left_by_the_previous_task(self) -> None:
        """A crash mid-task must not leave the next task mislabelled."""
        task_prerun.send(sender=celery_app, task=_task(args=("job-first",)))

        task_prerun.send(sender=celery_app, task=_task(args=("job-second",)))

        assert get_contextvars()["job_id"] == "job-second"


class TestTaskPostrun:
    """Prefork children are reused, so a stale job id must not leak."""

    def test_clears_the_context(self) -> None:
        task_prerun.send(sender=celery_app, task=_task(args=("job-42",)))

        task_postrun.send(sender=celery_app, task=_task())

        assert not set(CONTEXT_KEYS) & set(get_contextvars())

    def test_next_task_does_not_inherit_the_previous_job(self) -> None:
        task_prerun.send(sender=celery_app, task=_task(args=("job-first",)))
        task_postrun.send(sender=celery_app, task=_task())

        task_prerun.send(sender=celery_app, task=_task(args=("job-second",)))

        assert get_contextvars()["job_id"] == "job-second"


class TestSentryJobTag:
    """Celery-captured failures never pass through a logger call."""

    def test_prerun_tags_the_job_id(self, sentry_tags: dict) -> None:
        task_prerun.send(sender=celery_app, task=_task(args=("job-42",)))

        assert sentry_tags == {"job_id": "job-42"}

    def test_postrun_removes_the_tag(self, sentry_tags: dict) -> None:
        """The Sentry scope is process-wide, so a stale tag would misattribute."""
        task_prerun.send(sender=celery_app, task=_task(args=("job-42",)))

        task_postrun.send(sender=celery_app, task=_task())

        assert sentry_tags == {}

    def test_unrecognised_message_sets_no_tag(self, sentry_tags: dict) -> None:
        task_prerun.send(sender=celery_app, task=_task())

        assert sentry_tags == {}


class TestContextIsWiredIntoTheRenderer:
    """Without ``merge_contextvars`` every other test here would still pass."""

    def test_configure_logging_merges_contextvars(self) -> None:
        from app.core.logging import configure_logging

        configure_logging()
        try:
            processors = structlog.get_config()["processors"]
        finally:
            configure_logging()

        assert structlog.contextvars.merge_contextvars in processors
