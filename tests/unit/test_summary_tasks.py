"""Unit tests for the original-footage summary Celery task shell."""

import asyncio

import pytest

import app.tasks  # noqa: F401  (import registers the task modules)
from app.core.exceptions import ExternalServiceError
from app.tasks import summaries as summary_task
from app.tasks.celery_app import celery_app
from app.tasks.summaries import FAILURE_SAFE_MESSAGE, SUMMARY_QUOTA_MESSAGE


def _run(coro):
    """Execute a coroutine on a throwaway loop."""
    return asyncio.run(coro)


class TestTaskRegistration:
    """The task name must match the backend's dispatch constant."""

    def test_summary_task_name(self) -> None:
        assert "app.tasks.summaries.generate_video_summary" in celery_app.tasks


class TestGenerateVideoSummary:
    """The shell reports success or a user-safe failure."""

    def test_returns_ready_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_run(summary_id, client_factory=None):
            return "https://blob/summary.mp4"

        monkeypatch.setattr(summary_task, "run_summary", fake_run)
        monkeypatch.setattr(summary_task, "run_async", _run)

        assert summary_task.generate_video_summary("summary-1") == {"status": "ready"}

    def test_reports_failure_and_reraises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def boom(summary_id, client_factory=None):
            raise RuntimeError("summary ai down")

        failed: list[tuple[str, str]] = []

        async def fake_fail(summary_id, message, client_factory=None):
            failed.append((summary_id, message))

        monkeypatch.setattr(summary_task, "run_summary", boom)
        monkeypatch.setattr(summary_task, "fail_summary", fake_fail)
        monkeypatch.setattr(summary_task, "run_async", _run)

        with pytest.raises(RuntimeError):
            summary_task.generate_video_summary("summary-1")

        assert failed == [("summary-1", FAILURE_SAFE_MESSAGE)]

    def test_reports_quota_failure_with_a_specific_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(summary_id, client_factory=None):
            raise ExternalServiceError(
                "quota",
                service="openai",
                details={"quota_exhausted": True},
            )

        failed: list[tuple[str, str]] = []

        async def fake_fail(summary_id, message, client_factory=None):
            failed.append((summary_id, message))

        monkeypatch.setattr(summary_task, "run_summary", boom)
        monkeypatch.setattr(summary_task, "fail_summary", fake_fail)
        monkeypatch.setattr(summary_task, "run_async", _run)

        with pytest.raises(ExternalServiceError):
            summary_task.generate_video_summary("summary-1")

        assert failed == [("summary-1", SUMMARY_QUOTA_MESSAGE)]
