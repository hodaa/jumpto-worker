"""Unit tests for the lecture-notes Celery task shell."""

import asyncio

import pytest

import app.tasks  # noqa: F401  (import registers the task modules)
from app.core.exceptions import ExternalServiceError
from app.tasks import notes as notes_task
from app.tasks.celery_app import celery_app
from app.tasks.notes import FAILURE_SAFE_MESSAGE, NOTES_QUOTA_MESSAGE


def _run(coro):
    """Execute a coroutine on a throwaway loop."""
    return asyncio.run(coro)


class TestTaskRegistration:
    """The task name must match the backend's dispatch constant."""

    def test_notes_task_name(self) -> None:
        assert "app.tasks.notes.generate_lecture_notes" in celery_app.tasks


class TestGenerateLectureNotes:
    """The shell reports success or a user-safe failure."""

    def test_returns_ready_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_run(note_id, client_factory=None):
            return "https://blob/notes.pdf"

        monkeypatch.setattr(notes_task, "run_notes", fake_run)
        monkeypatch.setattr(notes_task, "run_async", _run)

        assert notes_task.generate_lecture_notes("note-1") == {"status": "ready"}

    def test_reports_failure_and_reraises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def boom(note_id, client_factory=None):
            raise RuntimeError("notes ai down")

        failed: list[tuple[str, str]] = []

        async def fake_fail(note_id, message, client_factory=None):
            failed.append((note_id, message))

        monkeypatch.setattr(notes_task, "run_notes", boom)
        monkeypatch.setattr(notes_task, "fail_notes", fake_fail)
        monkeypatch.setattr(notes_task, "run_async", _run)

        with pytest.raises(RuntimeError):
            notes_task.generate_lecture_notes("note-1")

        assert failed == [("note-1", FAILURE_SAFE_MESSAGE)]

    def test_reports_quota_failure_with_a_specific_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(note_id, client_factory=None):
            raise ExternalServiceError(
                "quota",
                service="openai",
                details={"quota_exhausted": True},
            )

        failed: list[tuple[str, str]] = []

        async def fake_fail(note_id, message, client_factory=None):
            failed.append((note_id, message))

        monkeypatch.setattr(notes_task, "run_notes", boom)
        monkeypatch.setattr(notes_task, "fail_notes", fake_fail)
        monkeypatch.setattr(notes_task, "run_async", _run)

        with pytest.raises(ExternalServiceError):
            notes_task.generate_lecture_notes("note-1")

        assert failed == [("note-1", NOTES_QUOTA_MESSAGE)]
