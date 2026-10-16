"""Unit tests for the download and format-probe Celery task shells.

The shells run their orchestration through ``run_async``; these tests swap
that for a plain ``asyncio.run`` so the task body executes synchronously.
"""

import asyncio

import pytest

import app.tasks  # noqa: F401  (import registers the task modules)
from app.tasks import downloads as downloads_task
from app.tasks import formats as formats_task
from app.tasks.celery_app import celery_app
from app.tasks.downloads import FAILURE_SAFE_MESSAGE


def _run(coro):
    """Execute a coroutine on a throwaway loop."""
    return asyncio.run(coro)


class TestTaskRegistration:
    """The tasks must be registered under the names the backend dispatches."""

    def test_download_task_name(self) -> None:
        assert "app.tasks.downloads.download_video" in celery_app.tasks

    def test_formats_task_name(self) -> None:
        assert "app.tasks.formats.probe_video_formats" in celery_app.tasks


class TestDownloadVideoTask:
    """download_video reports success or fails the download on the backend."""

    def test_returns_ready_on_success(self, monkeypatch) -> None:
        async def fake_run(download_id, client_factory=None):
            return "https://blob/x.mp4"

        monkeypatch.setattr(downloads_task, "run_download", fake_run)
        monkeypatch.setattr(downloads_task, "run_async", _run)

        assert downloads_task.download_video("d-1") == {"status": "ready"}

    def test_reports_safe_failure_and_reraises(self, monkeypatch) -> None:
        async def boom(download_id, client_factory=None):
            raise RuntimeError("yt-dlp exploded")

        failed: list[tuple[str, str]] = []

        async def fake_fail(download_id, message, client_factory=None):
            failed.append((download_id, message))

        monkeypatch.setattr(downloads_task, "run_download", boom)
        monkeypatch.setattr(downloads_task, "fail_download", fake_fail)
        monkeypatch.setattr(downloads_task, "run_async", _run)

        with pytest.raises(RuntimeError):
            downloads_task.download_video("d-1")

        assert failed == [("d-1", FAILURE_SAFE_MESSAGE)]


class TestProbeVideoFormatsTask:
    """probe_video_formats caches the ladder and swallows probe failures."""

    def test_returns_probed_heights(self, monkeypatch) -> None:
        async def fake_probe(youtube_video_id, youtube_url, client_factory=None):
            return [1080, 720]

        monkeypatch.setattr(formats_task, "probe_and_store_formats", fake_probe)
        monkeypatch.setattr(formats_task, "run_async", _run)

        result = formats_task.probe_video_formats("abcde12345", "https://youtu.be/x")

        assert result == {"status": "completed", "heights": [1080, 720]}

    def test_failed_probe_reports_failed_without_raising(self, monkeypatch) -> None:
        async def boom(youtube_video_id, youtube_url, client_factory=None):
            raise RuntimeError("probe exploded")

        monkeypatch.setattr(formats_task, "probe_and_store_formats", boom)
        monkeypatch.setattr(formats_task, "run_async", _run)

        result = formats_task.probe_video_formats("abcde12345", "https://youtu.be/x")

        assert result == {"status": "failed"}
