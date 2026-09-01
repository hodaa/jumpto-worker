"""Unit tests for the worker transcription pipeline."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import ExternalServiceError
from app.models import JobData, TranscriptSubmission
from app.tasks import transcription as transcription_module
from app.tasks.transcription import (
    _build_submission,
    _fetch_transcript_with_retry,
    _live_pipeline_enabled,
    _user_safe_message,
    run_pipeline,
)


def _job() -> JobData:
    """Build a pending job for tests."""
    return JobData(
        job_id="job-1",
        video_id="video-1",
        youtube_video_id="abcde12345",
        youtube_url="https://www.youtube.com/watch?v=abcde12345",
        status="pending",
    )


class TestBuildSubmission:
    """Tests for building a transcript submission payload."""

    def test_build_submission_normalizes_words(self) -> None:
        media = _media()
        transcript = _transcript()

        submission = _build_submission(media, transcript)

        assert isinstance(submission, TranscriptSubmission)
        assert submission.title == media.title
        assert submission.duration_seconds == media.duration_seconds
        assert submission.transcript_text == "Hello, world!"
        assert submission.words[0].word == "hello"
        assert submission.words[1].word == "world"
        assert submission.words[0].word_index == 0

    def test_build_submission_drops_empty_words(self) -> None:
        media = _media()
        transcript = SimpleNamespace(
            language="en",
            text="… —",
            words=[
                SimpleNamespace(word="…", start_time=0.0, end_time=1.0),
                SimpleNamespace(word="hello", start_time=1.0, end_time=2.0),
                SimpleNamespace(word="—", start_time=2.0, end_time=3.0),
            ],
        )

        submission = _build_submission(media, transcript)

        assert len(submission.words) == 1
        assert submission.words[0].word == "hello"
        assert submission.words[0].word_index == 0


class TestRunPipeline:
    """Tests for the full worker pipeline flow."""

    @pytest.mark.asyncio
    async def test_happy_path_calls_all_steps(self, monkeypatch) -> None:
        client = _FakeClient()
        monkeypatch.setattr(transcription_module, "BackendClient", lambda base, key: client)

        settings = _settings(live_calls=False)
        monkeypatch.setattr(transcription_module, "get_settings", lambda: settings)
        monkeypatch.setattr(transcription_module, "_perform_transcription", _perform_mock)

        result = await run_pipeline("job-1")

        assert result["status"] == "completed"
        assert client.calls == ["get_job", "advance", "progress:90", "store", "complete"]
        assert client.failed is False

    @pytest.mark.asyncio
    async def test_reports_progress_at_each_stage(self, monkeypatch) -> None:
        client = _FakeClient()
        monkeypatch.setattr(transcription_module, "BackendClient", lambda base, key: client)

        settings = _settings(live_calls=False)
        monkeypatch.setattr(transcription_module, "get_settings", lambda: settings)

        async def fake_fetch(url: str):
            return _transcript()

        monkeypatch.setattr(transcription_module, "get_media_info", lambda *a: _media())
        monkeypatch.setattr(transcription_module, "_fetch_transcript_with_retry", fake_fetch)

        await run_pipeline("job-1")

        assert client.progress_reports == [
            transcription_module.PROGRESS_MEDIA,
            transcription_module.PROGRESS_TRANSCRIPT,
            transcription_module.PROGRESS_STORE,
        ]

    @pytest.mark.asyncio
    async def test_skips_non_pending_job(self, monkeypatch) -> None:
        client = _FakeClient(status="completed")
        monkeypatch.setattr(transcription_module, "BackendClient", lambda base, key: client)

        result = await run_pipeline("job-1")

        assert result["status"] == "completed"
        assert client.calls == ["get_job"]

    @pytest.mark.asyncio
    async def test_failure_marks_job_failed(self, monkeypatch) -> None:
        client = _FakeClient()
        monkeypatch.setattr(transcription_module, "BackendClient", lambda base, key: client)

        settings = _settings(live_calls=False)
        monkeypatch.setattr(transcription_module, "get_settings", lambda: settings)

        def boom(job, report=None):
            raise RuntimeError("boom")

        monkeypatch.setattr(transcription_module, "_perform_transcription", boom)

        with pytest.raises(RuntimeError):
            await run_pipeline("job-1")

        assert client.failed is True
        assert client.calls[-1] == "fail"

    def test_user_safe_message_maps_errors(self) -> None:
        assert (
            _user_safe_message(RuntimeError("x")) == "Transcription failed. Please try again later."
        )
        assert "timed out" in _user_safe_message(TimeoutError())


class TestFetchTranscriptWithRetry:
    """Tests for the transcript fetch with retry behaviour."""

    @pytest.mark.asyncio
    async def test_uses_captions_fast_path_when_live(self, monkeypatch) -> None:
        settings = _settings(live_calls=True, mode="real")
        monkeypatch.setattr(transcription_module, "get_settings", lambda: settings)

        youtube_provider = AsyncMock()
        youtube_provider.fetch.return_value = _transcript()
        monkeypatch.setattr(
            transcription_module,
            "YouTubeCaptionTranscriptProvider",
            lambda: youtube_provider,
        )

        result = await _fetch_transcript_with_retry("https://youtu.be/abcde12345")

        assert result.text == "Hello, world!"
        youtube_provider.fetch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_audio_provider_after_captions_fail(self, monkeypatch) -> None:
        settings = _settings(live_calls=True, mode="real")
        monkeypatch.setattr(transcription_module, "get_settings", lambda: settings)

        def fail(url):
            raise ExternalServiceError("No captions", service="youtube-captions")

        youtube_provider = AsyncMock()
        youtube_provider.fetch.side_effect = fail
        monkeypatch.setattr(
            transcription_module, "YouTubeCaptionTranscriptProvider", lambda: youtube_provider
        )

        fallback = AsyncMock()
        fallback.fetch.return_value = _transcript()
        monkeypatch.setattr(transcription_module, "get_transcript_provider", lambda: fallback)

        result = await _fetch_transcript_with_retry("https://youtu.be/abcde12345")

        assert result.text == "Hello, world!"
        fallback.fetch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_after_retries_exhausted(self, monkeypatch) -> None:
        settings = _settings(live_calls=False)
        monkeypatch.setattr(transcription_module, "get_settings", lambda: settings)
        monkeypatch.setattr(transcription_module, "_RETRY_ATTEMPTS", 2)
        monkeypatch.setattr(transcription_module, "_RETRY_DELAY_SECONDS", 0)

        def fail(url):
            raise ExternalServiceError("boom", service="assemblyai")

        fallback = AsyncMock()
        fallback.fetch.side_effect = fail
        monkeypatch.setattr(transcription_module, "get_transcript_provider", lambda: fallback)

        with pytest.raises(ExternalServiceError):
            await _fetch_transcript_with_retry("https://youtu.be/abcde12345")

        assert fallback.fetch.await_count == 2


class TestLivePipelineEnabled:
    """Tests for the live-pipeline toggle."""

    def test_disabled_when_live_calls_off(self, monkeypatch) -> None:
        monkeypatch.setattr(
            transcription_module, "get_settings", lambda: _settings(live_calls=False)
        )

        assert _live_pipeline_enabled() is False

    def test_disabled_in_fake_mode(self, monkeypatch) -> None:
        settings = SimpleNamespace(
            jumpto_live_external_calls=True,
            jumpto_transcript_mode="fake",
        )
        monkeypatch.setattr(transcription_module, "get_settings", lambda: settings)

        assert _live_pipeline_enabled() is False

    def test_enabled_with_live_calls_and_real_mode(self, monkeypatch) -> None:
        settings = SimpleNamespace(
            jumpto_live_external_calls=True,
            jumpto_transcript_mode="real",
        )
        monkeypatch.setattr(transcription_module, "get_settings", lambda: settings)

        assert _live_pipeline_enabled() is True


class TestFailureMarking:
    """Tests for failure reporting to the backend."""

    @pytest.mark.asyncio
    async def test_fail_job_is_best_effort_when_marking_fails(self, monkeypatch) -> None:
        class _RaisingClient(_FakeClient):
            async def fail_job(self, job_id: str, error: str) -> None:
                self.calls.append("fail")
                raise RuntimeError("marking failed")

        client = _RaisingClient()
        monkeypatch.setattr(transcription_module, "BackendClient", lambda base, key: client)
        settings = _settings(live_calls=False)
        monkeypatch.setattr(transcription_module, "get_settings", lambda: settings)

        def boom(job, report=None):
            raise RuntimeError("transcription boom")

        monkeypatch.setattr(transcription_module, "_perform_transcription", boom)

        with pytest.raises(RuntimeError):
            await run_pipeline("job-1")

        assert client.calls[-1] == "fail"


class _FakeClient:
    """Minimal fake BackendClient that records its calls."""

    def __init__(self, status: str = "pending") -> None:
        self.status = status
        self.calls: list[str] = []
        self.progress_reports: list[int] = []
        self.failed = False

    async def get_job(self, job_id: str) -> JobData:
        self.calls.append("get_job")
        job = _job()
        if self.status != "pending":
            return JobData(**{**job.__dict__, "status": self.status})
        return job

    async def advance_job(self, job_id: str) -> None:
        self.calls.append("advance")

    async def report_progress(self, job_id: str, progress: int) -> None:
        self.calls.append(f"progress:{progress}")
        self.progress_reports.append(progress)

    async def store_transcript(self, job_id: str, submission) -> None:
        self.calls.append("store")

    async def complete_job(self, job_id: str) -> None:
        self.calls.append("complete")

    async def fail_job(self, job_id: str, error: str) -> None:
        self.calls.append("fail")
        self.failed = True


def _media():
    """Build a minimal fake media info object."""
    return SimpleNamespace(title="My Video", duration_seconds=240)


def _transcript():
    """Build a fake transcript with word data."""
    return SimpleNamespace(
        language="en",
        text="Hello, world!",
        words=[
            SimpleNamespace(word="Hello,", start_time=0.0, end_time=0.5),
            SimpleNamespace(word="world!", start_time=0.5, end_time=1.0),
        ],
    )


async def _perform_mock(job, report=None):
    """Mock the transcription step to return a submission."""
    return _build_submission(_media(), _transcript())


def _settings(*, live_calls: bool, mode: str = "fake") -> SimpleNamespace:
    """Build a minimal settings object."""
    return SimpleNamespace(
        backend_url="http://backend.test:8000",
        internal_api_key="key",
        job_timeout_seconds=600,
        jumpto_live_external_calls=live_calls,
        jumpto_transcript_mode=mode,
    )
