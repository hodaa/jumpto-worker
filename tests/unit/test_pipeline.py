"""Unit tests for the worker transcription pipeline."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings, _live_pipeline_enabled
from app.core.exceptions import ExternalServiceError
from app.models import JobData, TranscriptSubmission
from app.providers import (
    TranscriptData,
    TranscriptJobPending,
    TranscriptWordData,
    VideoTranscriptResult,
    VidWordsResult,
)
from app.providers import local as local_module
from app.providers.vidwords import VidWordsPermanentError
from app.tasks import transcription as transcription_module
from app.tasks.transcription import (
    _build_submission,
    _job_retry_countdown,
    _perform_transcription,
    _user_safe_message,
    download_and_transcribe,
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
        assert client.calls == ["get_job", "advance", "store", "complete"]
        assert client.failed is False

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

        def boom(job, resume_token="", resume_provider=""):
            raise RuntimeError("boom")

        monkeypatch.setattr(transcription_module, "_perform_transcription", boom)

        with pytest.raises(RuntimeError):
            await run_pipeline("job-1")

        assert client.failed is True
        assert client.calls[-1] == "fail"

    @pytest.mark.asyncio
    async def test_first_attempt_advances_then_raises_pending(self, monkeypatch) -> None:
        client = _FakeClient()
        monkeypatch.setattr(transcription_module, "BackendClient", lambda base, key: client)

        def pending(job, resume_token="", resume_provider=""):
            raise TranscriptJobPending(provider="supadata", resume_token="job-x")

        monkeypatch.setattr(transcription_module, "_perform_transcription", pending)

        with pytest.raises(TranscriptJobPending):
            await run_pipeline("job-1")

        assert client.calls == ["get_job", "advance"]
        assert client.failed is False

    @pytest.mark.asyncio
    async def test_resume_pending_is_not_advanced_or_failed(self, monkeypatch) -> None:
        client = _FakeClient(status="processing")
        monkeypatch.setattr(transcription_module, "BackendClient", lambda base, key: client)

        def pending(job, resume_token="", resume_provider=""):
            raise TranscriptJobPending(provider="supadata", resume_token="job-x")

        monkeypatch.setattr(transcription_module, "_perform_transcription", pending)

        with pytest.raises(TranscriptJobPending):
            await run_pipeline("job-1", resume_token="job-x", resume_provider="supadata")

        assert client.calls == ["get_job"]
        assert client.failed is False

    def test_user_safe_message_maps_errors(self) -> None:
        assert (
            _user_safe_message(RuntimeError("x")) == "Transcription failed. Please try again later."
        )
        assert "timed out" in _user_safe_message(TimeoutError())


class TestFetchTranscriptWithRetry:
    """Tests for the local transcript fetch with retry behaviour."""

    @pytest.mark.asyncio
    async def test_uses_captions_fast_path_when_live(self, monkeypatch) -> None:
        settings = _settings(live_calls=True, mode="real")
        monkeypatch.setattr(local_module, "get_settings", lambda: settings)

        youtube_provider = AsyncMock()
        youtube_provider.fetch.return_value = _transcript()
        monkeypatch.setattr(
            local_module,
            "YouTubeCaptionTranscriptProvider",
            lambda: youtube_provider,
        )

        info = {"duration": 240}
        result = await local_module._fetch_transcript_with_retry(
            "https://youtu.be/abcde12345", info
        )

        assert result.text == "Hello, world!"
        youtube_provider.fetch.assert_awaited_once_with("https://youtu.be/abcde12345", info=info)

    @pytest.mark.asyncio
    async def test_captions_fast_path_without_info_fetches_its_own(self, monkeypatch) -> None:
        settings = _settings(live_calls=True, mode="real")
        monkeypatch.setattr(local_module, "get_settings", lambda: settings)

        youtube_provider = AsyncMock()
        youtube_provider.fetch.return_value = _transcript()
        monkeypatch.setattr(
            local_module,
            "YouTubeCaptionTranscriptProvider",
            lambda: youtube_provider,
        )

        result = await local_module._fetch_transcript_with_retry("https://youtu.be/abcde12345")

        assert result.text == "Hello, world!"
        youtube_provider.fetch.assert_awaited_once_with("https://youtu.be/abcde12345", info=None)

    @pytest.mark.asyncio
    async def test_falls_back_to_audio_provider_after_captions_fail(self, monkeypatch) -> None:
        settings = _settings(live_calls=True, mode="real")
        monkeypatch.setattr(local_module, "get_settings", lambda: settings)

        def fail(url, info=None):
            raise ExternalServiceError("No captions", service="youtube-captions")

        youtube_provider = AsyncMock()
        youtube_provider.fetch.side_effect = fail
        monkeypatch.setattr(
            local_module, "YouTubeCaptionTranscriptProvider", lambda: youtube_provider
        )

        fallback = AsyncMock()
        fallback.fetch.return_value = _transcript()
        monkeypatch.setattr(local_module, "get_transcript_provider", lambda: fallback)

        result = await local_module._fetch_transcript_with_retry("https://youtu.be/abcde12345")

        assert result.text == "Hello, world!"
        fallback.fetch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_captionless_video_falls_back_to_audio(self, monkeypatch) -> None:
        """A fully caption-less video raises in the real caption provider,
        then falls back to the audio (Assembly) transcript path."""
        settings = _settings(live_calls=True, mode="real")
        monkeypatch.setattr(local_module, "get_settings", lambda: settings)

        info = {"id": "abcde12345", "automatic_captions": {}, "subtitles": {}}

        fallback = AsyncMock()
        fallback.fetch.return_value = _transcript()
        monkeypatch.setattr(local_module, "get_transcript_provider", lambda: fallback)

        result = await local_module._fetch_transcript_with_retry(
            "https://youtu.be/abcde12345", info
        )

        assert result.text == "Hello, world!"
        fallback.fetch.assert_awaited_once()
        fallback.fetch.assert_awaited_with("https://youtu.be/abcde12345", info=info)

    @pytest.mark.asyncio
    async def test_raises_after_retries_exhausted(self, monkeypatch) -> None:
        settings = _settings(live_calls=False)
        monkeypatch.setattr(local_module, "get_settings", lambda: settings)
        monkeypatch.setattr(local_module, "_RETRY_ATTEMPTS", 2)
        monkeypatch.setattr(local_module, "_RETRY_DELAY_SECONDS", 0)

        def fail(url, info=None):
            raise ExternalServiceError("boom", service="assemblyai")

        fallback = AsyncMock()
        fallback.fetch.side_effect = fail
        monkeypatch.setattr(local_module, "get_transcript_provider", lambda: fallback)

        with pytest.raises(ExternalServiceError):
            await local_module._fetch_transcript_with_retry("https://youtu.be/abcde12345")

        assert fallback.fetch.await_count == 2


class TestLivePipelineEnabled:
    """Tests for the live-pipeline toggle."""

    def test_disabled_when_live_calls_off(self) -> None:
        assert _live_pipeline_enabled(_settings(live_calls=False)) is False

    def test_disabled_in_fake_mode(self) -> None:
        assert _live_pipeline_enabled(_settings(live_calls=True, mode="fake")) is False

    def test_enabled_with_live_calls_and_real_mode(self) -> None:
        assert _live_pipeline_enabled(_settings(live_calls=True, mode="real")) is True


class TestPerformTranscription:
    """Tests for the configured-provider-first, yt-dlp-fallback flow."""

    @staticmethod
    def _fallback(result=None, error=None) -> SimpleNamespace:
        if error is not None:
            return SimpleNamespace(name="yt-dlp", fetch=AsyncMock(side_effect=error))
        return SimpleNamespace(
            name="yt-dlp", fetch=AsyncMock(return_value=result or _ytdlp_result())
        )

    @pytest.mark.asyncio

    async def test_configured_provider_used_when_it_returns_result(self, monkeypatch) -> None:
        provider = SimpleNamespace(name="vidwords", fetch=AsyncMock(return_value=_cloud_result()))
        monkeypatch.setattr(transcription_module, "_configured_provider", lambda: provider)

        submission = await _perform_transcription(_job())

        assert submission.title == "Me at the zoo"
        assert submission.duration_seconds == 19
        assert submission.language == "en"
        assert submission.provider == "vidwords"
        provider.fetch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_ytdlp_when_configured_provider_misses(self, monkeypatch) -> None:
        provider = SimpleNamespace(name="vidwords", fetch=AsyncMock(return_value=None))
        ytdlp = self._fallback()
        monkeypatch.setattr(transcription_module, "_configured_provider", lambda: provider)
        monkeypatch.setattr(transcription_module, "YtDlpTranscriptStrategy", lambda: ytdlp)

        submission = await _perform_transcription(_job())

        assert submission.provider == "yt-dlp"
        assert submission.title == "Fallback Video"
        assert submission.transcript_text == "Hello, world!"
        provider.fetch.assert_awaited_once()
        ytdlp.fetch.assert_awaited_once()

    @pytest.mark.asyncio

    async def test_permanent_provider_error_does_not_fall_through(self, monkeypatch) -> None:
        provider = SimpleNamespace(
            name="vidwords",
            fetch=AsyncMock(
                side_effect=VidWordsPermanentError(
                    "invalid credentials", service="vidwords"
                )
            ),
        )
        ytdlp = self._fallback()
        monkeypatch.setattr(transcription_module, "_configured_provider", lambda: provider)
        monkeypatch.setattr(transcription_module, "YtDlpTranscriptStrategy", lambda: ytdlp)

        with pytest.raises(VidWordsPermanentError):
            await _perform_transcription(_job())

        ytdlp.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resume_token_routes_to_its_provider(self, monkeypatch) -> None:
        provider = SimpleNamespace(name="supadata", fetch=AsyncMock(return_value=_cloud_result()))
        ytdlp = self._fallback()
        monkeypatch.setattr(transcription_module, "_configured_provider", lambda: provider)
        monkeypatch.setattr(transcription_module, "YtDlpTranscriptStrategy", lambda: ytdlp)

        submission = await _perform_transcription(
            _job(), resume_token="job-x", resume_provider="supadata"
        )

        assert submission.provider == "supadata"
        provider.fetch.assert_awaited_once_with(
            "https://www.youtube.com/watch?v=abcde12345", "abcde12345", resume_token="job-x"
        )
        ytdlp.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ytdlp_fallback_on_error(self, monkeypatch) -> None:
        provider = SimpleNamespace(
            name="vidwords",
            fetch=AsyncMock(side_effect=ExternalServiceError("oops", service="vidwords")),
        )
        ytdlp = self._fallback()
        monkeypatch.setattr(transcription_module, "_configured_provider", lambda: provider)
        monkeypatch.setattr(transcription_module, "YtDlpTranscriptStrategy", lambda: ytdlp)

        submission = await _perform_transcription(_job())

        assert submission.provider == "yt-dlp"
        ytdlp.fetch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_configured_provider_uses_ytdlp_only(self, monkeypatch) -> None:
        ytdlp = self._fallback()
        monkeypatch.setattr(transcription_module, "_configured_provider", lambda: None)
        monkeypatch.setattr(transcription_module, "YtDlpTranscriptStrategy", lambda: ytdlp)

        submission = await _perform_transcription(_job())

        assert submission.provider == "yt-dlp"
        ytdlp.fetch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ytdlp_default_does_not_run_fallback_twice(self, monkeypatch) -> None:
        ytdlp = SimpleNamespace(name="yt-dlp", fetch=AsyncMock(return_value=_ytdlp_result()))
        unused = SimpleNamespace(name="yt-dlp", fetch=AsyncMock())
        monkeypatch.setattr(transcription_module, "_configured_provider", lambda: ytdlp)
        monkeypatch.setattr(transcription_module, "YtDlpTranscriptStrategy", lambda: unused)

        submission = await _perform_transcription(_job())

        assert submission.provider == "yt-dlp"
        ytdlp.fetch.assert_awaited_once()
        unused.fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resume_token_routed_only_to_its_provider(self, monkeypatch) -> None:
        provider = SimpleNamespace(name="vidwords", fetch=AsyncMock(return_value=None))
        ytdlp = self._fallback()
        monkeypatch.setattr(transcription_module, "_configured_provider", lambda: provider)
        monkeypatch.setattr(transcription_module, "YtDlpTranscriptStrategy", lambda: ytdlp)

        submission = await _perform_transcription(
            _job(), resume_token="job-x", resume_provider="yt-dlp"
        )

        assert submission.provider == "yt-dlp"
        provider.fetch.assert_awaited_once_with(
            "https://www.youtube.com/watch?v=abcde12345", "abcde12345", resume_token=""
        )
        ytdlp.fetch.assert_awaited_once_with(
            "https://www.youtube.com/watch?v=abcde12345", "abcde12345", resume_token="job-x"
        )

    @pytest.mark.asyncio
    async def test_all_miss_raises_instead_of_completing_empty(self, monkeypatch) -> None:
        provider = SimpleNamespace(name="vidwords", fetch=AsyncMock(return_value=None))
        ytdlp = SimpleNamespace(name="yt-dlp", fetch=AsyncMock(return_value=None))
        monkeypatch.setattr(transcription_module, "_configured_provider", lambda: provider)
        monkeypatch.setattr(transcription_module, "YtDlpTranscriptStrategy", lambda: ytdlp)

        with pytest.raises(ExternalServiceError):
            await _perform_transcription(_job())


class TestConfiguredProvider:
    """Tests for the single .env-configured provider selection."""

    @staticmethod
    def _name(monkeypatch, settings) -> str | None:
        monkeypatch.setattr(transcription_module, "get_settings", lambda: settings)
        provider = transcription_module._configured_provider()
        return provider.name if provider is not None else None

    def test_none_when_no_default(self, monkeypatch) -> None:
        assert self._name(monkeypatch, _full_settings(default_video_provider="")) is None

    def test_returns_configured_cloud_provider(self, monkeypatch) -> None:
        assert (
            self._name(monkeypatch, _full_settings(default_video_provider="vidwords")) == "vidwords"
        )

    def test_default_is_case_insensitive(self, monkeypatch) -> None:
        assert (
            self._name(monkeypatch, _full_settings(default_video_provider="VidWords")) == "vidwords"
        )

    def test_ytdlp_default_stays_available(self, monkeypatch) -> None:
        assert self._name(monkeypatch, _full_settings(default_video_provider="yt-dlp")) == "yt-dlp"

    def test_cloud_provider_skipped_when_not_live(self, monkeypatch) -> None:
        settings = _full_settings(
            default_video_provider="vidwords", jumpto_live_external_calls=False
        )
        assert self._name(monkeypatch, settings) is None

    def test_none_for_unconfigured_default(self, monkeypatch) -> None:
        settings = _full_settings(default_video_provider="vidwords", vidwords_api_key="")
        assert self._name(monkeypatch, settings) is None

    def test_none_for_unknown_default(self, monkeypatch) -> None:
        assert self._name(monkeypatch, _full_settings(default_video_provider="bogus")) is None


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

        def boom(job, resume_token="", resume_provider=""):
            raise RuntimeError("transcription boom")

        monkeypatch.setattr(transcription_module, "_perform_transcription", boom)

        with pytest.raises(RuntimeError):
            await run_pipeline("job-1")

        assert client.calls[-1] == "fail"


class TestDownloadAndTranscribe:
    """Tests for the Celery task's retry-based async-job wait."""

    async def _pending_pipeline(
        self, job_id: str, resume_token: str = "", resume_provider: str = ""
    ) -> None:
        raise TranscriptJobPending(provider="supadata", resume_token="job-x")

    async def _non_resumable_pending_pipeline(
        self, job_id: str, resume_token: str = "", resume_provider: str = ""
    ) -> None:
        raise TranscriptJobPending(
            provider="transcriptfetch", resume_token="job-x", resumable=False
        )

    def test_retries_pending_job_with_backoff(self, monkeypatch) -> None:
        monkeypatch.setattr(transcription_module, "run_pipeline", self._pending_pipeline)

        stub = _StubTask(retries=0)
        with pytest.raises(_RetryRaised):
            download_and_transcribe.run.__func__(stub, "job-1")

        max_retries, countdown, args = stub.retry_call
        assert max_retries == transcription_module._CLOUD_JOB_ATTEMPTS
        assert countdown == 2
        assert args == ["job-1", "job-x", "supadata"]

    def test_escapes_successfully_when_job_completes(self, monkeypatch) -> None:
        async def completed(job_id: str, resume_token: str = "", resume_provider: str = "") -> dict:
            return {"status": "completed"}

        monkeypatch.setattr(transcription_module, "run_pipeline", completed)
        stub = _StubTask(retries=3)

        result = download_and_transcribe.run.__func__(stub, "job-1", "job-x", "supadata")

        assert result == {"status": "completed"}
        assert stub.retry_call is None

    def test_marks_job_failed_once_wait_budget_exhausted(self, monkeypatch) -> None:
        monkeypatch.setattr(transcription_module, "run_pipeline", self._pending_pipeline)

        client = _FakeClient()
        monkeypatch.setattr(transcription_module, "BackendClient", lambda base, key: client)
        settings = _settings(live_calls=False)
        monkeypatch.setattr(transcription_module, "get_settings", lambda: settings)

        stub = _StubTask(retries=transcription_module._CLOUD_JOB_ATTEMPTS - 1)
        with pytest.raises(TranscriptJobPending):
            download_and_transcribe.run.__func__(stub, "job-1", "job-x", "supadata")

        assert stub.retry_call is None
        assert client.failed is True
        assert client.calls == ["fail"]

    def test_non_resumable_pending_fails_job_without_retry(self, monkeypatch) -> None:
        monkeypatch.setattr(
            transcription_module, "run_pipeline", self._non_resumable_pending_pipeline
        )

        client = _FakeClient()
        monkeypatch.setattr(transcription_module, "BackendClient", lambda base, key: client)
        settings = _settings(live_calls=False)
        monkeypatch.setattr(transcription_module, "get_settings", lambda: settings)

        stub = _StubTask(retries=0)
        with pytest.raises(TranscriptJobPending):
            download_and_transcribe.run.__func__(stub, "job-1")

        assert stub.retry_call is None
        assert client.failed is True
        assert client.calls == ["fail"]

    def test_retries_failed_job_routes_resume_token_to_its_provider(self, monkeypatch) -> None:
        captured: dict = {}

        async def routed_pipeline(
            job_id: str, resume_token: str = "", resume_provider: str = ""
        ) -> dict:
            captured["resume_token"] = resume_token
            captured["resume_provider"] = resume_provider
            raise TranscriptJobPending(provider="supadata", resume_token="job-x")

        monkeypatch.setattr(transcription_module, "run_pipeline", routed_pipeline)
        stub = _StubTask(retries=1)
        with pytest.raises(_RetryRaised):
            download_and_transcribe.run.__func__(stub, "job-1", "job-x", "supadata")

        assert captured == {"resume_token": "job-x", "resume_provider": "supadata"}

    def test_job_retry_countdown_caps_at_max(self) -> None:
        assert _job_retry_countdown(0) == 2
        assert _job_retry_countdown(4) == 32
        assert _job_retry_countdown(10) == 60


class _RetryRaised(Exception):
    """Sentinel raised by the fake Celery task ``retry``."""


class _StubTask:
    """Minimal stand-in for the bound Celery task ``self``."""

    def __init__(self, retries: int = 0) -> None:
        self.request = SimpleNamespace(retries=retries)
        self.retry_call: tuple | None = None

    def retry(self, exc=None, max_retries=None, countdown=None, args=None) -> None:
        self.retry_call = (max_retries, countdown, args)
        raise _RetryRaised()


class _FakeClient:
    """Minimal fake BackendClient that records its calls."""

    def __init__(self, status: str = "pending") -> None:
        self.status = status
        self.calls: list[str] = []
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


def _cloud_result(title: str = "Me at the zoo") -> VidWordsResult:
    """Build a fake cloud-provider result."""
    return VidWordsResult(
        title=title,
        author="jawed",
        duration_seconds=19,
        is_generated=False,
        transcript=TranscriptData(
            language="en",
            text="All right, so here we are.",
            words=[TranscriptWordData(word="All", start_time=0.0, end_time=0.5)],
        ),
    )


def _ytdlp_result() -> VideoTranscriptResult:
    """Build a fake yt-dlp strategy result."""
    return VideoTranscriptResult(
        title="Fallback Video",
        author="",
        duration_seconds=240,
        is_generated=False,
        transcript=_transcript(),
    )


async def _perform_mock(job, resume_token="", resume_provider=""):
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
        vidwords_api_key="",
    )


def _full_settings(**overrides) -> Settings:
    """Build real worker settings with every cloud provider configured."""
    values = {
        "jumpto_live_external_calls": True,
        "jumpto_transcript_mode": "real",
        "transcriptfetch_api_key": "tf-key",
        "transcriptfetch_lang": "en",
        "transcriptfetch_mode": "auto",
        "supadata_api_key": "sp-key",
        "supadata_lang": "en",
        "supadata_mode": "auto",
        "vidwords_api_key": "vw-key",
        "vidwords_lang": "en",
        "vidwords_api_url": "https://vidwords.com",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)
