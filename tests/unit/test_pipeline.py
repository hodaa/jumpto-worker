"""Unit tests for the worker transcription pipeline."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

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
from app.providers import ytdlp as ytdlp_module
from app.providers.registry import resolve_provider as provider_resolve
from app.providers.vidwords import VidWordsPermanentError
from app.services import jobs as jobs_module
from app.services import pipeline as pipeline_module
from app.services.pipeline import (
    perform_transcription,
    run_pipeline,
    try_provider,
    user_safe_message,
)
from app.services.submissions import build_submission
from app.services.webhooks import build_assembly_webhook_url
from app.tasks import transcription as transcription_module
from app.tasks.transcription import _job_retry_countdown, download_and_transcribe


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
        submission = build_submission(
            title=_media().title,
            duration_seconds=_media().duration_seconds,
            transcript=_transcript(),
        )

        assert isinstance(submission, TranscriptSubmission)
        assert submission.title == _media().title
        assert submission.duration_seconds == _media().duration_seconds
        assert submission.transcript_text == "Hello, world!"
        assert submission.words[0].word == "hello"
        assert submission.words[1].word == "world"
        assert submission.words[0].word_index == 0

    def test_build_submission_drops_empty_words(self) -> None:
        transcript = SimpleNamespace(
            language="en",
            text="… —",
            words=[
                SimpleNamespace(word="…", start_time=0.0, end_time=1.0),
                SimpleNamespace(word="hello", start_time=1.0, end_time=2.0),
                SimpleNamespace(word="—", start_time=2.0, end_time=3.0),
            ],
        )

        submission = build_submission(
            title=_media().title,
            duration_seconds=_media().duration_seconds,
            transcript=transcript,
        )

        assert len(submission.words) == 1
        assert submission.words[0].word == "hello"
        assert submission.words[0].word_index == 0


class TestRunPipeline:
    """Tests for the full worker pipeline flow."""

    @pytest.mark.asyncio
    async def test_happy_path_calls_all_steps(self, monkeypatch) -> None:
        client = _FakeClient()
        monkeypatch.setattr(jobs_module, "BackendClient", lambda base, key: client)

        settings = _settings(live_calls=False)
        monkeypatch.setattr(pipeline_module, "get_settings", lambda: settings)
        monkeypatch.setattr(pipeline_module, "perform_transcription", _perform_mock)

        result = await run_pipeline("job-1")

        assert result["status"] == "completed"
        assert client.calls == ["get_job", "advance", "store", "complete"]
        assert client.failed is False

    @pytest.mark.asyncio
    async def test_transcribes_non_pending_job_without_advancing(self, monkeypatch) -> None:
        client = _FakeClient(status="completed")
        monkeypatch.setattr(jobs_module, "BackendClient", lambda base, key: client)
        settings = _settings(live_calls=False)
        monkeypatch.setattr(pipeline_module, "get_settings", lambda: settings)
        monkeypatch.setattr(pipeline_module, "perform_transcription", _perform_mock)

        result = await run_pipeline("job-1")

        assert result["status"] == "completed"
        assert client.failed is False
        assert client.calls == ["get_job", "store", "complete"]

    @pytest.mark.asyncio
    async def test_failure_marks_job_failed(self, monkeypatch) -> None:
        client = _FakeClient()
        monkeypatch.setattr(jobs_module, "BackendClient", lambda base, key: client)

        settings = _settings(live_calls=False)
        monkeypatch.setattr(pipeline_module, "get_settings", lambda: settings)

        def boom(job, resume_token="", resume_provider=""):
            raise RuntimeError("boom")

        monkeypatch.setattr(pipeline_module, "perform_transcription", boom)

        with pytest.raises(RuntimeError):
            await run_pipeline("job-1")

        assert client.failed is True
        assert client.calls[-1] == "fail"

    @pytest.mark.asyncio
    async def test_no_speech_submitted_as_normal_completion(self, monkeypatch) -> None:
        """An empty (no-speech) transcript goes through the normal submit path,
        so the backend stores a null transcript row instead of a note on a
        special-cased completion."""
        client = _FakeClient()
        monkeypatch.setattr(jobs_module, "BackendClient", lambda base, key: client)
        settings = _settings(live_calls=False)
        monkeypatch.setattr(pipeline_module, "get_settings", lambda: settings)

        async def no_speech(job, resume_token="", resume_provider=""):
            return build_submission(
                title="Silent Video",
                duration_seconds=60,
                transcript=TranscriptData(language="en", text="", words=[]),
            )

        monkeypatch.setattr(pipeline_module, "perform_transcription", no_speech)

        result = await run_pipeline("job-1")

        assert result["status"] == "completed"
        assert client.failed is False
        assert client.calls == ["get_job", "advance", "store", "complete"]
        assert client.last_complete_message == ""

    @pytest.mark.asyncio
    async def test_first_attempt_advances_then_raises_pending(self, monkeypatch) -> None:
        client = _FakeClient()
        monkeypatch.setattr(jobs_module, "BackendClient", lambda base, key: client)

        def pending(job, resume_token="", resume_provider=""):
            raise TranscriptJobPending(provider="supadata", resume_token="job-x")

        monkeypatch.setattr(pipeline_module, "perform_transcription", pending)

        with pytest.raises(TranscriptJobPending):
            await run_pipeline("job-1")

        assert client.calls == ["get_job", "advance"]
        assert client.failed is False

    @pytest.mark.asyncio
    async def test_resume_pending_is_not_advanced_or_failed(self, monkeypatch) -> None:
        client = _FakeClient(status="processing")
        monkeypatch.setattr(jobs_module, "BackendClient", lambda base, key: client)

        def pending(job, resume_token="", resume_provider=""):
            raise TranscriptJobPending(provider="supadata", resume_token="job-x")

        monkeypatch.setattr(pipeline_module, "perform_transcription", pending)

        with pytest.raises(TranscriptJobPending):
            await run_pipeline("job-1", resume_token="job-x", resume_provider="supadata")

        assert client.calls == ["get_job"]
        assert client.failed is False

    def test_user_safe_message_maps_errors(self) -> None:
        assert (
            user_safe_message(RuntimeError("x")) == "Transcription failed. Please try again later."
        )
        assert "timed out" in user_safe_message(TimeoutError())


class TestFetchTranscriptWithRetry:
    """Tests for the local transcript fetch with retry behaviour."""

    @staticmethod
    def _provider(captions_service=None, speech_to_text_provider=None, *, live_calls=True):
        return ytdlp_module.YtDlpTranscriptProvider(
            settings=_settings(live_calls=live_calls),
            captions_service=captions_service,
            speech_to_text_provider=speech_to_text_provider,
        )

    @staticmethod
    def _patch_audio(monkeypatch, tmp_path):
        """Stub the composite's audio download/cleanup with a fake file.

        The composite owns the audio download now, so tests must never reach
        the real yt-dlp downloader.
        """
        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"data")
        monkeypatch.setattr(
            ytdlp_module, "download_audio", lambda url, video_id="", settings=None: str(audio_file)
        )
        monkeypatch.setattr(ytdlp_module, "remove_file", lambda path: None)
        return audio_file

    @pytest.mark.asyncio
    async def test_uses_captions_fast_path_when_live(self) -> None:
        youtube_provider = AsyncMock()
        youtube_provider.fetch.return_value = _transcript()

        provider = self._provider(captions_service=youtube_provider)

        info = {"duration": 240}
        result = await provider._fetch_transcript_with_retry("https://youtu.be/abcde12345", info)

        assert result.text == "Hello, world!"
        youtube_provider.fetch.assert_awaited_once_with("https://youtu.be/abcde12345", info=info)

    @pytest.mark.asyncio
    async def test_captions_fast_path_without_info_fetches_its_own(self) -> None:
        youtube_provider = AsyncMock()
        youtube_provider.fetch.return_value = _transcript()

        provider = self._provider(captions_service=youtube_provider)

        result = await provider._fetch_transcript_with_retry("https://youtu.be/abcde12345")

        assert result.text == "Hello, world!"
        youtube_provider.fetch.assert_awaited_once_with("https://youtu.be/abcde12345", info=None)

    @pytest.mark.asyncio
    async def test_falls_back_to_audio_provider_after_captions_fail(
        self, monkeypatch, tmp_path
    ) -> None:
        def fail(url, info=None):
            raise ExternalServiceError("No captions", service="youtube-captions")

        youtube_provider = AsyncMock()
        youtube_provider.fetch.side_effect = fail

        fallback = AsyncMock()
        fallback.fetch.return_value = _transcript()

        provider = self._provider(
            captions_service=youtube_provider,
            speech_to_text_provider=lambda settings: fallback,
        )
        audio_file = self._patch_audio(monkeypatch, tmp_path)

        result = await provider._fetch_transcript_with_retry("https://youtu.be/abcde12345")

        assert result.text == "Hello, world!"
        fallback.fetch.assert_awaited_once_with(
            "https://youtu.be/abcde12345", audio_path=str(audio_file)
        )

    @pytest.mark.asyncio
    async def test_captionless_video_falls_back_to_audio(self, monkeypatch, tmp_path) -> None:
        """A fully caption-less video yields no captions from the real caption
        provider, then falls back to the audio (Assembly) transcript path."""
        info = {"id": "abcde12345", "automatic_captions": {}, "subtitles": {}}

        fallback = AsyncMock()
        fallback.fetch.return_value = _transcript()

        provider = self._provider(speech_to_text_provider=lambda settings: fallback)
        audio_file = self._patch_audio(monkeypatch, tmp_path)

        result = await provider._fetch_transcript_with_retry("https://youtu.be/abcde12345", info)

        assert result.text == "Hello, world!"
        fallback.fetch.assert_awaited_once()
        fallback.fetch.assert_awaited_with(
            "https://youtu.be/abcde12345", audio_path=str(audio_file)
        )

    @pytest.mark.asyncio
    async def test_pending_audio_job_bubbles_as_strategy_provider(
        self, monkeypatch, tmp_path
    ) -> None:
        """A pending Assembly job from the first attempt must route as yt-dlp
        so the retry resumes the same transcript instead of re-downloading."""

        def no_captions(url, info=None):
            raise ExternalServiceError("No captions", service="youtube-captions")

        captions = AsyncMock()
        captions.fetch.side_effect = no_captions

        audio = AsyncMock()
        audio.fetch.side_effect = TranscriptJobPending(
            message="still processing",
            provider="assemblyai",
            resume_token="asm-1",
            resumable=True,
        )

        provider = self._provider(
            captions_service=captions,
            speech_to_text_provider=lambda settings: audio,
        )
        self._patch_audio(monkeypatch, tmp_path)

        with pytest.raises(TranscriptJobPending) as excinfo:
            await provider._fetch_transcript_with_retry("https://youtu.be/abcde12345")

        assert excinfo.value.provider == "yt-dlp"
        assert excinfo.value.resume_token == "asm-1"
        assert excinfo.value.resumable is True
        assert audio.fetch.await_count == 1

    @pytest.mark.asyncio
    async def test_pending_webhook_flag_survives_strategy_remap(
        self, monkeypatch, tmp_path
    ) -> None:
        """A webhook-armed Assembly pending must reach the task still flagged,
        so it ends cleanly instead of consuming the in-worker retry budget."""

        def no_captions(url, info=None):
            raise ExternalServiceError("No captions", service="youtube-captions")

        captions = AsyncMock()
        captions.fetch.side_effect = no_captions

        audio = AsyncMock()
        audio.fetch.side_effect = TranscriptJobPending(
            message="still processing",
            provider="assemblyai",
            resume_token="asm-1",
            resumable=True,
            webhook=True,
        )

        provider = self._provider(
            captions_service=captions,
            speech_to_text_provider=lambda settings: audio,
        )
        self._patch_audio(monkeypatch, tmp_path)

        with pytest.raises(TranscriptJobPending) as excinfo:
            await provider._fetch_transcript_with_retry(
                "https://youtu.be/abcde12345",
                webhook_url="https://backend.test/api/webhooks/assembly?job_id=job-1",
            )

        assert excinfo.value.provider == "yt-dlp"
        assert excinfo.value.webhook is True
        assert excinfo.value.resume_token == "asm-1"

    @pytest.mark.asyncio
    async def test_webhook_url_forwarded_to_audio_provider_on_submit(
        self, monkeypatch, tmp_path
    ) -> None:
        def fail(url, info=None):
            raise ExternalServiceError("No captions", service="youtube-captions")

        captions = AsyncMock()
        captions.fetch.side_effect = fail

        fallback = AsyncMock()
        fallback.fetch.return_value = _transcript()

        provider = self._provider(
            captions_service=captions,
            speech_to_text_provider=lambda settings: fallback,
        )
        audio_file = self._patch_audio(monkeypatch, tmp_path)

        webhook = "https://backend.test/api/webhooks/assembly?job_id=job-1&provider=yt-dlp"
        result = await provider._fetch_transcript_with_retry(
            "https://youtu.be/abcde12345", webhook_url=webhook
        )

        assert result.text == "Hello, world!"
        fallback.fetch.assert_awaited_once()
        fallback.fetch.assert_awaited_with(
            "https://youtu.be/abcde12345", webhook_url=webhook, audio_path=str(audio_file)
        )

    @pytest.mark.asyncio
    async def test_raises_after_retries_exhausted(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(ytdlp_module, "_RETRY_ATTEMPTS", 2)
        monkeypatch.setattr(ytdlp_module, "_RETRY_DELAY_SECONDS", 0)

        def fail(url, audio_path=""):
            raise ExternalServiceError("boom", service="assemblyai")

        fallback = AsyncMock()
        fallback.fetch.side_effect = fail

        provider = self._provider(
            speech_to_text_provider=lambda settings: fallback,
            live_calls=False,
        )

        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"data")
        removed = []
        monkeypatch.setattr(
            ytdlp_module, "download_audio", lambda url, video_id="", settings=None: str(audio_file)
        )
        monkeypatch.setattr(ytdlp_module, "remove_file", lambda path: removed.append(path))

        with pytest.raises(ExternalServiceError):
            await provider._fetch_transcript_with_retry("https://youtu.be/abcde12345")

        assert fallback.fetch.await_count == 2
        assert fallback.fetch.await_args_list == [
            call("https://youtu.be/abcde12345", audio_path=str(audio_file)),
            call("https://youtu.be/abcde12345", audio_path=str(audio_file)),
        ]
        assert removed == [str(audio_file)]

    @pytest.mark.asyncio
    async def test_reuses_same_audio_file_across_retries(self, monkeypatch, tmp_path) -> None:
        """A speech-to-text failure must re-submit the same file, not re-download."""
        monkeypatch.setattr(ytdlp_module, "_RETRY_ATTEMPTS", 3)
        monkeypatch.setattr(ytdlp_module, "_RETRY_DELAY_SECONDS", 0)

        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"data")
        downloads = []
        removed = []
        monkeypatch.setattr(
            ytdlp_module,
            "download_audio",
            lambda url, video_id="", settings=None: downloads.append(url) or str(audio_file),
        )
        monkeypatch.setattr(ytdlp_module, "remove_file", lambda path: removed.append(path))

        seen: list[str] = []

        async def fetch(url, **kwargs):
            seen.append(kwargs.get("audio_path", ""))
            raise ExternalServiceError("boom", service="deepgram")

        fallback = AsyncMock()
        fallback.fetch.side_effect = fetch
        provider = self._provider(
            speech_to_text_provider=lambda settings: fallback,
            live_calls=False,
        )

        with pytest.raises(ExternalServiceError):
            await provider._fetch_transcript_with_retry("https://youtu.be/abcde12345")

        assert downloads == ["https://youtu.be/abcde12345"]
        assert len(seen) == 3
        assert set(seen) == {str(audio_file)}
        assert removed == [str(audio_file)]

    @pytest.mark.asyncio
    async def test_keyed_success_releases_persisted_audio(self, monkeypatch, tmp_path) -> None:
        """A settled (successful) job releases the persisted per-video audio."""
        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"data")
        removed = []
        monkeypatch.setattr(
            ytdlp_module, "download_audio", lambda url, video_id="", settings=None: str(audio_file)
        )
        monkeypatch.setattr(
            ytdlp_module,
            "remove_audio_cache",
            lambda video_id, settings=None: removed.append(video_id),
        )

        fallback = AsyncMock()
        fallback.fetch.return_value = _transcript()
        captions = AsyncMock()
        captions.fetch.side_effect = ExternalServiceError("No captions", service="youtube-captions")
        provider = self._provider(
            captions_service=captions,
            speech_to_text_provider=lambda settings: fallback,
            live_calls=True,
        )

        result = await provider._fetch_transcript_with_retry(
            "https://youtu.be/abcde12345", video_id="abcde12345"
        )

        assert result.text == "Hello, world!"
        assert removed == ["abcde12345"]

    @pytest.mark.asyncio
    async def test_keyed_final_failure_releases_persisted_audio(
        self, monkeypatch, tmp_path
    ) -> None:
        """Exhausting the retry budget also releases the persisted audio."""
        monkeypatch.setattr(ytdlp_module, "_RETRY_ATTEMPTS", 2)
        monkeypatch.setattr(ytdlp_module, "_RETRY_DELAY_SECONDS", 0)
        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"data")
        removed = []
        monkeypatch.setattr(
            ytdlp_module, "download_audio", lambda url, video_id="", settings=None: str(audio_file)
        )
        monkeypatch.setattr(
            ytdlp_module,
            "remove_audio_cache",
            lambda video_id, settings=None: removed.append(video_id),
        )

        def fail(url, audio_path=""):
            raise ExternalServiceError("boom", service="deepgram")

        fallback = AsyncMock()
        fallback.fetch.side_effect = fail
        captions = AsyncMock()
        captions.fetch.side_effect = ExternalServiceError("No captions", service="youtube-captions")
        provider = self._provider(
            captions_service=captions,
            speech_to_text_provider=lambda settings: fallback,
            live_calls=True,
        )

        with pytest.raises(ExternalServiceError):
            await provider._fetch_transcript_with_retry(
                "https://youtu.be/abcde12345", video_id="abcde12345"
            )

        assert fallback.fetch.await_count == 2
        assert removed == ["abcde12345"]

    @pytest.mark.asyncio
    async def test_resume_skips_audio_download(self, monkeypatch, tmp_path) -> None:
        """A resume polls the pending cloud job; the audio is never downloaded."""
        monkeypatch.setattr(ytdlp_module, "_RETRY_ATTEMPTS", 2)
        monkeypatch.setattr(ytdlp_module, "_RETRY_DELAY_SECONDS", 0)
        downloads = []
        removed = []
        monkeypatch.setattr(
            ytdlp_module,
            "download_audio",
            lambda url, video_id="", settings=None: downloads.append(url)
            or str(tmp_path / "x.webm"),
        )
        monkeypatch.setattr(
            ytdlp_module,
            "remove_audio_cache",
            lambda video_id, settings=None: removed.append(video_id),
        )
        cached_path = str(tmp_path / "cached.webm")
        monkeypatch.setattr(
            ytdlp_module, "audio_cache_path", lambda video_id, settings=None: Path(cached_path)
        )

        audio = AsyncMock()
        audio.fetch.return_value = _transcript()
        provider = self._provider(
            speech_to_text_provider=lambda settings: audio,
            live_calls=True,
        )

        result = await provider._fetch_transcript_with_retry(
            "https://youtu.be/abcde12345", resume_token="asm-1", video_id="abcde12345"
        )

        assert result.text == "Hello, world!"
        assert downloads == []
        audio.fetch.assert_awaited_once_with(
            "https://youtu.be/abcde12345", resume_token="asm-1", audio_path=cached_path
        )
        # Success on resume settles the job and releases the initial submit's file.
        assert removed == ["abcde12345"]


class TestLivePipelineEnabled:
    """Tests for the live-pipeline toggle."""

    def test_disabled_when_live_calls_off(self) -> None:
        assert _live_pipeline_enabled(_settings(live_calls=False)) is False

    def test_enabled_when_live_calls_on(self) -> None:
        assert _live_pipeline_enabled(_settings(live_calls=True)) is True


class TestPerformTranscription:
    """Tests for the single configured provider flow."""

    @staticmethod
    def _ytdlp() -> SimpleNamespace:
        return SimpleNamespace(
            name="yt-dlp", supports_resume=True, fetch=AsyncMock(return_value=_ytdlp_result())
        )

    @pytest.mark.asyncio
    async def test_configured_provider_used_when_it_returns_result(self, monkeypatch) -> None:
        provider = SimpleNamespace(
            name="vidwords", supports_resume=False, fetch=AsyncMock(return_value=_cloud_result())
        )
        monkeypatch.setattr(pipeline_module, "resolve_provider", lambda settings: provider)

        submission = await perform_transcription(_job())

        assert submission.title == "Me at the zoo"
        assert submission.duration_seconds == 19
        assert submission.language == "en"
        assert submission.provider == "vidwords"
        provider.fetch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_provider_miss_raises_external_error(self, monkeypatch) -> None:
        provider = SimpleNamespace(
            name="vidwords", supports_resume=False, fetch=AsyncMock(return_value=None)
        )
        monkeypatch.setattr(pipeline_module, "resolve_provider", lambda settings: provider)

        with pytest.raises(ExternalServiceError):
            await perform_transcription(_job())

    @pytest.mark.asyncio
    async def test_permanent_provider_error_is_not_swallowed(self, monkeypatch) -> None:
        provider = SimpleNamespace(
            name="vidwords",
            supports_resume=False,
            fetch=AsyncMock(
                side_effect=VidWordsPermanentError("invalid credentials", service="vidwords")
            ),
        )
        monkeypatch.setattr(pipeline_module, "resolve_provider", lambda settings: provider)

        with pytest.raises(VidWordsPermanentError):
            await perform_transcription(_job())

    @pytest.mark.asyncio
    async def test_resume_token_routes_to_its_provider(self, monkeypatch) -> None:
        provider = SimpleNamespace(
            name="supadata", supports_resume=True, fetch=AsyncMock(return_value=_cloud_result())
        )
        monkeypatch.setattr(pipeline_module, "resolve_provider", lambda settings: provider)

        submission = await perform_transcription(
            _job(), resume_token="job-x", resume_provider="supadata"
        )

        assert submission.provider == "supadata"
        provider.fetch.assert_awaited_once_with(
            "https://www.youtube.com/watch?v=abcde12345", "abcde12345", resume_token="job-x"
        )

    @pytest.mark.asyncio
    async def test_provider_error_raises_external_error(self, monkeypatch) -> None:
        provider = SimpleNamespace(
            name="vidwords",
            supports_resume=False,
            fetch=AsyncMock(side_effect=ExternalServiceError("oops", service="vidwords")),
        )
        monkeypatch.setattr(pipeline_module, "resolve_provider", lambda settings: provider)

        with pytest.raises(ExternalServiceError):
            await perform_transcription(_job())

    @pytest.mark.asyncio
    async def test_empty_transcript_submitted_normally(self, monkeypatch) -> None:
        """An empty (no-speech) transcript yields a normal empty submission."""
        provider = SimpleNamespace(
            name="yt-dlp",
            supports_resume=True,
            fetch=AsyncMock(return_value=_no_speech_result()),
        )
        monkeypatch.setattr(pipeline_module, "resolve_provider", lambda settings: provider)

        submission = await perform_transcription(_job())

        assert submission.provider == "yt-dlp"
        assert submission.transcript_text == ""
        assert submission.words == []

    @pytest.mark.asyncio
    async def test_default_ytdlp_provider_used(self, monkeypatch) -> None:
        ytdlp = self._ytdlp()
        monkeypatch.setattr(pipeline_module, "resolve_provider", lambda settings: ytdlp)

        submission = await perform_transcription(_job())

        assert submission.provider == "yt-dlp"
        ytdlp.fetch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_resume_token_routed_only_to_its_provider(self, monkeypatch) -> None:
        provider = SimpleNamespace(
            name="vidwords", supports_resume=True, fetch=AsyncMock(return_value=_cloud_result())
        )
        monkeypatch.setattr(pipeline_module, "resolve_provider", lambda settings: provider)

        submission = await perform_transcription(
            _job(), resume_token="job-x", resume_provider="yt-dlp"
        )

        assert submission.provider == "vidwords"
        provider.fetch.assert_awaited_once_with(
            "https://www.youtube.com/watch?v=abcde12345", "abcde12345", resume_token=""
        )

    @pytest.mark.asyncio
    async def test_all_soft_miss_raises_external_error(self, monkeypatch) -> None:
        ytdlp = SimpleNamespace(
            name="yt-dlp", supports_resume=True, fetch=AsyncMock(return_value=None)
        )
        monkeypatch.setattr(pipeline_module, "resolve_provider", lambda settings: ytdlp)

        with pytest.raises(ExternalServiceError):
            await perform_transcription(_job())


class TestResolveProvider:
    """Tests for the single configured provider resolution."""

    def test_ytdlp_alone_when_unset(self) -> None:
        assert provider_resolve(_full_settings()).name == "yt-dlp"

    def test_ytdlp_alone_when_empty(self) -> None:
        assert provider_resolve(_full_settings(video_provider="")).name == "yt-dlp"

    def test_configured_cloud_provider_is_built(self) -> None:
        assert provider_resolve(_full_settings(video_provider="vidwords")).name == "vidwords"

    def test_value_is_case_insensitive(self) -> None:
        assert provider_resolve(_full_settings(video_provider="VidWords")).name == "vidwords"

    def test_explicit_ytdlp(self) -> None:
        assert provider_resolve(_full_settings(video_provider="yt-dlp")).name == "yt-dlp"

    def test_cloud_provider_raises_when_not_live(self) -> None:
        settings = _full_settings(video_provider="vidwords", live_external_calls=False)
        with pytest.raises(ValueError):
            provider_resolve(settings)

    def test_unconfigured_provider_raises(self) -> None:
        settings = _full_settings(video_provider="vidwords", vidwords_api_key="")
        with pytest.raises(ValueError):
            provider_resolve(settings)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            provider_resolve(_full_settings(video_provider="bogus"))


class TestFailureMarking:
    """Tests for failure reporting to the backend."""

    @pytest.mark.asyncio
    async def test_fail_job_is_best_effort_when_marking_fails(self, monkeypatch) -> None:
        class _RaisingClient(_FakeClient):
            async def fail_job(self, job_id: str, error: str) -> None:
                self.calls.append("fail")
                raise RuntimeError("marking failed")

        client = _RaisingClient()
        monkeypatch.setattr(jobs_module, "BackendClient", lambda base, key: client)
        settings = _settings(live_calls=False)
        monkeypatch.setattr(pipeline_module, "get_settings", lambda: settings)

        def boom(job, resume_token="", resume_provider=""):
            raise RuntimeError("transcription boom")

        monkeypatch.setattr(pipeline_module, "perform_transcription", boom)

        with pytest.raises(RuntimeError):
            await run_pipeline("job-1")

        assert client.calls[-1] == "fail"


class TestDownloadAndTranscribe:
    """Tests for the Celery task's retry-based async-job wait."""

    async def _pending_pipeline(
        self,
        job_id: str,
        resume_token: str = "",
        resume_provider: str = "",
        client_factory=None,
    ) -> None:
        raise TranscriptJobPending(provider="supadata", resume_token="job-x")

    async def _non_resumable_pending_pipeline(
        self,
        job_id: str,
        resume_token: str = "",
        resume_provider: str = "",
        client_factory=None,
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
        async def completed(
            job_id: str,
            resume_token: str = "",
            resume_provider: str = "",
            client_factory=None,
        ) -> dict:
            return {"status": "completed"}

        monkeypatch.setattr(transcription_module, "run_pipeline", completed)
        stub = _StubTask(retries=3)

        result = download_and_transcribe.run.__func__(stub, "job-1", "job-x", "supadata")

        assert result == {"status": "completed"}
        assert stub.retry_call is None

    def test_marks_job_failed_once_wait_budget_exhausted(self, monkeypatch) -> None:
        monkeypatch.setattr(transcription_module, "run_pipeline", self._pending_pipeline)

        client = _FakeClient()
        monkeypatch.setattr(jobs_module, "BackendClient", lambda base, key: client)
        settings = _settings(live_calls=False)
        monkeypatch.setattr(jobs_module, "get_settings", lambda: settings)

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
        monkeypatch.setattr(jobs_module, "BackendClient", lambda base, key: client)
        settings = _settings(live_calls=False)
        monkeypatch.setattr(jobs_module, "get_settings", lambda: settings)

        stub = _StubTask(retries=0)
        with pytest.raises(TranscriptJobPending):
            download_and_transcribe.run.__func__(stub, "job-1")

        assert stub.retry_call is None
        assert client.failed is True
        assert client.calls == ["fail"]

    def test_retries_failed_job_routes_resume_token_to_its_provider(self, monkeypatch) -> None:
        captured: dict = {}

        async def routed_pipeline(
            job_id: str,
            resume_token: str = "",
            resume_provider: str = "",
            client_factory=None,
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

    def test_ends_cleanly_when_completion_webhook_armed(self, monkeypatch) -> None:
        async def webhook_pending(
            job_id: str,
            resume_token: str = "",
            resume_provider: str = "",
            client_factory=None,
        ) -> None:
            raise TranscriptJobPending(
                provider="yt-dlp", resume_token="asm-1", resumable=True, webhook=True
            )

        monkeypatch.setattr(transcription_module, "run_pipeline", webhook_pending)

        client = _FakeClient()
        monkeypatch.setattr(jobs_module, "BackendClient", lambda base, key: client)
        settings = _settings(live_calls=False)
        monkeypatch.setattr(jobs_module, "get_settings", lambda: settings)

        stub = _StubTask(retries=0)
        result = download_and_transcribe.run.__func__(stub, "job-1")

        assert result == {"status": "submitted"}
        assert stub.retry_call is None
        assert client.failed is False
        assert client.calls == []


class TestWebhookRouting:
    """Tests for per-job provider webhook callback URLs."""

    @staticmethod
    def _settings(base: str) -> SimpleNamespace:
        return SimpleNamespace(assembly_webhook_base_url=base)

    def test_empty_when_webhooks_not_configured(self) -> None:
        assert build_assembly_webhook_url(_settings(live_calls=True), "job-1", "yt-dlp") == ""

    def test_empty_without_job_id(self) -> None:
        settings = self._settings("https://backend.test/api/webhooks/assembly")
        assert build_assembly_webhook_url(settings, "", "yt-dlp") == ""

    def test_appends_job_context(self) -> None:
        settings = self._settings("https://backend.test/api/webhooks/assembly")
        url = build_assembly_webhook_url(settings, "job-1", "yt-dlp")
        assert url == "https://backend.test/api/webhooks/assembly?job_id=job-1&provider=yt-dlp"

    def test_joins_existing_query_string(self) -> None:
        settings = self._settings("https://backend.test/api/webhooks/assembly?src=asm")
        url = build_assembly_webhook_url(settings, "job-1", "yt-dlp")
        assert url == (
            "https://backend.test/api/webhooks/assembly?src=asm&job_id=job-1&provider=yt-dlp"
        )

    @pytest.mark.asyncio
    async def test_try_provider_forwards_webhook_url_only_on_first_submit(
        self, monkeypatch
    ) -> None:
        provider = SimpleNamespace(
            name="yt-dlp",
            supports_resume=True,
            supports_webhook=True,
            fetch=AsyncMock(return_value=_ytdlp_result()),
        )
        webhook = "https://backend.test/api/webhooks/assembly?job_id=job-1&provider=yt-dlp"

        await try_provider(provider, _job(), webhook_url=webhook)

        provider.fetch.assert_awaited_once_with(
            "https://www.youtube.com/watch?v=abcde12345",
            "abcde12345",
            resume_token="",
            webhook_url=webhook,
        )

        provider.fetch.reset_mock()
        await try_provider(
            provider,
            _job(),
            resume_token="asm-1",
            resume_provider="yt-dlp",
            webhook_url=webhook,
        )

        provider.fetch.assert_awaited_once_with(
            "https://www.youtube.com/watch?v=abcde12345",
            "abcde12345",
            resume_token="asm-1",
        )


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
        self.last_error: str | None = None
        self.last_complete_message = ""

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

    async def complete_job(self, job_id: str, message: str = "") -> None:
        self.calls.append("complete")
        self.last_complete_message = message

    async def fail_job(self, job_id: str, error: str) -> None:
        self.calls.append("fail")
        self.failed = True
        self.last_error = error


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


def _no_speech_result() -> VideoTranscriptResult:
    """Build a provider result whose transcript contains no speech."""
    return VideoTranscriptResult(
        title="Silent Video",
        author="",
        duration_seconds=60,
        is_generated=False,
        transcript=TranscriptData(language="en", text="", words=[]),
    )


async def _perform_mock(job, resume_token="", resume_provider=""):
    """Mock the transcription step to return a submission."""
    return build_submission(
        title=_media().title,
        duration_seconds=_media().duration_seconds,
        transcript=_transcript(),
    )


def _settings(*, live_calls: bool) -> SimpleNamespace:
    """Build a minimal settings object."""
    return SimpleNamespace(
        backend_url="http://backend.test:8000",
        internal_api_key="key",
        job_timeout_seconds=600,
        live_external_calls=live_calls,
        vidwords_api_key="",
    )


def _full_settings(**overrides) -> Settings:
    """Build real worker settings with every cloud provider configured."""
    values = {
        "live_external_calls": True,
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
