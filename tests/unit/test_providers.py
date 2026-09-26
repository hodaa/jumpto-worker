"""Unit tests for external service providers (selection, audio flow, captions)."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from app.core.exceptions import ExternalServiceError, PermanentExternalServiceError
from app.integrations.ytdlp import build_ydlp_options, release_temp_cookie
from app.providers.assembly import (
    _UPLOAD_CHUNK_BYTES,
    AssemblyTranscriptService,
    _log_upload_finished,
    _parse_assembly_transcript,
    _stream_audio_with_progress,
    get_transcript_provider,
)
from app.providers.audio import run_download
from app.providers.deepgram import (
    DeepgramTranscriptService,
    _parse_deepgram_transcript,
    get_deepgram_provider,
)
from app.providers.media import get_media_info
from app.providers.models import TranscriptData, TranscriptJobPending
from app.providers.speech_to_text import get_speech_to_text_provider
from app.providers.transcript import (
    YouTubeCaptionTranscriptService,
    _caption_language,
    _caption_targets,
    _download_caption,
    _parse_vtt,
    _preferred_vtt_file,
)
from app.utils.text import detect_language_from_title


def _settings(*, live: bool, api_key: str) -> SimpleNamespace:
    """Build a minimal settings object for provider selection."""
    return SimpleNamespace(
        live_external_calls=live,
        assembly_api_key=api_key,
        speech_to_text_provider="deepgram",
        deepgram_api_key="",
        deepgram_api_url="https://api.deepgram.com/v1",
    )


class TestMediaInfoProvider:
    """Tests for the media info provider."""

    def test_no_live_calls_raises(self, monkeypatch) -> None:
        settings = SimpleNamespace(live_external_calls=False)
        monkeypatch.setattr("app.providers.media.get_settings", lambda: settings)

        with pytest.raises(ExternalServiceError, match="disabled"):
            get_media_info("abcde12345", "https://youtu.be/abcde12345")

    def test_live_provider_error_is_wrapped(self, monkeypatch) -> None:
        settings = SimpleNamespace(live_external_calls=True)
        monkeypatch.setattr("app.providers.media.get_settings", lambda: settings)

        def boom(url: str):
            raise OSError("network down")

        monkeypatch.setattr("app.providers.media._fetch_from_yt_dlp", boom)

        with pytest.raises(ExternalServiceError):
            get_media_info("abcde12345", "https://youtu.be/abcde12345")


class TestTranscriptProviderSelection:
    """Tests for transcript provider factory selection."""

    def test_returns_none_when_live_not_configured(self, monkeypatch) -> None:
        settings = _settings(live=False, api_key="")
        monkeypatch.setattr("app.providers.assembly.get_settings", lambda: settings)

        provider = get_transcript_provider()

        assert provider is None

    def test_returns_none_when_no_api_key(self, monkeypatch) -> None:
        settings = _settings(live=True, api_key="")
        monkeypatch.setattr("app.providers.assembly.get_settings", lambda: settings)

        provider = get_transcript_provider()

        assert provider is None

    def test_returns_assembly_when_fully_configured(self, monkeypatch) -> None:
        settings = _settings(live=True, api_key="key-123")
        monkeypatch.setattr("app.providers.assembly.get_settings", lambda: settings)

        provider = get_transcript_provider()

        assert isinstance(provider, AssemblyTranscriptService)
        assert provider.api_key == "key-123"


class TestAssemblyParser:
    """Tests for Assembly.ai response parsing."""

    def test_parse_assembly_transcript(self) -> None:
        data = {
            "language_code": "en",
            "text": "hello world",
            "words": [
                {"text": "hello", "start": 100, "end": 250},
                {"text": "world", "start": 250, "end": 500},
            ],
        }

        parsed = _parse_assembly_transcript(data)

        assert parsed.language == "en"
        assert parsed.text == "hello world"
        assert parsed.words[0].start_time == 0.1
        assert parsed.words[1].end_time == 0.5


class TestAssignmentFetcher:
    """Tests for the Assembly.ai HTTP flow."""

    @pytest.mark.asyncio
    async def test_fetch_downloads_uploads_and_polls(self, monkeypatch, tmp_path) -> None:
        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"fake-audio")

        monkeypatch.setattr("app.providers.assembly.download_audio", lambda url: str(audio_file))

        upload_response = Mock(status_code=200)
        upload_response.json.return_value = {"upload_url": "https://cdn.assemblyai.com/fake"}

        submit_response = Mock(status_code=200)
        submit_response.json.return_value = {"id": "transcript-1"}

        completed_body = {
            "status": "completed",
            "language_code": "en",
            "text": "hello world",
            "words": [{"text": "hello", "start": 0, "end": 100}],
        }
        poll_response = Mock(status_code=200)
        poll_response.json.return_value = completed_body

        client = AsyncMock()
        client.post.side_effect = [upload_response, submit_response]
        client.get.return_value = poll_response
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr("app.providers.assembly.httpx.AsyncClient", lambda: client)

        provider = AssemblyTranscriptService("key")
        transcript = await provider.fetch("https://youtu.be/abcde12345")

        assert transcript.text == "hello world"
        assert transcript.words[0].word == "hello"
        assert client.post.call_count == 2
        assert not audio_file.exists()

    @pytest.mark.asyncio
    async def test_fetch_reuses_caller_provided_audio_path(self, monkeypatch, tmp_path) -> None:
        """An injected audio_path must be uploaded in place: no download, no
        removal (the caller owns the file and reuses it across retries)."""
        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"fake-audio")
        downloads = []

        def must_not_download(url):
            downloads.append(url)
            raise AssertionError("must not download when audio_path is injected")

        monkeypatch.setattr("app.providers.assembly.download_audio", must_not_download)

        upload_response = Mock(status_code=200)
        upload_response.json.return_value = {"upload_url": "https://cdn.assemblyai.com/fake"}

        submit_response = Mock(status_code=200)
        submit_response.json.return_value = {"id": "transcript-1"}

        completed_body = {
            "status": "completed",
            "language_code": "en",
            "text": "hello world",
            "words": [{"text": "hello", "start": 0, "end": 100}],
        }
        poll_response = Mock(status_code=200)
        poll_response.json.return_value = completed_body

        client = AsyncMock()
        client.post.side_effect = [upload_response, submit_response]
        client.get.return_value = poll_response
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr("app.providers.assembly.httpx.AsyncClient", lambda: client)

        provider = AssemblyTranscriptService("key")
        transcript = await provider.fetch("https://youtu.be/abcde12345", audio_path=str(audio_file))

        assert transcript.text == "hello world"
        assert not downloads
        assert client.post.call_count == 2
        assert audio_file.exists()

    @pytest.mark.asyncio
    async def test_upload_sends_bytes_not_file_object(self, tmp_path) -> None:
        """Regression: AsyncClient rejects sync file objects as content, so the
        audio body must be read to bytes before upload."""
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["content"] = request.read()
            return httpx.Response(200, json={"upload_url": "https://cdn.assemblyai.com/x"})

        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"fake-audio-bytes")

        provider = AssemblyTranscriptService("key")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            upload_url = await provider._upload(client, {}, str(audio_file))

        assert upload_url == "https://cdn.assemblyai.com/x"
        assert captured["content"] == b"fake-audio-bytes"

    @pytest.mark.asyncio
    async def test_upload_streams_chunks_with_content_length(self, tmp_path) -> None:
        """The upload body must stream from the file with an explicit size,
        not buffer the whole file or fall back to chunked encoding."""
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["content"] = request.read()
            captured["content_length"] = request.headers.get("content-length")
            return httpx.Response(200, json={"upload_url": "https://cdn.assemblyai.com/x"})

        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"A" * (3 * _UPLOAD_CHUNK_BYTES))

        provider = AssemblyTranscriptService("key")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            upload_url = await provider._upload(client, {}, str(audio_file))

        assert upload_url == "https://cdn.assemblyai.com/x"
        assert captured["content"] == b"A" * (3 * _UPLOAD_CHUNK_BYTES)
        assert captured["content_length"] == str(3 * _UPLOAD_CHUNK_BYTES)

    @pytest.mark.asyncio
    async def test_upload_raises_on_non_200(self, tmp_path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"data")

        provider = AssemblyTranscriptService("key")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ExternalServiceError) as excinfo:
                await provider._upload(client, {}, str(audio_file))

        assert "Audio upload failed" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_upload_raises_when_no_upload_url(self, tmp_path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"data")

        provider = AssemblyTranscriptService("key")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ExternalServiceError) as excinfo:
                await provider._upload(client, {}, str(audio_file))

        assert "Audio upload failed" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_stream_progress_logs_per_boundary(self, tmp_path, monkeypatch) -> None:
        """Progress must be logged when each MiB boundary is crossed while
        streaming, so a slow upload is not silent."""
        boundary_bytes = 3
        monkeypatch.setattr("app.providers.assembly._UPLOAD_PROGRESS_LOG_BYTES", boundary_bytes)
        audit: list[str] = []
        monkeypatch.setattr(
            "app.providers.assembly.logger.info",
            lambda event, **kwargs: audit.append(event),
        )

        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"x" * 10)

        chunks = bytearray()
        async for chunk in _stream_audio_with_progress(str(audio_file), 10):
            chunks += chunk

        assert bytes(chunks) == b"x" * 10
        progress_events = [e for e in audit if e == "Assembly upload progress"]
        assert len(progress_events) == 3

    def test_log_upload_finished_does_not_divide_by_zero(self, monkeypatch) -> None:
        """A zero-duration upload must not crash the throughput logging line."""
        audit: list[str] = []
        monkeypatch.setattr(
            "app.providers.assembly.logger.info",
            lambda event, **kwargs: audit.append(event),
        )

        _log_upload_finished(200, 100, 0.0)

        assert audit == ["Assembly upload finished"]

    @pytest.mark.asyncio
    async def test_resume_checks_once_and_raises_pending(self, monkeypatch) -> None:
        poll_response = Mock(status_code=200)
        poll_response.json.return_value = {"status": "processing"}
        client = AsyncMock()
        client.get.return_value = poll_response
        monkeypatch.setattr("app.providers.assembly.httpx.AsyncClient", lambda: client)

        provider = AssemblyTranscriptService("key")
        with pytest.raises(TranscriptJobPending) as excinfo:
            await provider.fetch("https://youtu.be/abcde12345", resume_token="transcript-1")

        assert excinfo.value.resume_token == "transcript-1"
        assert excinfo.value.resumable is True
        client.get.assert_awaited_once()
        client.post.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_submit_attaches_webhook_url_when_given(self) -> None:
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["json"] = json.loads(request.content.decode())
            return httpx.Response(200, json={"id": "transcript-1"})

        provider = AssemblyTranscriptService("key")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            transcript_id = await provider._submit(
                client,
                {},
                "https://cdn.assemblyai.com/fake",
                "https://backend.test/api/webhooks/assembly?job_id=job-1",
            )

        assert transcript_id == "transcript-1"
        assert captured["json"] == {
            "audio_url": "https://cdn.assemblyai.com/fake",
            "webhook_url": "https://backend.test/api/webhooks/assembly?job_id=job-1",
        }

    @pytest.mark.asyncio
    async def test_submit_omits_webhook_url_when_not_given(self) -> None:
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["json"] = json.loads(request.content.decode())
            return httpx.Response(200, json={"id": "transcript-1"})

        provider = AssemblyTranscriptService("key")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await provider._submit(client, {}, "https://cdn.assemblyai.com/fake")

        assert captured["json"] == {"audio_url": "https://cdn.assemblyai.com/fake"}

    @pytest.mark.asyncio
    async def test_fetch_with_webhook_flags_pending(self, monkeypatch, tmp_path) -> None:
        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"fake-audio")
        monkeypatch.setattr("app.providers.assembly.download_audio", lambda url: str(audio_file))

        upload_response = Mock(status_code=200)
        upload_response.json.return_value = {"upload_url": "https://cdn.assemblyai.com/fake"}
        submit_response = Mock(status_code=200)
        submit_response.json.return_value = {"id": "transcript-1"}
        poll_response = Mock(status_code=200)
        poll_response.json.return_value = {"status": "processing"}

        client = AsyncMock()
        client.post.side_effect = [upload_response, submit_response]
        client.get.return_value = poll_response
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr("app.providers.assembly.httpx.AsyncClient", lambda: client)

        provider = AssemblyTranscriptService("key")
        with pytest.raises(TranscriptJobPending) as excinfo:
            await provider.fetch(
                "https://youtu.be/abcde12345",
                webhook_url="https://backend.test/api/webhooks/assembly",
            )

        assert excinfo.value.webhook is True
        assert excinfo.value.resume_token == "transcript-1"
        assert not audio_file.exists()


class TestDeepgramParser:
    """Tests for Deepgram /listen response parsing."""

    def test_parse_deepgram_transcript(self) -> None:
        data = {
            "metadata": {"duration": 2.0},
            "results": {
                "channels": [
                    {
                        "detected_language": "ar",
                        "alternatives": [
                            {
                                "transcript": "مرحبا بالعالم",
                                "words": [
                                    {"word": "مرحبا", "start": 0.0, "end": 0.5},
                                    {"word": "بالعالم", "start": 0.5, "end": 1.0},
                                ],
                            }
                        ],
                    }
                ]
            },
        }

        parsed = _parse_deepgram_transcript(data)

        assert parsed.language == "ar"
        assert parsed.text == "مرحبا بالعالم"
        assert [w.word for w in parsed.words] == ["مرحبا", "بالعالم"]
        assert parsed.words[0].start_time == 0.0
        assert parsed.words[1].end_time == 1.0

    def test_parse_falls_back_to_english_without_detected_language(self) -> None:
        data = {
            "results": {
                "channels": [
                    {
                        "alternatives": [{"transcript": "hi there", "words": []}],
                    }
                ]
            }
        }

        parsed = _parse_deepgram_transcript(data)

        assert parsed.language == "en"
        assert parsed.text == "hi there"

    def test_parse_raises_without_channels(self) -> None:
        with pytest.raises(ExternalServiceError):
            _parse_deepgram_transcript({"results": {}})


class TestDeepgramFetcher:
    """Tests for the Deepgram HTTP flow."""

    @staticmethod
    def _body() -> dict:
        return {
            "results": {
                "channels": [
                    {
                        "detected_language": "en",
                        "alternatives": [
                            {
                                "transcript": "hello world",
                                "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
                            }
                        ],
                    }
                ]
            }
        }

    @pytest.mark.asyncio
    async def test_fetch_posts_audio_and_parses(self, monkeypatch, tmp_path) -> None:
        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"fake-audio")
        monkeypatch.setattr("app.providers.deepgram.download_audio", lambda url: str(audio_file))

        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = str(request.url)
            captured["authorization"] = request.headers.get("authorization")
            captured["content"] = request.read()
            return httpx.Response(200, json=self._body())

        monkeypatch.setattr(
            "app.providers.deepgram.get_shared_http_client",
            lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        monkeypatch.setattr("app.providers.deepgram._TRANSCRIBE_TIMEOUT_SECONDS", 5)

        provider = DeepgramTranscriptService("dg-key", base_url="https://api.deepgram.test/v1")
        transcript = await provider.fetch("https://youtu.be/abcde12345", language="ar")

        assert transcript.text == "hello world"
        assert transcript.words[0].word == "hello"
        assert "language=ar" in captured["path"]
        assert captured["authorization"] == "Token dg-key"
        assert captured["content"] == b"fake-audio"
        assert not audio_file.exists()

    @pytest.mark.asyncio
    async def test_fetch_omits_language_param_when_empty(self, monkeypatch, tmp_path) -> None:
        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"data")
        monkeypatch.setattr("app.providers.deepgram.download_audio", lambda url: str(audio_file))

        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = str(request.url)
            return httpx.Response(200, json=self._body())

        monkeypatch.setattr(
            "app.providers.deepgram.get_shared_http_client",
            lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        provider = DeepgramTranscriptService("dg-key", base_url="https://api.deepgram.test/v1")
        await provider.fetch("https://youtu.be/abcde12345", language="")

        assert "language=" not in captured["path"]

    @pytest.mark.asyncio
    async def test_transcribe_raises_on_non_200(self, tmp_path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"data")

        provider = DeepgramTranscriptService("key", base_url="https://api.deepgram.test/v1")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ExternalServiceError) as excinfo:
                await provider._transcribe(client, str(audio_file), "en")

        assert "rejected" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_transcribe_retries_general_model_on_model_language_mismatch(
        self, tmp_path
    ) -> None:
        """When Deepgram 400s an unsupported model/language pair, the leaf
        honors the API's hint and retries with the ''general'' model on the
        same tier before failing."""
        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"fake-audio")

        captured = {"paths": []}
        mismatch_body = {
            "err_code": "Bad Request",
            "err_msg": (
                "Bad Request: No such model/language/tier combination found "
                'You could try the "general" model (language: ar, Nova-3 tier).'
            ),
        }

        def handler(request: httpx.Request) -> httpx.Response:
            captured["paths"].append(str(request.url))
            if request.url.params.get("model") == "nova-3":
                return httpx.Response(400, json=mismatch_body)
            return httpx.Response(200, json=self._body())

        provider = DeepgramTranscriptService("key", base_url="https://api.deepgram.test/v1")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            transcript = await provider._transcribe(client, str(audio_file), "ar")

        assert transcript.text == "hello world"
        assert len(captured["paths"]) == 2
        assert "model=nova-3" in captured["paths"][0]
        assert "language=ar" in captured["paths"][0]
        assert "model=general" in captured["paths"][1]
        assert "tier=nova-3" in captured["paths"][1]
        assert "language=ar" in captured["paths"][1]

    @pytest.mark.asyncio
    async def test_model_language_mismatch_retry_still_failing_is_permanent(self, tmp_path) -> None:
        """If even the general-model retry is rejected, the leaf fails the job
        loudly instead of guessing further."""
        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"fake-audio")
        mismatch_body = {"err_msg": "Bad Request: No such model/language/tier combination found"}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json=mismatch_body)

        provider = DeepgramTranscriptService("key", base_url="https://api.deepgram.test/v1")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(PermanentExternalServiceError) as excinfo:
                await provider._transcribe(client, str(audio_file), "ar")

        assert excinfo.value.details["service"] == "deepgram"
        assert excinfo.value.details["status_code"] == 400

    @pytest.mark.asyncio
    async def test_non_model_mismatch_4xx_does_not_retry(self, tmp_path) -> None:
        """Ordinary 400s (bad audio, malformed request) never trigger the
        general-model retry."""
        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"fake-audio")
        captured = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["count"] += 1
            return httpx.Response(400, json={"err_msg": "Bad request."})

        provider = DeepgramTranscriptService("key", base_url="https://api.deepgram.test/v1")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(PermanentExternalServiceError):
                await provider._transcribe(client, str(audio_file), "ar")

        assert captured["count"] == 1

    @pytest.mark.asyncio
    async def test_network_error_maps_to_soft_error(self, tmp_path) -> None:
        """A dropped connection mid-upload (httpx.ReadError) must be retried as
        a soft miss, not escape the pipeline unclassified."""
        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"fake-audio")

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadError("connection lost mid-read") from None

        provider = DeepgramTranscriptService("key", base_url="https://api.deepgram.test/v1")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ExternalServiceError) as excinfo:
                await provider._transcribe(client, str(audio_file), "ar")

        assert excinfo.value.details["service"] == "deepgram"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [401, 403])
    async def test_auth_failure_raises_permanent_error(self, tmp_path, status_code: int) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code, json={"err_msg": "Invalid credentials."})

        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"data")

        provider = DeepgramTranscriptService("key", base_url="https://api.deepgram.test/v1")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(PermanentExternalServiceError) as excinfo:
                await provider._transcribe(client, str(audio_file), "en")

        assert excinfo.value.details["service"] == "deepgram"
        assert excinfo.value.details["status_code"] == status_code

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [400, 404, 405, 409, 422])
    async def test_config_4xx_raises_permanent_error(self, tmp_path, status_code: int) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code, json={"err_msg": "Bad request."})

        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"data")

        provider = DeepgramTranscriptService("key", base_url="https://api.deepgram.test/v1")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(PermanentExternalServiceError) as excinfo:
                await provider._transcribe(client, str(audio_file), "en")

        assert excinfo.value.details["service"] == "deepgram"
        assert excinfo.value.details["status_code"] == status_code

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [408, 429, 500, 502, 503, 504])
    async def test_transient_status_codes_remain_soft_errors(
        self, tmp_path, status_code: int
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code, text="try later")

        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"data")

        provider = DeepgramTranscriptService("key", base_url="https://api.deepgram.test/v1")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ExternalServiceError):
                await provider._transcribe(client, str(audio_file), "en")

    @pytest.mark.asyncio
    async def test_fetch_reuses_caller_provided_audio_path(self, monkeypatch, tmp_path) -> None:
        """An injected audio_path must be transcribed in place: no download, no
        removal (the caller owns the file and reuses it across retries)."""
        audio_file = tmp_path / "audio.webm"
        audio_file.write_bytes(b"data")
        downloads = []

        def must_not_download(url):
            downloads.append(url)
            raise AssertionError("must not download when audio_path is injected")

        monkeypatch.setattr("app.providers.deepgram.download_audio", must_not_download)

        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["content"] = request.read()
            return httpx.Response(200, json=self._body())

        monkeypatch.setattr(
            "app.providers.deepgram.get_shared_http_client",
            lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        provider = DeepgramTranscriptService("dg-key", base_url="https://api.deepgram.test/v1")
        transcript = await provider.fetch(
            "https://youtu.be/abcde12345", language="ar", audio_path=str(audio_file)
        )

        assert transcript.text == "hello world"
        assert not downloads
        assert captured["content"] == b"data"
        assert audio_file.exists()


class TestSpeechToTextSelection:
    """Tests for the speech-to-text provider selection."""

    @staticmethod
    def _settings(
        *, live: bool, provider: str, deepgram_key: str = "", assembly_key: str = ""
    ) -> SimpleNamespace:
        return SimpleNamespace(
            live_external_calls=live,
            speech_to_text_provider=provider,
            deepgram_api_key=deepgram_key,
            deepgram_api_url="https://api.deepgram.com/v1",
            deepgram_model="nova-3",
            assembly_api_key=assembly_key,
        )

    def test_returns_none_when_live_not_configured(self, monkeypatch) -> None:
        settings = self._settings(live=False, provider="deepgram", deepgram_key="k")
        monkeypatch.setattr("app.providers.speech_to_text.get_settings", lambda: settings)

        assert get_speech_to_text_provider() is None

    def test_returns_none_when_deepgram_no_key(self, monkeypatch) -> None:
        settings = self._settings(live=True, provider="deepgram")
        monkeypatch.setattr("app.providers.speech_to_text.get_settings", lambda: settings)

        assert get_speech_to_text_provider() is None

    def test_returns_deepgram_when_configured(self, monkeypatch) -> None:
        settings = self._settings(live=True, provider="deepgram", deepgram_key="dg-key")
        monkeypatch.setattr("app.providers.speech_to_text.get_settings", lambda: settings)

        provider = get_speech_to_text_provider()

        assert isinstance(provider, DeepgramTranscriptService)
        assert provider.api_key == "dg-key"

    def test_returns_assembly_when_selected(self, monkeypatch) -> None:
        settings = self._settings(live=True, provider="assembly", assembly_key="asm-key")
        monkeypatch.setattr("app.providers.speech_to_text.get_settings", lambda: settings)

        provider = get_speech_to_text_provider()

        assert isinstance(provider, AssemblyTranscriptService)
        assert provider.api_key == "asm-key"

    def test_returns_none_for_unknown_provider(self, monkeypatch) -> None:
        settings = self._settings(live=True, provider="bogus", deepgram_key="dg-key")
        monkeypatch.setattr("app.providers.speech_to_text.get_settings", lambda: settings)

        assert get_speech_to_text_provider() is None

    def test_get_deepgram_provider_requires_live_and_key(self) -> None:
        assert (
            get_deepgram_provider(self._settings(live=False, provider="deepgram", deepgram_key="k"))
            is None
        )
        assert get_deepgram_provider(self._settings(live=True, provider="deepgram")) is None


class TestLanguageFromTitle:
    """Tests for script-based speech-to-text language detection."""

    def test_arabic_title_is_arabic(self) -> None:
        assert detect_language_from_title("درس البرمجة التجريبي") == "ar"

    def test_english_title_is_english(self) -> None:
        assert detect_language_from_title("Build a Python API from scratch") == "en"

    def test_mixed_title_with_arabic_script_is_arabic(self) -> None:
        assert detect_language_from_title("How to code 2024 | دليل كامل") == "ar"

    def test_empty_title_is_english(self) -> None:
        assert detect_language_from_title("") == "en"

    def test_arabic_supplement_block_is_arabic(self) -> None:
        assert detect_language_from_title("\u0777\u0777") == "ar"


class TestCaptionSelection:
    """Tests for the caption file selection across downloaded tracks."""

    @staticmethod
    def _paths(names: list[str]) -> list:
        from pathlib import Path

        return [Path(name) for name in names]

    def test_prefers_original_audio_track_file(self) -> None:
        files = self._paths(["video.en.vtt", "video.en-orig.vtt", "video.fr.vtt"])

        chosen = _preferred_vtt_file(files)

        assert chosen.name == "video.en-orig.vtt"

    def test_picks_any_language(self) -> None:
        files = self._paths(["video.it.vtt"])

        chosen = _preferred_vtt_file(files)

        assert chosen.name == "video.it.vtt"

    def test_reduces_original_language_files(self) -> None:
        assert _caption_language("video.ar-orig.vtt") == "ar"


class TestCaptionLanguageSelection:
    """Tests for the caption-track selection (original-language restricted)."""

    def test_prefers_manual_subtitles_over_auto(self) -> None:
        info = {
            "automatic_captions": {"en-orig": []},
            "subtitles": {"it": []},
            "original_language": "it",
        }

        assert _caption_targets(info) == ["it"]

    def test_prefers_original_language_and_orig_track(self) -> None:
        info = {
            "automatic_captions": {"en-orig": [], "it-1-orig": [], "fr": []},
            "subtitles": {},
            "original_language": "it",
        }

        targets = _caption_targets(info)

        assert targets[0] == "it-1-orig"

    def test_excludes_translated_tracks_when_original_known(self) -> None:
        info = {
            "automatic_captions": {"en-orig": [], "en": [], "fr-orig": [], "de": []},
            "subtitles": {},
            "original_language": "en",
        }

        targets = _caption_targets(info)

        assert targets == ["en-orig", "en"]

    def test_no_original_language_track_is_a_soft_miss(self) -> None:
        info = {
            "automatic_captions": {"fr-orig": [], "de": []},
            "subtitles": {"es": []},
            "original_language": "en",
        }

        assert _caption_targets(info) == []

    def test_unknown_original_language_uses_any_track(self) -> None:
        info = {"automatic_captions": {"fr": [], "de": []}, "subtitles": {}}

        targets = _caption_targets(info)

        assert targets == ["de"]

    def test_falls_back_to_any_auto_caption(self) -> None:
        info = {"automatic_captions": {"ar": [], "en": []}, "subtitles": {}}

        assert _caption_targets(info) == ["ar"] or _caption_targets(info) == ["en"]

    def test_uses_language_metadata_when_original_language_missing(self) -> None:
        info = {
            "automatic_captions": {"ar": [], "ar-orig": [], "en": [], "en-orig": []},
            "subtitles": {},
            "language": "en-US",
        }

        assert _caption_targets(info) == ["en-orig", "en"]

    def test_language_hint_prefers_manual_track_for_that_language(self) -> None:
        info = {
            "automatic_captions": {"ar-orig": [], "en-orig": []},
            "subtitles": {"en": []},
            "language": "en",
        }

        assert _caption_targets(info) == ["en"]

    def test_language_hint_with_no_matching_track_uses_best_available(self) -> None:
        info = {
            "automatic_captions": {"ar": [], "fr": []},
            "subtitles": {},
            "language": "en-US",
        }

        assert _caption_targets(info) == ["ar"]

    def test_no_captions_returns_empty_targets(self) -> None:
        info = {"automatic_captions": {}, "subtitles": {}}

        assert _caption_targets(info) == []


class TestCaptionParser:
    """Tests for VTT caption parsing into timestamped words."""

    def test_parses_inline_word_timestamps(self) -> None:
        vtt = (
            "WEBVTT\n"
            "Kind: captions\n\n"
            "00:00:01.199 --> 00:00:03.389 align:start\n"
            "hello<00:00:01.480><c> world</c><00:00:02.240><c> from</c>\n"
        )

        transcript = _parse_vtt(vtt, "en")

        assert transcript.language == "en"
        assert [w.word for w in transcript.words] == ["world", "from"]
        assert transcript.words[0].start_time == pytest.approx(1.48)
        assert transcript.text == "world from"

    def test_parses_line_level_cues_without_word_timing(self) -> None:
        vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nplain line only\n"

        transcript = _parse_vtt(vtt, "en")

        assert [w.word for w in transcript.words] == ["plain", "line", "only"]

    def test_strips_musical_note_symbols(self) -> None:
        vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.500\n\u266a HELLO WORLD \u266a\n"

        transcript = _parse_vtt(vtt, "en")

        assert [w.word for w in transcript.words] == ["HELLO", "WORLD"]
        assert transcript.text == "HELLO WORLD"


class TestYouTubeCaptionFetcher:
    """Tests for the YouTube caption transcript provider."""

    @pytest.mark.asyncio
    async def test_fetch_returns_parsed_transcript(self, monkeypatch) -> None:
        provider = YouTubeCaptionTranscriptService()
        monkeypatch.setattr(
            "app.providers.transcript._download_caption",
            lambda url, langs, info=None: (
                "00:00:00.000 --> 00:00:02.000\n<00:00:00.500><c> hi</c>",
                "en",
            ),
        )

        transcript = await provider.fetch("https://www.youtube.com/watch?v=abcde12345")

        assert isinstance(transcript, TranscriptData)
        assert transcript.words[0].word == "hi"
        assert transcript.words[0].start_time == pytest.approx(0.5)


class _ReplayYoutubeDL:
    """Stands in for yt-dlp and records whether it replayed cached info."""

    def __init__(self, options: dict) -> None:
        self.options = options
        self.replayed: tuple[dict, bool] | None = None
        self.downloaded: list[str] | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def process_ie_result(self, info: dict, download: bool = True) -> dict:
        self.replayed = (info, download)
        return info

    def download(self, url_list: list[str]) -> None:
        self.downloaded = url_list


class TestDownloadCaptionMetadataReuse:
    """Tests that pre-fetched yt-dlp metadata is replayed, not re-extracted."""

    def test_reuses_provided_info_without_re_extracting(self, monkeypatch, tmp_path) -> None:
        extract = Mock()
        fake = _ReplayYoutubeDL({})
        monkeypatch.setattr("app.providers.transcript._extract_video_info", extract)
        monkeypatch.setattr("yt_dlp.YoutubeDL", lambda options: fake)
        info = {
            "automatic_captions": {"en-orig": [{"ext": "vtt"}]},
            "subtitles": {},
        }

        result = _download_caption("https://youtu.be/abcde12345", info)

        assert result is None
        extract.assert_not_called()
        assert fake.replayed == (info, True)

    def test_extracts_when_info_not_provided(self, monkeypatch, tmp_path) -> None:
        from unittest.mock import Mock

        extract = Mock(return_value={"automatic_captions": {}, "subtitles": {}})
        monkeypatch.setattr("app.providers.transcript._extract_video_info", extract)

        result = _download_caption("https://youtu.be/abcde12345")

        assert result is None
        extract.assert_called_once_with("https://youtu.be/abcde12345")


class TestRunDownload:
    """run_download always re-extracts and downloads fresh URLs."""

    def test_never_replays_stale_info_dict(self, monkeypatch) -> None:
        """Regression: replaying a pre-extracted info dict reuses YouTube
        video-serving URLs that expire within seconds and 403s when the audio
        fallback runs. Even when stale ``info`` is available it must not be
        replayed."""
        fake = _ReplayYoutubeDL({})
        monkeypatch.setattr("yt_dlp.YoutubeDL", lambda options: fake)

        run_download({}, "https://youtu.be/abc")

        assert fake.replayed is None
        assert fake.downloaded == ["https://youtu.be/abc"]


class TestYdlpOptions:
    """Tests for the shared yt-dlp options builder."""

    def _settings(self, cookie_file):
        return SimpleNamespace(
            resolved_ytdlp_cookie_file=cookie_file,
            ytdlp_bgutil_url="",
            ytdlp_socket_timeout=30,
        )

    def test_sets_writable_cookie_copy_when_configured(self, monkeypatch, tmp_path) -> None:
        source = tmp_path / "cookies.txt"
        source.write_text("# Netscape HTTP Cookie File\n")
        monkeypatch.setattr(
            "app.integrations.ytdlp.get_settings", lambda: self._settings(str(source))
        )

        options = build_ydlp_options()

        cookie_path = options["cookiefile"]
        assert cookie_path != str(source)
        assert Path(cookie_path).read_text() == source.read_text()
        assert os.access(cookie_path, os.W_OK)

    def test_cookie_copy_sanitizes_broken_netscape_rows(self, monkeypatch, tmp_path) -> None:
        source = tmp_path / "cookies.txt"
        source.write_text(
            "# Netscape HTTP Cookie File\n"
            ".youtube.com\tFALSE\t/\tTRUE\t1804257102\t__Secure-YNID\t21.YT=abc\n"
            ".youtube.com\tFALSE\t/\tTRUE\t-1\tYSC\tp6lOdrlT3gs\n"
            "youtube.com\tTRUE\t/\tFALSE\t-1\tCONSENT\tYES\n"
            "accounts.google.com\tTRUE\t/\tTRUE\t1791297119\tOTZ\t8773352\n"
        )
        monkeypatch.setattr(
            "app.integrations.ytdlp.get_settings", lambda: self._settings(str(source))
        )

        options = build_ydlp_options()

        body = Path(options["cookiefile"]).read_text()
        assert ".youtube.com\tTRUE\t/\tTRUE\t1804257102\t__Secure-YNID\t21.YT=abc\n" in body
        assert ".youtube.com\tTRUE\t/\tTRUE\t0\tYSC\tp6lOdrlT3gs\n" in body
        assert "youtube.com\tFALSE\t/\tFALSE\t0\tCONSENT\tYES\n" in body
        assert "accounts.google.com\tFALSE\t/\tTRUE\t1791297119\tOTZ\t8773352\n" in body

    def test_cookie_copy_sanitizes_httponly_dotted_domain(self, monkeypatch, tmp_path) -> None:
        source = tmp_path / "cookies.txt"
        source.write_text(
            "# Netscape HTTP Cookie File\n#HttpOnly_.youtube.com\tFALSE\t/\tTRUE\t0\tSID\tv\n"
        )
        monkeypatch.setattr(
            "app.integrations.ytdlp.get_settings", lambda: self._settings(str(source))
        )

        options = build_ydlp_options()

        body = Path(options["cookiefile"]).read_text()
        assert "#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tv\n" in body

    def test_cookie_copy_is_cached_across_builds(self, monkeypatch, tmp_path) -> None:
        source = tmp_path / "cookies.txt"
        source.write_text("# Netscape HTTP Cookie File\nuser_token=abc\n")
        monkeypatch.setattr(
            "app.integrations.ytdlp.get_settings", lambda: self._settings(str(source))
        )

        first = build_ydlp_options()["cookiefile"]
        second = build_ydlp_options()["cookiefile"]

        assert first == second
        assert Path(first).read_text() == "# Netscape HTTP Cookie File\nuser_token=abc\n"

    def test_cookie_copy_is_refreshed_when_source_changes(self, monkeypatch, tmp_path) -> None:
        source = tmp_path / "cookies.txt"
        source.write_text("# Netscape HTTP Cookie File\nuser_token=abc\n")
        monkeypatch.setattr(
            "app.integrations.ytdlp.get_settings", lambda: self._settings(str(source))
        )

        first = build_ydlp_options()["cookiefile"]
        source.write_text("# Netscape HTTP Cookie File\nuser_token=replacement\n")

        second = build_ydlp_options()["cookiefile"]

        assert first != second
        assert Path(second).read_text() == "# Netscape HTTP Cookie File\nuser_token=replacement\n"
        assert not Path(first).exists()

    def test_cookie_copy_uncached_when_source_unreadable(self, monkeypatch, tmp_path) -> None:
        source = tmp_path / "cookies.txt"
        source.write_text("# Netscape HTTP Cookie File\nuser_token=abc\n")
        monkeypatch.setattr(
            "app.integrations.ytdlp.get_settings", lambda: self._settings(str(source))
        )

        import app.integrations.ytdlp as ytdlp_mod

        monkeypatch.setattr(ytdlp_mod, "_cookie_copy_key", lambda cookie_file: None)

        first = build_ydlp_options()["cookiefile"]
        second = build_ydlp_options()["cookiefile"]

        assert first != second
        assert Path(first).read_text() == "# Netscape HTTP Cookie File\nuser_token=abc\n"
        assert Path(second).read_text() == "# Netscape HTTP Cookie File\nuser_token=abc\n"

    def test_release_temp_cookie_deletes_uncached_copy(self, monkeypatch, tmp_path) -> None:
        source = tmp_path / "cookies.txt"
        source.write_text("# Netscape HTTP Cookie File\nuser_token=abc\n")
        monkeypatch.setattr(
            "app.integrations.ytdlp.get_settings", lambda: self._settings(str(source))
        )
        orphaned = tmp_path / "orphaned.txt"
        orphaned.write_text("# Netscape HTTP Cookie File\n")

        release_temp_cookie({"cookiefile": str(orphaned)})

        assert not orphaned.exists()

    def test_omits_cookiefile_when_unset(self, monkeypatch) -> None:
        settings = SimpleNamespace(
            resolved_ytdlp_cookie_file=None,
            ytdlp_bgutil_url="",
            ytdlp_socket_timeout=30,
        )
        monkeypatch.setattr("app.integrations.ytdlp.get_settings", lambda: settings)

        options = build_ydlp_options()

        assert "cookiefile" not in options
        assert "extractor_args" not in options

    def test_sets_bgutil_extractor_args_when_configured(self, monkeypatch, tmp_path) -> None:
        source = tmp_path / "cookies.txt"
        source.write_text("# Netscape HTTP Cookie File\n")
        settings = SimpleNamespace(
            resolved_ytdlp_cookie_file=str(source),
            ytdlp_bgutil_url="http://bgutil-pot:4416",
            ytdlp_socket_timeout=30,
        )
        monkeypatch.setattr("app.integrations.ytdlp.get_settings", lambda: settings)

        options = build_ydlp_options()

        assert options["extractor_args"] == {
            "youtubepot-bgutilhttp": {"base_url": ["http://bgutil-pot:4416"]},
        }

    def test_overrides_win_over_base_options(self, monkeypatch, tmp_path) -> None:
        source = tmp_path / "cookies.txt"
        source.write_text("# Netscape HTTP Cookie File\n")
        monkeypatch.setattr(
            "app.integrations.ytdlp.get_settings", lambda: self._settings(str(source))
        )

        options = build_ydlp_options(format="best", noplaylist=False)

        assert options["format"] == "best"
        assert options["noplaylist"] is False
        assert options["cookiefile"] != str(source)
