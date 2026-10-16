"""Unit tests for the Vercel Blob upload integration."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.integrations import blob as blob_module
from app.integrations.blob import upload_mp4

_TOKEN_ENV = "BLOB_READ_WRITE_TOKEN"


@pytest.fixture(autouse=True)
def _scrub_written_token():
    """The bridge writes os.environ directly; never leak it across tests."""
    yield
    blob_module.os.environ.pop(_TOKEN_ENV, None)


def _install_fake_put(monkeypatch) -> list[dict]:
    """Capture put_blob arguments and return a canned blob result."""
    calls: list[dict] = []

    def fake_put(pathname, body, *, access, **kwargs):
        calls.append({"pathname": pathname, "content": body.read(), "access": access, **kwargs})
        return SimpleNamespace(url="https://blob.example/downloads/video.mp4")

    monkeypatch.setattr(blob_module.blob, "put_blob", fake_put)
    return calls


def _settings_with_token(token: str) -> Settings:
    """Build isolated settings carrying a blob token (or not)."""
    return Settings(_env_file=None, blob_read_write_token=token)


class TestUploadMp4:
    """upload_mp4 streams the artifact and returns its public URL."""

    def test_uploads_with_streamed_body_and_public_access(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv(_TOKEN_ENV, raising=False)
        monkeypatch.setattr(blob_module, "get_settings", lambda: _settings_with_token(""))
        calls = _install_fake_put(monkeypatch)
        path = tmp_path / "video.mp4"
        path.write_bytes(b"video-bytes")

        url = upload_mp4(path)

        assert url == "https://blob.example/downloads/video.mp4"
        assert len(calls) == 1
        call = calls[0]
        assert call["pathname"] == "downloads/video.mp4"
        assert call["content"] == b"video-bytes"
        assert call["access"] == "public"
        assert call["content_length"] == len(b"video-bytes")
        assert call["content_type"] == "video/mp4"
        assert call["add_random_suffix"] is True

    def test_mirrors_dotenv_token_into_the_environment(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv(_TOKEN_ENV, raising=False)
        monkeypatch.setattr(blob_module, "get_settings", lambda: _settings_with_token("tok-1"))
        _install_fake_put(monkeypatch)

        upload_mp4(_write_video(tmp_path))

        assert blob_module.os.environ[_TOKEN_ENV] == "tok-1"

    def test_existing_environment_token_is_not_overwritten(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv(_TOKEN_ENV, "tok-env")
        monkeypatch.setattr(blob_module, "get_settings", lambda: _settings_with_token("tok-2"))
        _install_fake_put(monkeypatch)

        upload_mp4(_write_video(tmp_path))

        assert blob_module.os.environ[_TOKEN_ENV] == "tok-env"


def _write_video(tmp_path: Path) -> Path:
    """Write a minimal fake MP4 and return its path."""
    path = tmp_path / "video.mp4"
    path.write_bytes(b"video-bytes")
    return path
