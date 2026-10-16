"""Upload produced video downloads to Vercel Blob storage.

The blob SDK resolves its read-write token from the process environment
(``BLOB_READ_WRITE_TOKEN``). Compose passes the worker's ``.env`` through
``env_file`` so the variable is present in production; for runs where only
pydantic has read the file, the token is mirrored into the environment the
SDK reads, once.
"""

from __future__ import annotations

import os
from pathlib import Path

from vercel.blob import sync as blob

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_TOKEN_ENV = "BLOB_READ_WRITE_TOKEN"
_UPLOAD_PREFIX = "downloads"
_PUBLIC_ACCESS = "public"
_MP4_CONTENT_TYPE = "video/mp4"


def _ensure_token_in_environment() -> None:
    """Mirror a .env-only blob token into the environment the SDK reads."""
    token = get_settings().blob_read_write_token
    if token and not os.environ.get(_TOKEN_ENV):
        os.environ[_TOKEN_ENV] = token


def upload_mp4(path: Path) -> str:
    """Upload an MP4 to blob storage and return its public URL.

    The open file is streamed by the SDK (``content_length`` keeps it off the
    buffering path), so the artifact is never held in memory whole. A random
    suffix keeps two downloads of the same name from overwriting each other's
    blobs.
    """
    _ensure_token_in_environment()
    with path.open("rb") as body:
        result = blob.put_blob(
            f"{_UPLOAD_PREFIX}/{path.name}",
            body,
            access=_PUBLIC_ACCESS,
            content_length=path.stat().st_size,
            content_type=_MP4_CONTENT_TYPE,
            add_random_suffix=True,
        )
    logger.info("Uploaded video download to blob storage", path=str(path))
    return result.url
