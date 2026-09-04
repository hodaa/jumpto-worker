"""Shared yt-dlp option building so every provider uses the same auth/cookies."""

import atexit
import contextlib
import os
import shutil
import tempfile
from pathlib import Path

from app.core.config import get_settings

# Temp copies of the cookie file created so yt-dlp can refresh them even when
# the mounted source (e.g. /etc/jumpto/cookies.txt) is read-only.
_temp_cookie_copies: list[str] = []


def _cleanup_temp_cookie_copies() -> None:
    for path in _temp_cookie_copies:
        with contextlib.suppress(OSError):
            Path(path).unlink(missing_ok=True)


atexit.register(_cleanup_temp_cookie_copies)


def _writable_cookie_copy(cookie_file: str) -> str:
    """Return a writable copy of ``cookie_file`` for use by a single yt-dlp run."""
    fd, tmp = tempfile.mkstemp(prefix="jumpto-cookies-", suffix=".txt")
    try:
        shutil.copyfile(cookie_file, tmp)
        return tmp
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise
    finally:
        os.close(fd)
        _temp_cookie_copies.append(tmp)


def build_ydlp_options(**overrides: object) -> dict:
    """
    Build a base yt-dlp options dict wired with the shared cookie file and
    optional proxy.

    The cookie file is copied to a writable temp file first, so yt-dlp can
    read and refresh it even when the mounted source is read-only.

    Any ``overrides`` are merged on top of the base options.
    """
    settings = get_settings()
    options: dict = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    cookie_file = settings.resolved_ytdlp_cookie_file
    if cookie_file:
        options["cookiefile"] = _writable_cookie_copy(cookie_file)
    if settings.ytdlp_proxy:
        options["proxy"] = settings.ytdlp_proxy
    if settings.ytdlp_bgutil_url:
        options["extractor_args"] = {
            # yt-dlp stores extractor_args in nested {ie: {key: [value]}} form.
            "youtubepot-bgutilhttp": {"base_url": [settings.ytdlp_bgutil_url]},
        }
    # Allow yt-dlp to fetch the EJS challenge-solver scripts (required to pass
    # YouTube's JS challenges; runs them with the bundled Deno runtime).
    options["remote_components"] = ["ejs:github"]
    options.update(overrides)
    return options
