"""Shared yt-dlp option building and bot-check handling."""

import atexit
import contextlib
import os
import tempfile
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# YouTube's bot-check message surfaced by yt-dlp when the requester looks
# like an automated client rather than a signed-in browser.
_BOT_CHECK_MARKERS = ("sign in to confirm you're not a bot",)

# Temp copies of the cookie file created so yt-dlp can refresh them even when
# the mounted source (e.g. /etc/jumpto/cookies.txt) is read-only.
_temp_cookie_copies: list[str] = []


def _cleanup_temp_cookie_copies() -> None:
    for path in _temp_cookie_copies:
        with contextlib.suppress(OSError):
            Path(path).unlink(missing_ok=True)


atexit.register(_cleanup_temp_cookie_copies)


def _sanitize_cookie_lines(source: str, dest: str) -> None:
    """Copy ``source`` to ``dest``, normalizing rows yt-dlp's cookiejar rejects.

    Python's ``http.cookiejar`` is strict about Netscape format: a dotted
    domain (e.g. ``.youtube.com``) must have the includeSubdomains flag set to
    TRUE, and expiry -1 (session cookies) must be 0. Files exported by older
    cookie exporters (or the mac_export path) get this wrong, and yt-dlp then
    aborts the whole download with "assert domain_specified == initial_dot".
    Sanitizing here makes the worker resilient to any source file, stale or
    freshly exported.
    """
    with Path(source).open(encoding="utf-8") as src, Path(dest).open("w", encoding="utf-8") as out:
        for line in src:
            content = line.rstrip("\n")
            if (not content.strip()) or (content.startswith("#") and not content.startswith("#HttpOnly_")):
                out.write(line)
                continue
            fields = content.split("\t")
            if len(fields) >= 7:
                domain = fields[0]
                if domain.startswith("#HttpOnly_"):
                    domain = domain[len("#HttpOnly_"):]
                if domain.startswith("."):
                    fields[1] = "TRUE"
                try:
                    expires = int(fields[4])
                except ValueError:
                    expires = 0
                fields[4] = str(max(expires, 0))
                out.write("\t".join(fields) + "\n")
            else:
                out.write(line)


def _writable_cookie_copy(cookie_file: str) -> str:
    """Return a writable, sanitized copy of ``cookie_file`` for one yt-dlp run."""
    fd, tmp = tempfile.mkstemp(prefix="jumpto-cookies-", suffix=".txt")
    try:
        _sanitize_cookie_lines(cookie_file, tmp)
        return tmp
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise
    finally:
        os.close(fd)
        _temp_cookie_copies.append(tmp)


def release_temp_cookie(options: dict) -> None:
    """Delete and forget the temp cookie copy wired into ``options``.

    Call after the yt-dlp run that built ``options`` finishes (success or
    failure) so temp copies don't accumulate for the process lifetime. Keeps
    the ``_temp_cookie_copies`` registry in sync so ``atexit`` still cleans up
    copies leaked by crashes or call sites that forget to release. No-op when
    no cookie was wired or the file is already gone.
    """
    cookie = options.get("cookiefile")
    if not cookie:
        return
    with contextlib.suppress(OSError):
        Path(cookie).unlink(missing_ok=True)
    with contextlib.suppress(ValueError):
        _temp_cookie_copies.remove(cookie)


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
        "socket_timeout": settings.ytdlp_socket_timeout,
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


def is_youtube_bot_check(error: BaseException | str) -> bool:
    """Return True when a yt-dlp error is YouTube's "not a bot" bot-check."""
    text = str(error).lower()
    return any(marker in text for marker in _BOT_CHECK_MARKERS)


def request_cookie_refresh() -> None:
    """Touch the cookie-refresh marker so the host re-exports cookies.

    No-op when ``cookie_refresh_marker_path`` is empty or the marker already
    exists, so a bot-check storm only raises the flag once.
    """
    marker = Path(get_settings().cookie_refresh_marker_path or "").expanduser()
    if not marker.parts or marker.exists():
        return
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch(exist_ok=True)
    except OSError as exc:
        logger.warning("Could not write cookie-refresh marker", path=str(marker), error=str(exc))
