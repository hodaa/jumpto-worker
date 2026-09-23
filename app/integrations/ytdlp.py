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
# the mounted source (e.g. /etc/jumpto/cookies.txt) is read-only. Copies are
# cached per source-file revision so a job's repeated yt-dlp runs (metadata,
# captions, audio) reuse one sanitized copy instead of re-copying it each time.
_temp_cookie_copies: list[str] = []

# Source-revision -> sanitized copy path. The key is the source path plus its
# stat, so a host cookie-refresh cron (which atomically swaps the file) changes
# the key and forces a fresh copy.
_cookie_copy_cache: dict[tuple[str, int, int], str] = {}


def _cleanup_temp_cookie_copies() -> None:
    for path in _temp_cookie_copies:
        with contextlib.suppress(OSError):
            Path(path).unlink(missing_ok=True)


atexit.register(_cleanup_temp_cookie_copies)


def _sanitize_cookie_lines(source: str, dest: str) -> None:
    """Copy ``source`` to ``dest``, normalizing rows yt-dlp's cookiejar rejects.

    Python's ``http.cookiejar`` is strict about Netscape format: the
    includeSubdomains flag (2nd field) must exactly match whether the domain
    starts with a dot (TRUE for ``.youtube.com``, FALSE for plain
    ``accounts.google.com``), and expiry -1 (session cookies) must be 0. Old
    cookie exporters get both directions wrong, and yt-dlp then aborts the
    whole download with "assert domain_specified == initial_dot". Sanitizing
    here makes the worker resilient to any source file, stale or freshly
    exported.
    """
    with Path(source).open(encoding="utf-8") as src, Path(dest).open("w", encoding="utf-8") as out:
        for line in src:
            content = line.rstrip("\n")
            if (not content.strip()) or (
                content.startswith("#") and not content.startswith("#HttpOnly_")
            ):
                out.write(line)
                continue
            fields = content.split("\t")
            if len(fields) >= 7:
                domain = fields[0]
                if domain.startswith("#HttpOnly_"):
                    domain = domain[len("#HttpOnly_") :]
                fields[1] = "TRUE" if domain.startswith(".") else "FALSE"
                try:
                    expires = int(fields[4])
                except ValueError:
                    expires = 0
                fields[4] = str(max(expires, 0))
                out.write("\t".join(fields) + "\n")
            else:
                out.write(line)


def _cookie_copy_key(cookie_file: str) -> tuple[str, int, int] | None:
    """Return the (path, mtime_ns, size) identity of a cookiefile, or None.

    None means the source file is unreadable, so the caller must fall back to
    a fresh, uncached copy. The key doubles as the cache invalidation signal:
    the host cron atomically swaps the cookie file, changing its mtime/size.
    """
    try:
        stat = Path(cookie_file).stat()
    except OSError:
        return None
    return (cookie_file, stat.st_mtime_ns, stat.st_size)


def _evict_cookie_copy(cookie_file: str) -> None:
    """Drop every cached copy that belongs to ``cookie_file``.

    Unlinks the cached files and removes them from the atexit registry so an
    invalidated entry does not linger for the process lifetime. Eviction
    happens because the source file changed (host cron swap → new identity)
    or a cached file went missing under the same identity.
    """
    stale = [key for key in _cookie_copy_cache if key[0] == cookie_file]
    for key in stale:
        cached = _cookie_copy_cache.pop(key, "")
        if not cached:
            continue
        with contextlib.suppress(OSError):
            Path(cached).unlink(missing_ok=True)
        with contextlib.suppress(ValueError):
            _temp_cookie_copies.remove(cached)


def _make_cookie_copy(cookie_file: str, cache_key: tuple[str, int, int] | None) -> str:
    """Sanitize a writable copy and register it for the process lifetime."""
    fd, tmp = tempfile.mkstemp(prefix="jumpto-cookies-", suffix=".txt")
    try:
        _sanitize_cookie_lines(cookie_file, tmp)
        if cache_key is not None:
            _cookie_copy_cache[cache_key] = tmp
        _temp_cookie_copies.append(tmp)
        return tmp
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise
    finally:
        os.close(fd)


def _writable_cookie_copy(cookie_file: str) -> str:
    """Return a writable, sanitized copy of ``cookie_file`` for one yt-dlp run.

    The copy is cached against the source file's identity, so a job's up-to-3
    yt-dlp invocations on the same cookies share one sanitized copy instead of
    sanitizing the whole file each time. A fresh copy is made whenever the
    source changes (host cron swap) — the stale cached copy is evicted then.
    """
    key = _cookie_copy_key(cookie_file)
    cached = _cookie_copy_cache.get(key, "")
    if cached and Path(cached).exists():
        return cached
    _evict_cookie_copy(cookie_file)
    return _make_cookie_copy(cookie_file, key)


def release_temp_cookie(options: dict) -> None:
    """Delete and forget the temp cookie copy wired into ``options``.

    Call after the yt-dlp run that built ``options`` finishes (success or
    failure). The cached copy for the current source revision is kept — later
    runs in the same job (or other jobs, until the host swaps cookies) reuse
    it — while any out-of-date copy the run switched away from is removed.
    Keeps the ``_temp_cookie_copies`` registry in sync so ``atexit`` still
    cleans up copies leaked by crashes or call sites that forget to release.
    No-op when no cookie was wired or the file is already gone.
    """
    cookie = options.get("cookiefile")
    if not cookie:
        return
    if cookie in _cookie_copy_cache.values():
        return
    with contextlib.suppress(OSError):
        Path(cookie).unlink(missing_ok=True)
    with contextlib.suppress(ValueError):
        _temp_cookie_copies.remove(cookie)


def build_ydlp_options(*, settings=None, **overrides: object) -> dict:
    """
    Build a base yt-dlp options dict wired with the shared cookie file.

    ``settings`` is an optional injected instance; when ``None`` the process-wide
    singleton is read. Accepting an explicit instance lets leaf callers that
    already hold a reference pass it through instead of re-reading the global,
    which makes the concrete call-site testable without monkeypatching.

    The cookie file is copied to a writable temp file first, so yt-dlp can
    read and refresh it even when the mounted source is read-only.

    Any ``overrides`` are merged on top of the base options.
    """
    if settings is None:
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
