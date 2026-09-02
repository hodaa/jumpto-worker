"""Shared yt-dlp option building so every provider uses the same auth/cookies."""

from app.core.config import get_settings


def build_ydlp_options(**overrides: object) -> dict:
    """
    Build a base yt-dlp options dict wired with the shared cookie file.

    Any ``overrides`` are merged on top of the base options.
    """
    options: dict = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    cookie_file = get_settings().resolved_ytdlp_cookie_file
    if cookie_file:
        options["cookiefile"] = cookie_file
    options.update(overrides)
    return options
