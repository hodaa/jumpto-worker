"""Video URL parsing helpers shared across the worker.

The YouTube id parser lives here, free of any provider import, so both the
source adapters and the transcript cache can use it without cycling through
the providers package.
"""

from __future__ import annotations

import re

# YouTube video id extraction from common URL shapes (watch, youtu.be, shorts,
# embed, live). The id is 11 chars of [A-Za-z0-9_-], accepted at 6+ here to
# keep the historical short ids working.
_VIDEO_ID_PATTERNS = (
    re.compile(r"[?&]v=([A-Za-z0-9_-]{6,})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{6,})"),
    re.compile(r"/(?:shorts|embed|live)/([A-Za-z0-9_-]{6,})"),
)


def extract_youtube_video_id(youtube_url: str) -> str:
    """Best-effort YouTube video id from a URL, or '' when it cannot be parsed."""
    if not youtube_url:
        return ""
    for pattern in _VIDEO_ID_PATTERNS:
        match = pattern.search(youtube_url)
        if match:
            return match.group(1)
    return ""
