#!/usr/bin/env python3
"""Export YouTube/Google cookies from the dedicated Firefox "jumpto" profile.

Runs on the Mac. Uses yt-dlp's own browser-cookie loader (battle-tested Firefox
cookie decryption + macOS Keychain "Firefox Safe Storage" lookup), filters to
Google/YouTube cookies only, and writes a Netscape-format cookies.txt for the
worker's yt-dlp.

Prereqs (once):
  - Firefox installed:  brew install --cask firefox
  - profile seeded:     scripts/jumpto-cookie-auto.sh --setup   (or manually: mkdir
    -p \"~/Library/Application Support/Firefox/Profiles/jumpto\" + profiles.ini)
  - log into youtube.com once in that profile:  firefox -P jumpto
  - venv with yt-dlp:  ~/.jumpto-cookies/.venv/bin/pip install yt-dlp

Usage:
  mac_export_cookies.py [--profile NAME] [--out COOKIES.txt]

Defaults: profile = jumpto   out = ~/.jumpto-cookies/cookies.txt
Exit codes: 0 exported OK, 1 no profile db, 2 not logged in / no auth cookies,
            3 other error.
"""

import argparse
import sys
from pathlib import Path

from yt_dlp.cookies import extract_cookies_from_browser

RELEVANT_DOMAINS = ("youtube.com", "google.com", "ytimg.com", "googlevideo.com")
AUTH_COOKIES = ("SID", "PSID", "__Secure-1PSID", "__Secure-3PSID")

FIREFOX_PROFILES = Path.home() / "Library/Application Support/Firefox/Profiles"


def find_profile_dir(pattern: str) -> Path:
    """Resolve a glob-ish profile name to the matching profile DIRECTORY."""
    if FIREFOX_PROFILES.exists():
        matches = sorted(FIREFOX_PROFILES.glob(f"*{pattern}*"))
        if matches:
            return matches[-1]
    return FIREFOX_PROFILES / pattern


def netscape_rows(jar) -> list:
    rows = []
    for c in jar:
        domain = c.domain
        if not any(domain.endswith(d) for d in RELEVANT_DOMAINS):
            continue
        host_only = "TRUE" if not domain.startswith(".") else "FALSE"
        secure = "TRUE" if c.secure else "FALSE"
        expires = int(c.expires) if c.expires and c.expires > 0 else 0
        rows.append(f"{domain}\t{host_only}\t{c.path or '/'}\t{secure}\t{expires}\t{c.name}\t{c.value}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="jumpto")
    ap.add_argument("--out", default=str(Path.home() / ".jumpto-cookies/cookies.txt"))
    args = ap.parse_args()

    profile_dir = find_profile_dir(args.profile)
    if not (profile_dir / "cookies.sqlite").exists():
        print(
            f"No cookies.sqlite for profile '{args.profile}' ({profile_dir}).",
            file=sys.stderr,
        )
        print("Open the profile once and log into youtube.com, e.g.:", file=sys.stderr)
        print("  /Applications/Firefox.app/Contents/MacOS/firefox -P jumpto", file=sys.stderr)
        return 1

    try:
        jar = extract_cookies_from_browser("firefox", profile=str(profile_dir))
    except Exception as exc:  # noqa: BLE001 - yt-dlp raises on missing keychain etc.
        print(f"Failed to load cookies from {profile_dir}: {exc}", file=sys.stderr)
        return 3

    rows = netscape_rows(jar)
    names = [r.split("\t", 6)[5] for r in rows]

    if not rows:
        print("No YouTube/Google cookies found - is the profile logged in?", file=sys.stderr)
        return 2
    if not any(name in AUTH_COOKIES for name in names):
        print(
            f"No YouTube auth cookie exported ({', '.join(AUTH_COOKIES)}). "
            f"Log into youtube.com in Firefox profile '{args.profile}'.",
            file=sys.stderr,
        )
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Netscape HTTP Cookie File\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"Exported {len(rows)} cookies -> {out} (Firefox profile '{args.profile}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
