#!/usr/bin/env python3
"""Export YouTube/Google cookies from a running Chromium via the DevTools API.

Runs INSIDE the chromium container, where the loopback DevTools endpoint
(http://127.0.0.1:9222) is reachable. Connects over WebSocket, asks for every
cookie in the browser, keeps only Google/YouTube domains, and writes them in
the Netscape format yt-dlp expects.

Exit codes: 0 exported OK, 1 no DevTools endpoint, 2 not logged in / no auth
cookies, 3 CDP call failed.
"""

import json
import sys
import urllib.request
from pathlib import Path

WS_TIMEOUT_SECONDS = 15
HTTP_TIMEOUT_SECONDS = 5

RELEVANT_DOMAINS = ("youtube.com", "google.com", "ytimg.com", "googlevideo.com")
AUTH_COOKIES = ("SID", "PSID", "__Secure-1PSID", "__Secure-3PSID")

CDP_BASE_URL = "http://127.0.0.1:9222"

EXIT_OK = 0
EXIT_NO_DEVTOOLS = 1
EXIT_NOT_LOGGED_IN = 2
EXIT_CDP_FAILED = 3


class CdpError(RuntimeError):
    """A DevTools protocol call returned an error."""


def _get_json(endpoint: str) -> dict:
    """Fetch and parse a JSON payload from a DevTools HTTP endpoint."""
    with urllib.request.urlopen(endpoint, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _list_targets() -> list[dict]:
    """Return the page/browser targets exposed by Chromium."""
    return _get_json(f"{CDP_BASE_URL}/json/list")


def _pick_page_target(targets: list[dict]) -> dict | None:
    """Pick the most useful page target (a YouTube tab when one exists)."""
    pages = [t for t in targets if t.get("type") == "page"]
    if not pages:
        return None
    youtube = next(
        (t for t in pages if "youtube.com" in t.get("url", "")),
        None,
    )
    return youtube or pages[0]


def _cdp_call(ws_url: str, method: str, params: dict | None = None) -> dict:
    """Send one CDP call over WebSocket and return its result."""
    import websocket  # Installed in the chromium-cdp image

    # suppress_origin=True: without it websocket-client 1.8 auto-sends a
    # browser-style Origin header, which Chromium rejects unless the DevTools
    # server was started with --remote-allow-origins.
    connection = websocket.create_connection(
        ws_url, timeout=WS_TIMEOUT_SECONDS, suppress_origin=True
    )
    try:
        connection.send(json.dumps({"id": 1, "method": method, "params": params or {}}))
        while True:
            message = json.loads(connection.recv())
            message_id = message.get("id")
            if message_id != 1:
                continue
            if "error" in message:
                raise CdpError(message["error"].get("message", method))
            return message["result"]
    finally:
        connection.close()


def _fetch_all_cookies() -> list[dict]:
    """Return every browser cookie by trying page then browser targets."""
    targets = _list_targets()
    page = _pick_page_target(targets)
    if page is not None:
        try:
            _cdp_call(page["webSocketDebuggerUrl"], "Network.enable")
            result = _cdp_call(page["webSocketDebuggerUrl"], "Network.getAllCookies")
            return result.get("cookies", [])
        except Exception as exc:  # noqa: BLE001 - fall back to the browser target
            if isinstance(exc, CdpError):
                raise
    version = _get_json(f"{CDP_BASE_URL}/json/version")
    result = _cdp_call(version["webSocketDebuggerUrl"], "Browser.getCookies")
    return result.get("cookies", [])


def to_netscape_rows(cookies: list[dict]) -> list[dict]:
    """Convert CDP cookies to filtered rows ready for Netscape formatting."""
    rows = []
    for cookie in cookies:
        domain = cookie.get("domain", "")
        if not any(domain.endswith(d) for d in RELEVANT_DOMAINS):
            continue
        rows.append(
            {
                "domain": domain,
                "host_only": "TRUE" if not domain.startswith(".") else "FALSE",
                "path": cookie.get("path") or "/",
                "secure": "TRUE" if cookie.get("secure") else "FALSE",
                "expires": int(cookie.get("expires") or 0),
                "name": cookie.get("name", ""),
                "value": cookie.get("value", ""),
            }
        )
    return rows


def format_netscape(rows: list[dict]) -> str:
    """Render rows as a Netscape cookies.txt body (without the header)."""
    lines = [
        f"{row['domain']}\t{row['host_only']}\t{row['path']}\t"
        f"{row['secure']}\t{row['expires']}\t{row['name']}\t{row['value']}"
        for row in rows
    ]
    return "\n".join(lines)


def write_cookies(cookies: list[dict], output: Path) -> int:
    """Filter, validate, and write the cookie export; return an exit code."""
    rows = to_netscape_rows(cookies)
    names = [row["name"] for row in rows]
    if not rows:
        print("No YouTube/Google cookies found - is Chromium logged in?", file=sys.stderr)
        return EXIT_NOT_LOGGED_IN
    if not any(name in AUTH_COOKIES for name in names):
        print(
            "No YouTube auth cookie (SID/PSID) found; Chromium may not be signed in.",
            file=sys.stderr,
        )
        return EXIT_NOT_LOGGED_IN
    header = "# Netscape HTTP Cookie File\n# Exported from Chromium by jumpto cookie refresh\n"
    output.write_text(header + format_netscape(rows) + "\n", encoding="utf-8")
    print(f"Exported {len(rows)} cookies -> {output}", file=sys.stderr)
    return EXIT_OK


def main() -> int:
    """Run the export: fetch cookies and write them to the output file."""
    output = Path("/etc/jumpto/fresh-cookies.txt")
    try:
        cookies = _fetch_all_cookies()
    except CdpError as exc:
        print(f"CDP call failed: {exc}", file=sys.stderr)
        return EXIT_CDP_FAILED
    except Exception as exc:  # noqa: BLE001 - surface any DevTools failure cleanly
        print(f"No DevTools endpoint reachable: {exc}", file=sys.stderr)
        return EXIT_NO_DEVTOOLS
    return write_cookies(cookies, output)


if __name__ == "__main__":
    sys.exit(main())
