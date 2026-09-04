#!/usr/bin/env bash
#
# JumpTo cookie health prober.
#
# Runs a yt-dlp video-metadata probe INSIDE the worker container and pushes an
# ntfy alert when YouTube rejects the mounted account cookies (expired/rotated)
# or bot-checks the request. Alerts only on STATE CHANGES, so a broken cookie
# file pings once per outage instead of spamming.
#
# Runs on the VPS. Install:
#   1. Copy to ~/jumpto-worker/scripts/cookie_health.sh
#   2. chmod +x scripts/cookie_health.sh
#   3. Cron (every 4h): 0 */4 * * * ~/jumpto-worker/scripts/cookie_health.sh >> /tmp/jumpto_cookie_health.log 2>&1
#
# Optional env: NTFY_TOPIC, NTFY_URL, STATE_FILE, PROBE_URL
#
# Usage: cookie_health.sh [--force]   (--force alerts even if state is unchanged)

set -u

NTFY_URL="${NTFY_URL:-https://ntfy.sh}"
NTFY_TOPIC="${NTFY_TOPIC:-jumpto-cookie-alerts}"
STATE_FILE="${STATE_FILE:-/tmp/jumpto_cookie_health_state}"
PROBE_URL="${PROBE_URL:-https://www.youtube.com/watch?v=jNQXAC9IVRw}"

probe() {
  docker exec -i jumpto-worker python - <<'PY' 2>&1
import yt_dlp
from app.providers.ytdlp import build_ydlp_options

URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
try:
    with yt_dlp.YoutubeDL(build_ydlp_options()) as ydl:
        info = ydl.extract_info(URL, download=False)
    print("HEALTH_OK", str((info or {}).get("title", "")) if isinstance(info, dict) else "")
except Exception as exc:  # noqa: BLE001 - report a clean one-liner
    print("HEALTH_FAIL", str(exc).splitlines()[-1] if str(exc) else repr(exc))
    raise
PY
}

notify() {
  local title="$1"
  local msg="$2"
  curl -s -L -o /dev/null \
    -H "Title: $title" \
    -H "Priority: high" \
    -H "Tags: warning" \
    -d "$msg" \
    "$NTFY_URL/$NTFY_TOPIC"
}

status=""
reason=""

out="$(probe)"

if grep -q "HEALTH_OK" <<<"$out"; then
  status="ok"
elif grep -qi "cookies are no longer valid" <<<"$out"; then
  status="expired"
  reason="YouTube reports the mounted account cookies are no longer valid/rotated."$'\n'"Export fresh cookies from your browser, then run redeploy_cookies.sh on the Mac."
elif grep -qi "Sign in to confirm you're not a bot" <<<"$out"; then
  status="bot"
  reason="YouTube returned a bot-check ('Sign in to confirm you're not a bot'). Cookies or the PO-token provider need attention."
else
  status="fail"
  reason="Metadata probe failed for an unexpected reason."
fi

if [ "$status" = "ok" ]; then
  previous="$(cat "$STATE_FILE" 2>/dev/null || true)"
  if [ -n "$previous" ] && [ "$previous" != "ok" ]; then
    notify "JumpTo: cookies recovered" "The cookie probe is healthy again."
  fi
  echo "ok" > "$STATE_FILE"
  exit 0
fi

previous="$(cat "$STATE_FILE" 2>/dev/null || true)"
echo "$status" > "$STATE_FILE"

if [ "$previous" != "$status" ] || [ "${1:-}" = "--force" ]; then
  notify "JumpTo: cookies need attention" "$reason

Probe URL: $PROBE_URL
Status: $status"
  printf '=== %s ===\n%s\n' "$(date '+%F %T')" "$out" >> /tmp/jumpto_cookie_health.log 2>/dev/null || true
fi

exit 1