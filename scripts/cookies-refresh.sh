#!/usr/bin/env bash
#
# JumpTo host-side cookie refresh (runs on the VPS via cron, every 1 min).
#
# Reads the worker's refresh-requested marker. When set, exports the live
# session cookies from the logged-in Chromium container over its DevTools
# loopback endpoint (running INSIDE the container), replaces /etc/jumpto/
# cookies.txt, and clears the marker.
#
# Install (once):
#   1. Build the extended image:  docker build -f scripts/chromium-cdp-Dockerfile -t chromium-cdp scripts/
#      (re-apply to the chromium container, keeping it logged in)
#   2. Cron (every 1 min):
#      * * * * * /opt/jumpto-worker/scripts/cookies-refresh.sh >> /var/log/jumpto_cookie_refresh.log 2>&1
#
# Usage: cookies-refresh.sh [--force]

set -u

MARKER="${JUMPTO_REFRESH_MARKER:-/etc/jumpto/state/refresh-requested}"
# Canonical host-side cookie dir shared with the worker (COOKIE_DIR).
COOKIE_DIR="${COOKIE_DIR:-/etc/jumpto}"
# Container-side view of the same dir (chromium mounts it at /host-cookies).
FRESH="/host-cookies/fresh-cookies.txt"
COOKIES="${COOKIE_FILE:-$COOKIE_DIR/cookies.txt}"
CHROMIUM_CONTAINER="${CHROMIUM_CONTAINER:-chromium}"
CDP_SCRIPT="/opt/jumpto/refresh_cookies.py"

NTFY_URL="${NTFY_URL:-https://ntfy.sh}"
NTFY_TOPIC="${NTFY_TOPIC:-jumpto-cookie-alerts}"

notify() {
  local title="$1"
  local msg="$2"
  local tag="${3:-warning}"
  curl -s -L -o /dev/null \
    -H "Title: $title" \
    -H "Priority: high" \
    -H "Tags: $tag" \
    -d "$msg" \
    "$NTFY_URL/$NTFY_TOPIC"
}

# --- Marker gate ------------------------------------------------------------
if [ "${1:-}" != "--force" ] && [ ! -f "$MARKER" ]; then
  exit 0
fi

echo "$(date '+%F %T') cookie refresh requested; exporting from Chromium ..."

# --- Export from the chromium container -------------------------------------
# --user root: the export writes the fresh cookiejar, and the container's
# default user (abc) cannot write into the mounted cookie dir.
# The exporter writes the CONTAINER-side view (/host-cookies/...); the
# freshness check below reads the same file via the HOST-side view.
if ! docker exec --user root "$CHROMIUM_CONTAINER" python3 "$CDP_SCRIPT" --out "$FRESH"; then
  notify "JumpTo: Chromium cookie export failed" \
    "The chromium container could not export cookies. Is it running and logged in?" warning
  exit 1
fi

if [ ! -s "$COOKIE_DIR/fresh-cookies.txt" ]; then
  notify "JumpTo: Chromium cookie export produced no cookies" \
    "The chromium container returned an empty export; check the container." exclamation
  exit 1
fi

# --- Atomic replace + marker cleanup ----------------------------------------
mv "$COOKIE_DIR/fresh-cookies.txt" "$COOKIES"
rm -f "$MARKER"

echo "$(date '+%F %T') refreshed $COOKIES and cleared marker"