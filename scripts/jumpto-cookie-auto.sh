#!/usr/bin/env bash
#
# JumpTo fully-automatic cookie refresher (runs on the Mac via cron).
#
# Loop:
#   1. Check the VPS cookie-health state -- if healthy, do nothing.
#   2. If blocked, re-EXPORT the live session cookies from the dedicated
#      Firefox "jumpto" profile (mac_export_cookies.py).
#   3. Sanity-check the export is actually logged in.
#   4. Deploy via redeploy_cookies.sh (upload -> restart -> verify -> ntfy).
#   5. If the verify still fails, notify that a manual one-time login is needed.
#
# The only manual step in the whole chain is keeping Firefox "jumpto" logged
# into YouTube (weeks/months between rotations). Everything else is hands-off.
#
# Install (once):  scripts/jumpto-cookie-auto.sh --setup
# Cron (every 15 min):
#   */15 * * * * ~/jumpto-worker/scripts/jumpto-cookie-auto.sh >> /tmp/jumpto_cookie_auto.log 2>&1
#
# Usage: jumpto-cookie-auto.sh [--force]   (--force: run end-to-end regardless)

set -u

SSH_TARGET="jumpto"
STATE_FILE="/tmp/jumpto_cookie_health_state"
COOKIE_EXPORT="$HOME/.jumpto-cookies/cookies.txt"
PYTHON="$HOME/.jumpto-cookies/.venv/bin/python"
EXPORTER="$(cd "$(dirname "$0")" && pwd)/mac_export_cookies.py"
DEPLOYER="$(cd "$(dirname "$0")" && pwd)/redeploy_cookies.sh"

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

setup() {
  local ff="/Applications/Firefox.app/Contents/MacOS/firefox"
  if [ ! -x "$ff" ]; then
    echo "Firefox is not installed. Install it first: brew install --cask firefox" >&2
    exit 1
  fi

  # Seed the dedicated profile directory + profiles.ini (macOS -CreateProfile is unreliable).
  local profiles_dir="$HOME/Library/Application Support/Firefox/Profiles"
  mkdir -p "$profiles_dir/jumpto"
  if [ ! -f "$HOME/Library/Application Support/Firefox/profiles.ini" ]; then
    cat > "$HOME/Library/Application Support/Firefox/profiles.ini" <<INI
[General]
StartWithLastProfile=0
Version=2

[Profile0]
Name=jumpto
IsRelative=1
Path=Profiles/jumpto
Default=1
INI
    echo "Created Firefox 'jumpto' profile (Profiles/jumpto)."
  else
    echo "Firefox profiles.ini already exists; make sure a 'jumpto' profile points at Profiles/jumpto."
  fi

  echo
  echo "NEXT (one-time, manual): open this profile, sign into YouTube, and"
  echo "leave it logged in:"
  echo "    $ff -P jumpto"
  echo "    -> sign in at www.youtube.com"
  echo
  echo "Then test the export with:"
  echo "    $HOME/.jumpto-cookies/.venv/bin/python '$EXPORTER'"
  echo "And add the cron line every 15 minutes:"
  echo "    */15 * * * * '$0' >> /tmp/jumpto_cookie_auto.log 2>&1"
  exit 0
}

if [ "${1:-}" = "--setup" ]; then
  setup
fi

MODE="auto"
[ "${1:-}" = "--force" ] && MODE="force"

# --- 1. Is the VPS cookie state healthy? ------------------------------------
health="$(ssh -o ConnectTimeout=10 "$SSH_TARGET" "cat $STATE_FILE 2>/dev/null || true")"
health="$(printf '%s' "$health" | tr -d '[:space:]')"

if [ "$health" = "ok" ] && [ "$MODE" != "force" ]; then
  echo "$(date '+%F %T') healthy ($health); nothing to do"
  exit 0
fi

echo "$(date '+%F %T') state='${health:-unknown}'; refreshing cookies"

# --- 2. Export the live session cookies from the "jumpto" profile -----------
echo "Exporting cookies from Firefox 'jumpto' profile ..."
if ! "$PYTHON" "$EXPORTER" --out "$COOKIE_EXPORT"; then
  notify "JumpTo: cookie refresh needs help" \
    "Auto-export failed (Firefox profile 'jumpto' not logged in?). Open it, sign into YouTube, and the next run will pick it up." warning
  exit 1
fi

# --- 3 + 4. Deploy (sanity check + upload + restart + verify + ntfy) -------
"$DEPLOYER" --force "$COOKIE_EXPORT"
rc=$?

if [ "$rc" -ne 0 ]; then
  notify "JumpTo: auto-refresh did not verify" \
    "Cookies were replaced but the probe still fails. Likely need a one-time re-login in the Firefox 'jumpto' profile." exclamation
fi

exit "$rc"