#!/usr/bin/env bash
#
# JumpTo cookie deployer (runs on the Mac).
#
# Takes the freshest browser cookie export, ships it to the VPS, replaces
# /etc/jumpto/cookies.txt, restarts the worker, re-probes to VERIFY the fix,
# then notifies via ntfy. Idempotent: does nothing if the export is unchanged
# since the last successful deploy.
#
# Edit the CONFIG block below (SSH_TARGET at minimum).
#
# Usage:
#   redeploy_cookies.sh [file]          deploy once (default: ~/Downloads/cookies.txt)
#   redeploy_cookies.sh --force [file]  deploy even if unchanged since last run
#   redeploy_cookies.sh --watch [file]  loop: watch for a new export every WATCH_INTERVAL s
#
# Suggested from the VPS cron alert: run it, then forget the mechanics.

set -u

# ---- CONFIG: adjust these -------------------------------------------------
SSH_TARGET="opc@1.2.3.4"                 # VPS ssh target (e.g. opc@x.y.z.w)
REMOTE_TMP="/tmp/jumpto-cookies.new.txt"  # staging path on the VPS
REMOTE_FINAL="/etc/jumpto/cookies.txt"    # host-side bind-mount source
REMOTE_DIR="jumpto-worker"                # compose directory on the VPS ($HOME/jumpto-worker)
# --------------------------------------------------------------------------

LOCAL_COOKIES="$HOME/Downloads/cookies.txt"
MODE="once"

while [ $# -gt 0 ]; do
  case "$1" in
    --force) MODE="force" ;;
    --watch) MODE="watch" ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) LOCAL_COOKIES="$1" ;;
  esac
  shift
done

NTFY_URL="${NTFY_URL:-https://ntfy.sh}"
NTFY_TOPIC="${NTFY_TOPIC:-jumpto-cookie-alerts}"
STATE_FILE="${STATE_FILE:-$HOME/.jumpto_cookies_deployed_state}"
WATCH_INTERVAL="${WATCH_INTERVAL:-60}"

notify() {
  local title="$1"
  local msg="$2"
  curl -s -L -o /dev/null \
    -H "Title: $title" \
    -H "Priority: high" \
    -H "Tags: $3" \
    -d "$msg" \
    "$NTFY_URL/$NTFY_TOPIC"
}

deploy() {
  local file="$1"

  if [ ! -f "$file" ]; then
    echo "No cookie export at: $file" >&2
    return 1
  fi

  # Signature to skip re-deploying an unchanged file.
  local sig
  sig="$(stat -f '%z:%m' "$file")"
  if [ "$(cat "$STATE_FILE" 2>/dev/null || true)" = "$sig" ] && [ "$MODE" != "force" ]; then
    echo "Cookie export unchanged since last deploy; skipping."
    return 0
  fi

  echo "Uploading $file -> $SSH_TARGET ..."
  scp -q "$file" "$SSH_TARGET:$REMOTE_TMP" || { notify "JumpTo: cookie deploy failed" "scp upload to $SSH_TARGET failed." warning; return 1; }

  echo "Replacing $REMOTE_FINAL and restarting worker ..."
  ssh "$SSH_TARGET" "sudo cp $REMOTE_TMP $REMOTE_FINAL && cd \$HOME/$REMOTE_DIR && docker compose up -d --no-deps worker" \
    || { notify "JumpTo: cookie deploy failed" "Failed to replace/restart on $SSH_TARGET." warning; return 1; }

  echo "Verifying with a live probe ..."
  if ssh "$SSH_TARGET" "cd \$HOME/$REMOTE_DIR && scripts/cookie_health.sh" 2>/dev/null; then
    echo "$sig" > "$STATE_FILE"
    notify "JumpTo: cookies deployed" "Fresh cookies uploaded, worker restarted, probe healthy." white_check_mark
    echo "OK: cookies deployed and verified."
    return 0
  fi

  notify "JumpTo: cookie deploy did not verify" "Cookies replaced but the probe still fails. Check cookie_health.sh output." exclamation
  return 1
}

case "$MODE" in
  watch)
    echo "Watching $LOCAL_COOKIES every ${WATCH_INTERVAL}s (Ctrl-C to stop)"
    while true; do
      deploy "$LOCAL_COOKIES"
      sleep "$WATCH_INTERVAL"
    done
    ;;
  *)
    deploy "$LOCAL_COOKIES"
    ;;
esac