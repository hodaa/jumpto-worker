# JumpTo Worker

Celery transcription worker for JumpTo. Communicates with the JumpTo backend
via its internal API to read job/video data and store transcripts.

## Setup

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# edit .env to point BACKEND_URL at your backend and set INTERNAL_API_KEY
```

## Run

```sh
celery -A app.tasks.celery_app.celery_app worker --loglevel=info --concurrency=8
```

## Test

```sh
pytest
```

## YouTube cookie auto-refresh (host-side, VPS)

When yt-dlp hits YouTube's "Sign in to confirm you're not a bot", the worker
touches the cookie-refresh marker (`COOKIE_REFRESH_MARKER_PATH`, default
`/var/lib/jumpto/state/refresh-requested`) instead of retrying blindly. A
host-side cron then re-exports cookies from the logged-in Chromium container
over its DevTools loopback endpoint and replaces `/etc/jumpto/cookies.txt`.

Flow:

1. Worker detects the bot-check in a `DownloadError` and touches the marker
   (see `app/providers/ytdlp.py`).
2. On the VPS, `scripts/cookies-refresh.sh` (cron, every 1 min) sees the
   marker and runs `docker exec chromium python3 /opt/jumpto/refresh_cookies.py`
   — which must run **inside** the container, since Chromium only exposes
   DevTools on its own loopback.
3. The export is written to `/etc/jumpto/fresh-cookies.txt`, atomically moved
   onto `/etc/jumpto/cookies.txt`, and the marker is cleared.

Deploy on the VPS:

- Build the extended image (adds `python3` + `websocket-client` and copies the
  exporter in): `docker build -f scripts/chromium-cdp-Dockerfile -t chromium-cdp scripts/`
- Recreate the `chromium` container from `chromium-cdp` keeping a signed-in
  YouTube session (see the Dockerfile header for the `docker run` line).
- Add the cron line from the `scripts/cookies-refresh.sh` header.
- `docker-compose.yml` already mounts `/etc/jumpto/state` (the marker dir)
  writable into the worker and keeps the cookie file read-only.
