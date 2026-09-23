FROM python:3.12-slim AS builder

WORKDIR /app

COPY pyproject.toml ./
COPY app ./app

RUN pip install --no-cache-dir --prefix=/install . \
    && pip install --no-cache-dir --prefix=/install \
        bgutil-ytdlp-pot-provider


FROM denoland/deno:bin AS deno


FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Deno binary
COPY --from=deno /deno /usr/local/bin/deno

# Copy Python packages from builder
COPY --from=builder /install /usr/local

# Copy application
COPY app ./app

# Run celery as an unprivileged user. Both writable runtime paths live off
# the app user: the on-disk transcript cache and the cookie-refresh marker
# dir. /app only needs read access, so the owner is left as root with
# world-readable perms from COPY. The marker dir is a host bind mount in
# docker-compose, so a fixed UID (10001) keeps ownership deterministic for
# the host-side cookie-refresh cron.
RUN groupadd --system --gid 10001 jumpto \
    && useradd --system --create-home --shell /usr/sbin/nologin --uid 10001 --gid 10001 jumpto \
    && mkdir -p /var/tmp/jumpto-transcript-cache /var/lib/jumpto/state \
    && chown -R jumpto:jumpto /var/tmp/jumpto-transcript-cache /var/lib/jumpto/state

USER jumpto

CMD ["celery", "-A", "app.tasks.celery_app.celery_app", "worker", "--loglevel=info"]
