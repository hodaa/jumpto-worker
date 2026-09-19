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

CMD ["celery", "-A", "app.tasks.celery_app.celery_app", "worker", "--loglevel=info"]