FROM python:3.12-slim AS builder

WORKDIR /app

COPY pyproject.toml ./
COPY app ./app

RUN pip install --no-cache-dir --prefix=/install . \
    && pip install --no-cache-dir --prefix=/install \
        bgutil-ytdlp-pot-provider


FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        ffmpeg \
    && curl -fsSL https://deno.land/install.sh | sh \
    && ln -s /root/.deno/bin/deno /usr/local/bin/deno \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only the installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app ./app

CMD ["celery", "-A", "app.tasks.celery_app.celery_app", "worker", "--loglevel=info"]