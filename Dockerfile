FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Deno for yt-dlp YouTube EJS support
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl unzip ca-certificates \
    && curl -fsSL https://deno.land/install.sh | sh \
    && ln -s /root/.deno/bin/deno /usr/local/bin/deno \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
COPY pyproject.toml ./
COPY app ./app

RUN pip install --no-cache-dir .

# Run the Celery worker (concurrency 8)
CMD ["celery", "-A", "app.tasks.celery_app.celery_app", "worker", "--loglevel=info", "--concurrency=8"]
