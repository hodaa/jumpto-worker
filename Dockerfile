FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install ffmpeg + Deno for yt-dlp YouTube EJS support
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        unzip \
        ca-certificates \
        ffmpeg \
    && curl -fsSL https://deno.land/install.sh | sh \
    && ln -s /root/.deno/bin/deno /usr/local/bin/deno \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY app ./app

RUN pip install --no-cache-dir .

# Install yt-dlp PO Token provider plugin
RUN pip install --no-cache-dir bgutil-ytdlp-pot-provider

# Debugging: debugpy lets the worker process accept a remote attach (VSCode
# "Debugger: Attach" on port 5678). Nothing listens on the port by default; a
# debug entrypoint must call debugpy.listen() first.
RUN pip install --no-cache-dir debugpy

EXPOSE 5678

CMD ["celery", "-A", "app.tasks.celery_app.celery_app", "worker", "--loglevel=info"]