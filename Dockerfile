FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY app ./app

RUN pip install --no-cache-dir .

# Run the Celery worker (concurrency 8)
CMD ["celery", "-A", "app.tasks.celery_app.celery_app", "worker", "--loglevel=info", "--concurrency=8"]
