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
