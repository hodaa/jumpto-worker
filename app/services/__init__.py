"""Worker service layer: orchestration extracted from the Celery task module.

Each module owns one responsibility:

- ``event_loop`` — process-wide asyncio loop lifecycle for async services.
- ``webhooks`` — completion-callback URL building for provider jobs.
- ``submissions`` — transcript submission payload building.
- ``jobs`` — the backend client lifecycle for a single job (``JobService``).
- ``pipeline`` — the transcription pipeline orchestration flow.

The Celery task module stays a thin scheduling shell over these services.
"""
