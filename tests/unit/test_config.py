"""Unit tests for worker configuration settings."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings


class TestCeleryWorkerConcurrencySetting:
    """Tests for the Celery worker concurrency setting."""

    def test_default_is_eight(self) -> None:
        assert Settings(_env_file=None).celery_worker_concurrency == 8

    def test_reads_configured_value_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("CELERY_WORKER_CONCURRENCY", "3")
        assert Settings(_env_file=None).celery_worker_concurrency == 3

    def test_rejects_non_positive_values(self) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, celery_worker_concurrency=0)
