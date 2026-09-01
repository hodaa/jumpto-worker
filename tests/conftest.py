"""Shared test fixtures for the JumpTo worker."""

import os


def pytest_configure(config) -> None:
    """Force a deterministic, offline-friendly environment for tests."""
    os.environ.setdefault("JUMPTO_LIVE_EXTERNAL_CALLS", "false")
    os.environ.setdefault("JUMPTO_TRANSCRIPT_MODE", "fake")
    os.environ.setdefault("BACKEND_URL", "http://backend.test:8000")
    os.environ.setdefault("INTERNAL_API_KEY", "test-key")
