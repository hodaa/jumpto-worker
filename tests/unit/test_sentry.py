"""Unit tests for the Sentry initialisation helper."""

from unittest.mock import Mock

import pytest

from app.core.sentry import init_sentry


class TestInitSentry:
    """Tests for ``init_sentry`` behaviour."""

    def test_no_op_without_dsn(self, monkeypatch: pytest.MonkeyPatch, mocker) -> None:
        mocker.patch("sentry_sdk.init")
        monkeypatch.delenv("SENTRY_DSN", raising=False)
        monkeypatch.setattr("app.core.sentry.get_settings", lambda: Mock(sentry_dsn=""))

        init_sentry()

        from sentry_sdk import init as sentry_init  # noqa: F401

        sentry_init.assert_not_called()

    def test_initialises_when_dsn_configured(self, monkeypatch: pytest.MonkeyPatch, mocker) -> None:
        mock_init = mocker.patch("sentry_sdk.init")
        settings = Mock(
            sentry_dsn="https://key@o1.ingest.sentry.io/123",
            environment="development",
            sentry_traces_sample_rate=0.0,
        )
        monkeypatch.setattr("app.core.sentry.get_settings", lambda: settings)

        init_sentry()

        mock_init.assert_called_once()
        kwargs = mock_init.call_args.kwargs
        assert kwargs["dsn"] == "https://key@o1.ingest.sentry.io/123"
        assert kwargs["environment"] == "development"
        assert kwargs["traces_sample_rate"] == settings.sentry_traces_sample_rate == 0.0
