"""Send a test exception to Sentry to verify the worker's error tracking."""

import sys

import sentry_sdk

from app.core.config import get_settings
from app.core.sentry import init_sentry


def main() -> int:
    init_sentry()
    settings = get_settings()

    if not settings.sentry_dsn:
        print("Sentry is disabled: SENTRY_DSN is not set in .env")
        return 1

    try:
        raise RuntimeError("jumpto-worker Sentry test event")
    except RuntimeError:
        sentry_sdk.capture_exception()

    sentry_sdk.flush()
    host = settings.sentry_dsn.split("@")[-1]
    print(f"Sentry test event sent to {host}")
    return 0


if __name__ == "__main__":
    sys.exit(main())