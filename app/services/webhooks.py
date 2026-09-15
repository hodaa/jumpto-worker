"""Completion webhook URL building for cloud provider jobs."""

from urllib.parse import urlencode


def build_assembly_webhook_url(settings, job_id: str, provider_name: str) -> str:
    """Build the public completion-callback URL for a provider job.

    The backend exposes a fixed webhook base URL; job context is appended so
    its receiver can resume the right job. Returns ``""`` when webhooks are not
    armed (no base URL configured or no job id), preserving the legacy
    in-worker retry behaviour.
    """
    base = str(getattr(settings, "assembly_webhook_base_url", "") or "").strip()
    if not base or not job_id:
        return ""
    query = urlencode({"job_id": job_id, "provider": provider_name})
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}{query}"
