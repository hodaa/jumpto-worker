"""OpenAI-backed transcript cleanup for lecture notes.

The model reads the ASR transcript and writes structured notes, reconstructing
equations when the lecture contains them without reading video frames. The API
key stays in the worker's environment; failures are reported as
external-service errors, never exposed to the browser.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.client.http import get_shared_http_client
from app.core.config import Settings
from app.core.exceptions import ExternalServiceError, PermanentExternalServiceError
from app.core.logging import get_logger
from app.integrations.equation_review import (
    VALID_VERDICTS,
    EquationReview,
    equation_review_prompt,
    reviews_from_payload,
)
from app.integrations.notes_ai import (
    MAX_RETRIES,
    RETRYABLE_STATUS_CODES,
    blocks_from_payload,
    notes_prompt,
    overview_prompt,
    parse_json,
    retry_delay,
)
from app.models import NoteBlock

logger = get_logger(__name__)

_SERVICE = "openai"
_REQUEST_TIMEOUT_SECONDS = 180
_MAX_OUTPUT_TOKENS = 8192
_NOTES_SYSTEM = "You write concise structured lecture notes. Return JSON only."
_REVIEW_SYSTEM = "You are a mathematical reviewer. Return JSON only."

_OPENAI_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["heading", "paragraph", "bullet", "equation"],
                    },
                    "text": {"type": "string"},
                    "note": {"type": "string"},
                    "latex": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["type", "text", "note", "latex", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["blocks"],
    "additionalProperties": False,
}

_OVERVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"overview": {"type": "string"}},
    "required": ["overview"],
    "additionalProperties": False,
}

_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "verdict": {"type": "string", "enum": sorted(VALID_VERDICTS)},
                    "corrected_latex": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "verdict", "corrected_latex", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["reviews"],
    "additionalProperties": False,
}


def _quota_exhausted(response: httpx.Response) -> bool:
    """Return True when a 429 means an exhausted quota, not a short burst."""
    try:
        payload = response.json()
    except ValueError:
        return False
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return False
    code = str(error.get("code", "")).lower()
    message = str(error.get("message", "")).lower()
    return code in {"insufficient_quota", "billing_hard_limit_reached"} or ("quota" in message)


def _raise_quota(response: httpx.Response) -> None:
    """Stop retrying when the account has no quota left."""
    logger.warning("OpenAI quota exhausted", status=response.status_code)
    raise ExternalServiceError(
        "Lecture notes AI quota is exhausted",
        service=_SERVICE,
        details={"status": response.status_code, "quota_exhausted": True},
    )


class OpenAINotesAI:
    """Transcript cleanup through the OpenAI Chat Completions API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Wire the client; an injected HTTP client is used by tests."""
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = client or get_shared_http_client(timeout=_REQUEST_TIMEOUT_SECONDS)

    @classmethod
    def from_settings(
        cls, settings: Settings, client: httpx.AsyncClient | None = None
    ) -> OpenAINotesAI:
        """Build the client from worker settings."""
        return cls(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_api_url,
            client=client,
        )

    @property
    def configured(self) -> bool:
        """Whether an API key is available for cleanup."""
        return bool(self._api_key)

    async def overview(self, *, title: str, language: str | None, transcript: str) -> str:
        """Return a brief overview of the lecture in its own language."""
        prompt = overview_prompt(title, language, transcript)
        payload = await self._post(
            _completion_body(
                self._model, prompt, _OVERVIEW_SCHEMA, "lecture_overview", _NOTES_SYSTEM
            )
        )
        return str(payload.get("overview", "")).strip()

    async def clean_chunk(
        self,
        *,
        title: str,
        language: str | None,
        transcript: str,
        start_seconds: int,
        end_seconds: int,
    ) -> list[NoteBlock]:
        """Clean one transcript window into note blocks."""
        prompt = notes_prompt(title, language, transcript, start_seconds, end_seconds)
        payload = await self._post(
            _completion_body(self._model, prompt, _OPENAI_SCHEMA, "lecture_notes", _NOTES_SYSTEM)
        )
        return blocks_from_payload(payload)

    async def review_equations(
        self, transcript: str, equations: list[NoteBlock]
    ) -> list[EquationReview]:
        """Verify one chunk's equations against its transcript."""
        if not equations:
            return []
        prompt = equation_review_prompt(transcript, equations)
        payload = await self._post(
            _completion_body(self._model, prompt, _REVIEW_SCHEMA, "equation_review", _REVIEW_SYSTEM)
        )
        return reviews_from_payload(payload, len(equations))

    async def _send(self, body: dict[str, Any]) -> httpx.Response:
        """POST one completion request with the key in the auth header."""
        url = f"{self._base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        return await self._client.post(url, headers=headers, json=body)

    async def _retry_network(self, attempt: int, exc: httpx.HTTPError) -> None:
        """Wait before retrying, or raise once the attempts are spent."""
        logger.warning("OpenAI request failed (network)", attempt=attempt + 1)
        if attempt + 1 >= MAX_RETRIES:
            raise ExternalServiceError(
                "Could not reach lecture notes AI", service=_SERVICE
            ) from exc
        await asyncio.sleep(retry_delay(attempt))

    async def _retry_status(self, attempt: int, response: httpx.Response) -> None:
        """Wait before retrying a transient status, or raise on the last try."""
        logger.warning(
            "OpenAI request failed (status)",
            attempt=attempt + 1,
            status=response.status_code,
        )
        if attempt + 1 >= MAX_RETRIES:
            raise ExternalServiceError(
                f"Lecture notes AI returned {response.status_code}", service=_SERVICE
            )
        await asyncio.sleep(retry_delay(attempt, response))

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST one completion request, retrying transient failures."""
        for attempt in range(MAX_RETRIES):
            try:
                response = await self._send(body)
            except httpx.HTTPError as exc:
                await self._retry_network(attempt, exc)
                continue
            if response.status_code in RETRYABLE_STATUS_CODES:
                if _quota_exhausted(response):
                    _raise_quota(response)
                await self._retry_status(attempt, response)
                continue
            if response.status_code >= 400:
                raise PermanentExternalServiceError(
                    "Lecture notes AI rejected the request",
                    service=_SERVICE,
                    details={"status": response.status_code},
                )
            return parse_json(_message_text(response.json()), _SERVICE)
        raise ExternalServiceError("Lecture notes AI request failed", service=_SERVICE)


def _completion_body(
    model: str,
    prompt: str,
    schema: dict[str, Any],
    name: str,
    system: str,
) -> dict[str, Any]:
    """Build the Chat Completions request body with strict JSON output."""
    return {
        "model": model,
        "messages": _messages(prompt, system),
        "max_completion_tokens": _MAX_OUTPUT_TOKENS,
        "response_format": _json_schema_format(name, schema),
    }


def _messages(prompt: str, system: str) -> list[dict[str, str]]:
    """Build the system and user messages for one call."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


def _json_schema_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Describe the strict JSON schema expected from the model."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def _message_text(payload: dict[str, Any]) -> str:
    """Extract the assistant's text from the first choice."""
    choices = payload.get("choices") or []
    if not choices:
        raise ExternalServiceError("Lecture notes AI returned no choices", service=_SERVICE)
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    if not content:
        raise ExternalServiceError("Lecture notes AI returned no text", service=_SERVICE)
    return str(content)
