"""Domain exceptions for the JumpTo worker service."""

from typing import Any


class DomainError(Exception):
    """Base exception for domain errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "DOMAIN_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)


class ExternalServiceError(DomainError):
    """Transient external-service failure (network, API, captions, timeout).

    The pipeline treats these as a soft miss: the job moves to the next
    transcript provider. The ``service`` details key names the failing
    provider/service. Permanent failures subclass
    :class:`PermanentExternalServiceError`; an async job still processing is
    signalled separately by ``TranscriptJobPending`` (in
    ``app.providers.models``), and failures talking to the producer API use
    :class:`BackendCommunicationError`.
    """

    def __init__(
        self,
        message: str,
        *,
        service: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="EXTERNAL_SERVICE_ERROR",
            details={**({"service": service} if service else {}), **(details or {})},
        )


class PermanentExternalServiceError(ExternalServiceError):
    """External-service failure that fallback providers cannot recover from.

    Raised for credential/account errors and per-video errors that would fail
    every provider the same way; the pipeline fails the job instead of falling
    through to the next candidate.
    """


class BackendCommunicationError(DomainError):
    """Exception for failures while communicating with the backend API."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="BACKEND_COMMUNICATION_ERROR",
            details=details or {},
        )
