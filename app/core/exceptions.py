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
    """Exception for external service failures."""

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
