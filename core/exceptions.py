"""Custom exceptions for CZ Certification Automation."""


class CZAutomationError(Exception):
    """Base exception for all CZ automation errors."""

    pass


class ScraperError(CZAutomationError):
    """Raised when scraper operations fail."""

    pass


class NavigationError(ScraperError):
    """Raised when page navigation fails."""

    pass


class ExtractionError(ScraperError):
    """Raised when data extraction from the portal fails."""

    pass


class TriggerError(ScraperError):
    """Raised when clicking the Play button or triggering events fails."""

    pass


class LLMAgentError(CZAutomationError):
    """Raised when LLM agent operations fail."""

    pass


class ContextLoadingError(LLMAgentError):
    """Raised when loading repository context fails."""

    pass


class PayloadGenerationError(LLMAgentError):
    """Raised when LLM fails to generate a valid payload."""

    pass


class ExecutionError(CZAutomationError):
    """Raised when executing generated commands fails."""

    pass


class K8SError(CZAutomationError):
    """Raised when Kubernetes operations fail."""

    pass


class PortForwardError(K8SError):
    """Raised when port-forwarding fails or drops."""

    pass


class ValidationError(CZAutomationError):
    """Raised when validation operations fail."""

    pass


class UIValidationError(ValidationError):
    """Raised when UI state validation fails."""

    pass


class TimeoutError(CZAutomationError):
    """Raised when operations exceed their timeout."""

    pass


class StateMachineError(CZAutomationError):
    """Raised when state machine transitions fail."""

    pass
