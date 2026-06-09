"""Enumeration definitions for CZ Certification Automation."""

from enum import Enum, auto


class TestStatus(str, Enum):
    """Possible statuses for a test case throughout its lifecycle."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class FailureCategory(str, Enum):
    """Categories for test case failures."""

    TIMEOUT = "TIMEOUT"
    SCHEMA_VALIDATION = "SCHEMA_VALIDATION"
    INCORRECT_RESPONSE = "INCORRECT_RESPONSE"
    NETWORK_ERROR = "NETWORK_ERROR"
    UNKNOWN = "UNKNOWN"


class ExecutionMode(str, Enum):
    """Execution modes for the test suite."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class ReviewAction(str, Enum):
    """Possible actions from human review."""

    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"
