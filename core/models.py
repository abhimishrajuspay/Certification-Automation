"""Core data models for CZ Certification Automation."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from core.enums import FailureCategory, TestStatus


@dataclass
class TestCase:
    """Represents a single certification test case."""

    id: str
    description: str
    expected_status: str
    payload_requirements: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    play_button_selector: Optional[str] = None

    # Execution state
    status: TestStatus = TestStatus.PENDING
    failure_category: Optional[FailureCategory] = None

    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Artifacts
    screenshots: Dict[str, Path] = field(default_factory=dict)
    pod_logs: Optional[str] = None
    portal_logs: Optional[str] = None

    # LLM tracking
    llm_attempts: List[dict] = field(default_factory=list)
    correlation_ids: List[str] = field(default_factory=list)

    # Results
    generated_curl_command: Optional[str] = None
    execution_output: Optional[str] = None
    execution_error: Optional[str] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict:
        """Serialize to dictionary for reporting."""
        return {
            "id": self.id,
            "description": self.description,
            "expected_status": self.expected_status,
            "payload_requirements": self.payload_requirements,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "failure_category": (
                self.failure_category.value if self.failure_category else None
            ),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "screenshots": {k: str(v) for k, v in self.screenshots.items()},
            "pod_logs": self.pod_logs,
            "portal_logs": self.portal_logs,
            "llm_attempt_count": len(self.llm_attempts),
            "correlation_ids": self.correlation_ids,
            "generated_curl_command": self.generated_curl_command,
            "execution_output": self.execution_output,
            "execution_error": self.execution_error,
        }


@dataclass
class TestSuiteResult:
    """Aggregated results for the entire test suite."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    pending: int = 0
    test_cases: List[TestCase] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def update_counts(self) -> None:
        """Recalculate counts from test_cases list."""
        self.total = len(self.test_cases)
        self.passed = sum(1 for tc in self.test_cases if tc.status == TestStatus.PASSED)
        self.failed = sum(1 for tc in self.test_cases if tc.status == TestStatus.FAILED)
        self.skipped = sum(1 for tc in self.test_cases if tc.status == TestStatus.SKIPPED)
        self.pending = sum(1 for tc in self.test_cases if tc.status == TestStatus.PENDING)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate total suite duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict:
        """Serialize to dictionary for reporting."""
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "pending": self.pending,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
        }
