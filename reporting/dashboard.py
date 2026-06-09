"""HTML dashboard generator for CZ Certification Automation results."""

import json
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape

from config.settings import Settings
from core.enums import FailureCategory, TestStatus
from core.models import TestCase, TestSuiteResult


class DashboardGenerator:
    """Generates HTML dashboard reports."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def generate(self, result: TestSuiteResult, output_path: Path) -> Path:
        """Generate HTML dashboard report.

        Args:
            result: Test suite execution results.
            output_path: Path to save the HTML file.

        Returns:
            Path to generated HTML file.
        """
        template = self.env.get_template("dashboard.html")

        html = template.render(
            result=result,
            test_cases=result.test_cases,
            stats=self._compute_stats(result),
            duration=self._format_duration(result.duration_seconds),
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

        return output_path

    def _compute_stats(self, result: TestSuiteResult) -> Dict[str, Any]:
        """Compute statistics for the dashboard."""
        stats = {
            "total": result.total,
            "passed": result.passed,
            "failed": result.failed,
            "skipped": result.skipped,
            "pending": result.pending,
            "pass_rate": (result.passed / result.total * 100) if result.total > 0 else 0,
            "fail_rate": (result.failed / result.total * 100) if result.total > 0 else 0,
        }

        # Category breakdown
        categories = {}
        for tc in result.test_cases:
            if tc.failure_category:
                cat = tc.failure_category.value
                categories[cat] = categories.get(cat, 0) + 1

        stats["categories"] = categories
        return stats

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds is None:
            return "N/A"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")

        return " ".join(parts)
