"""Screenshot capture and audit trail management."""

from datetime import datetime
from pathlib import Path
from typing import Dict

from playwright.async_api import Page

from config.settings import Settings
from core.models import TestCase


class Auditor:
    """Manages screenshot capture for audit trail purposes."""

    def __init__(self, page: Page, settings: Settings):
        self.page = page
        self.settings = settings

    async def capture_before(self, tc: TestCase) -> Path:
        """Capture screenshot before triggering test case.

        Args:
            tc: Test case being executed.

        Returns:
            Path to saved screenshot.
        """
        path = self._get_screenshot_path(tc, "before")
        await self.page.screenshot(path=path, full_page=True)
        tc.screenshots["before"] = path
        return path

    async def capture_during(self, tc: TestCase) -> Path:
        """Capture screenshot during test case execution.

        Args:
            tc: Test case being executed.

        Returns:
            Path to saved screenshot.
        """
        path = self._get_screenshot_path(tc, "during")
        await self.page.screenshot(path=path, full_page=True)
        tc.screenshots["during"] = path
        return path

    async def capture_after(self, tc: TestCase) -> Path:
        """Capture screenshot after test case completion.

        Args:
            tc: Test case being executed.

        Returns:
            Path to saved screenshot.
        """
        path = self._get_screenshot_path(tc, "after")
        await self.page.screenshot(path=path, full_page=True)
        tc.screenshots["after"] = path
        return path

    async def capture_on_failure(self, tc: TestCase) -> Path:
        """Capture screenshot specifically on failure for diagnostics.

        Args:
            tc: Failed test case.

        Returns:
            Path to saved screenshot.
        """
        path = self._get_screenshot_path(tc, "failure")
        await self.page.screenshot(path=path, full_page=True)
        tc.screenshots["failure"] = path
        return path

    def _get_screenshot_path(self, tc: TestCase, phase: str) -> Path:
        """Generate screenshot file path.

        Args:
            tc: Test case.
            phase: Execution phase (before, during, after, failure).

        Returns:
            Path object for the screenshot file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{tc.id}_{phase}_{timestamp}.png"
        return self.settings.screenshots_dir / filename

    def get_all_screenshots(self, tc: TestCase) -> Dict[str, str]:
        """Get all screenshot paths for a test case as strings.

        Args:
            tc: Test case.

        Returns:
            Dictionary mapping phase names to screenshot paths.
        """
        return {k: str(v) for k, v in tc.screenshots.items()}
