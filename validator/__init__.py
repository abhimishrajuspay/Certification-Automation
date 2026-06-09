"""Main validator orchestrator for test case validation."""

from typing import Optional

from playwright.async_api import Page

from config.settings import Settings
from core.enums import FailureCategory, TestStatus
from core.exceptions import ValidationError
from core.k8s_bridge import LogFetcher
from core.models import TestCase
from validator.ui_validator import Categorizer, UIValidator


class Validator:
    """Main validator orchestrating UI and log validation."""

    def __init__(self, page: Page, settings: Settings):
        self.ui_validator = UIValidator(page, settings)
        self.log_fetcher = LogFetcher(settings)
        self.categorizer = Categorizer()
        self.settings = settings

    async def validate_test_case(self, tc: TestCase) -> TestCase:
        """Validate a test case through UI and pod logs.

        Args:
            tc: Test case to validate.

        Returns:
            Updated TestCase with validation results.
        """
        try:
            # Step 1: Validate UI status
            ui_status = await self.ui_validator.validate_status(tc)
            tc.status = ui_status

            if tc.status == TestStatus.FAILED:
                # Step 2: Fetch pod logs for correlation IDs
                if tc.correlation_ids:
                    try:
                        logs = self.log_fetcher.fetch_logs(tc.correlation_ids)
                        tc.pod_logs = logs
                    except Exception as e:
                        print(f"Warning: Could not fetch pod logs: {e}")

                # Step 3: Click logs icon on portal for detailed error info
                # This will be handled by the scraper

                # Step 4: Categorize failure
                tc.failure_category = self.categorizer.categorize(tc)

            elif tc.status == TestStatus.PASSED:
                # Optionally verify pod logs for successful callbacks
                if tc.correlation_ids:
                    try:
                        logs = self.log_fetcher.fetch_logs(tc.correlation_ids)
                        tc.pod_logs = logs
                    except Exception:
                        pass

            return tc

        except Exception as e:
            tc.status = TestStatus.FAILED
            tc.execution_error = f"Validation failed: {e}"
            tc.failure_category = FailureCategory.UNKNOWN
            return tc
