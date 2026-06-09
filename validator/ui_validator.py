"""Validator for UI status and pod log verification."""

import asyncio
import re
from typing import Optional

from playwright.async_api import Page

from config.settings import Settings
from core.enums import FailureCategory, TestStatus
from core.exceptions import UIValidationError
from core.models import TestCase


class UIValidator:
    """Validates test case status from CZ portal UI."""

    def __init__(self, page: Page, settings: Settings):
        self.page = page
        self.settings = settings

    async def validate_status(self, tc: TestCase) -> TestStatus:
        """Check the portal UI for test case status.

        Args:
            tc: Test case to validate.

        Returns:
            TestStatus based on UI indicators.
        """
        try:
            # Find the test case row
            row = await self._find_test_case_row(tc.id)
            if not row:
                tc.execution_error = "Test case row not found during validation"
                return TestStatus.FAILED

            # Check for passed indicators
            if await self._has_passed_indicator(row):
                return TestStatus.PASSED

            # Check for failed indicators
            if await self._has_failed_indicator(row):
                return TestStatus.FAILED

            # Default to failed if unclear
            return TestStatus.FAILED

        except Exception as e:
            tc.execution_error = f"Validation error: {e}"
            return TestStatus.FAILED

    async def _find_test_case_row(self, tc_id: str) -> Optional[any]:
        """Find test case row by ID."""
        selectors = [
            f'[data-tc-id="{tc_id}"]',
            f'tr:has-text("{tc_id}")',
        ]

        for selector in selectors:
            try:
                row = await self.page.query_selector(selector)
                if row:
                    return row
            except Exception:
                continue

        return None

    async def _has_passed_indicator(self, row: any) -> bool:
        """Check if row shows passed status."""
        passed_selectors = [
            '.status-passed',
            '[data-status="passed"]',
            '.fa-check-circle',
            '.green-tick',
            '.icon-success',
            'img[src*="pass"]',
            'svg[class*="success"]',
        ]

        for selector in passed_selectors:
            try:
                element = await row.query_selector(selector)
                if element and await element.is_visible():
                    return True
            except Exception:
                continue

        # Check inline styles for green color
        try:
            style = await row.get_attribute("style")
            if style and ("green" in style.lower() or "#00" in style):
                return True
        except Exception:
            pass

        return False

    async def _has_failed_indicator(self, row: any) -> bool:
        """Check if row shows failed status."""
        failed_selectors = [
            '.status-failed',
            '[data-status="failed"]',
            '.fa-times-circle',
            '.red-cross',
            '.icon-error',
            'img[src*="fail"]',
            'svg[class*="error"]',
        ]

        for selector in failed_selectors:
            try:
                element = await row.query_selector(selector)
                if element and await element.is_visible():
                    return True
            except Exception:
                continue

        # Check inline styles for red color
        try:
            style = await row.get_attribute("style")
            if style and ("red" in style.lower() or "#ff" in style):
                return True
        except Exception:
            pass

        return False


class Categorizer:
    """Categorizes test case failures by type."""

    def categorize(self, tc: TestCase) -> FailureCategory:
        """Determine failure category based on test case data.

        Args:
            tc: Failed test case.

        Returns:
            FailureCategory enum value.
        """
        error_lower = (tc.execution_error or "").lower()
        output_lower = (tc.execution_output or "").lower()
        logs_lower = (tc.pod_logs or "").lower()

        # Check for timeout
        if any(keyword in error_lower for keyword in ["timeout", "timed out", "deadline"]):
            return FailureCategory.TIMEOUT

        if tc.duration_seconds and tc.duration_seconds >= 50:
            return FailureCategory.TIMEOUT

        # Check for schema validation errors
        if any(keyword in logs_lower for keyword in [
            "schema", "xsd", "validation failed", "invalid xml",
            "malformed", "parse error", "schema violation"
        ]):
            return FailureCategory.SCHEMA_VALIDATION

        # Check for incorrect response
        if any(keyword in logs_lower for keyword in [
            "incorrect", "mismatch", "unexpected", "wrong response",
            "assertion failed", "expected", "but got"
        ]):
            return FailureCategory.INCORRECT_RESPONSE

        # Check for network errors
        if any(keyword in error_lower for keyword in [
            "connection", "network", "refused", "unreachable",
            "dns", "socket", "econnrefused", "timeout"
        ]):
            return FailureCategory.NETWORK_ERROR

        # Default
        return FailureCategory.UNKNOWN
