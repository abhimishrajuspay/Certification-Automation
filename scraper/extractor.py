"""Test case metadata extraction from CZ portal DOM."""

import re
from typing import List, Optional

from playwright.async_api import ElementHandle, Page

from core.exceptions import ExtractionError
from core.models import TestCase


class Extractor:
    """Extracts test case metadata from the CZ portal page."""

    # Flexible selectors - will try multiple patterns
    TC_ROW_SELECTORS = [
        '[data-testid="test-case-row"]',
        '.test-case-item',
        'tr[data-tc-id]',
        '.certification-test-case',
        '[class*="test-case"]',
    ]

    TC_ID_SELECTORS = [
        '[data-tc-id]',
        '.tc-id',
        '.test-case-id',
        'td:first-child',
    ]

    DESCRIPTION_SELECTORS = [
        '.tc-description',
        '.test-case-desc',
        'td:nth-child(2)',
        '[data-field="description"]',
    ]

    STATUS_SELECTORS = [
        '.tc-status',
        '.expected-status',
        'td:nth-child(3)',
        '[data-field="expected-status"]',
    ]

    DEPENDENCY_SELECTORS = [
        '[data-dependencies]',
        '.tc-dependencies',
        'td:nth-child(4)',
    ]

    PLAY_BUTTON_SELECTORS = [
        'button:has-text("Play")',
        '.btn-play',
        '[data-action="play"]',
        '.play-button',
        'button[class*="play"]',
    ]

    def __init__(self, page: Page):
        self.page = page

    async def extract_all_test_cases(self) -> List[TestCase]:
        """Extract all test cases from the current page.

        Returns:
            List of TestCase objects.

        Raises:
            ExtractionError: If extraction fails.
        """
        test_cases: List[TestCase] = []

        try:
            # Find all test case rows
            rows = await self._find_test_case_rows()

            for row in rows:
                tc = await self._extract_single_test_case(row)
                if tc:
                    test_cases.append(tc)

            return test_cases

        except Exception as e:
            raise ExtractionError(f"Failed to extract test cases: {e}")

    async def _find_test_case_rows(self) -> List[ElementHandle]:
        """Find all test case row elements on the page."""
        for selector in self.TC_ROW_SELECTORS:
            try:
                rows = await self.page.query_selector_all(selector)
                if rows:
                    return rows
            except Exception:
                continue

        # Fallback: try to find table rows with identifiable data
        try:
            rows = await self.page.query_selector_all('table tbody tr')
            return rows
        except Exception:
            pass

        return []

    async def _extract_single_test_case(
        self, row: ElementHandle
    ) -> Optional[TestCase]:
        """Extract a single test case from a row element.

        Args:
            row: DOM element representing a test case row.

        Returns:
            TestCase object or None if extraction fails.
        """
        try:
            # Extract ID
            tc_id = await self._extract_field(row, self.TC_ID_SELECTORS)
            if not tc_id:
                return None

            # Extract description
            description = await self._extract_field(
                row, self.DESCRIPTION_SELECTORS, default=""
            )

            # Extract expected status
            expected_status = await self._extract_field(
                row, self.STATUS_SELECTORS, default=""
            )

            # Extract dependencies
            dependencies_str = await self._extract_field(
                row, self.DEPENDENCY_SELECTORS, default=""
            )
            dependencies = self._parse_dependencies(dependencies_str)

            # Find Play button selector for this row
            play_selector = await self._find_play_button_selector(row)

            return TestCase(
                id=tc_id.strip(),
                description=description.strip(),
                expected_status=expected_status.strip(),
                dependencies=dependencies,
                play_button_selector=play_selector,
            )

        except Exception as e:
            print(f"Warning: Failed to extract test case from row: {e}")
            return None

    async def _extract_field(
        self,
        row: ElementHandle,
        selectors: List[str],
        default: str = "",
    ) -> str:
        """Extract text content from a field using multiple selector strategies.

        Args:
            row: Parent row element.
            selectors: List of CSS selectors to try.
            default: Default value if all selectors fail.

        Returns:
            Extracted text or default value.
        """
        for selector in selectors:
            try:
                element = await row.query_selector(selector)
                if element:
                    text = await element.inner_text()
                    if text:
                        return text.strip()
            except Exception:
                continue
        return default

    def _parse_dependencies(self, deps_str: str) -> List[str]:
        """Parse dependency string into list of test case IDs.

        Args:
            deps_str: Raw dependency string (e.g., "TC-001, TC-002" or "TC-001").

        Returns:
            List of dependency test case IDs.
        """
        if not deps_str:
            return []

        # Split by common delimiters
        deps = re.split(r"[,;|\s]+", deps_str.strip())
        return [d.strip() for d in deps if d.strip() and d.strip().upper() != "NONE"]

    async def _find_play_button_selector(self, row: ElementHandle) -> Optional[str]:
        """Find the Play button associated with a test case row.

        Args:
            row: Test case row element.

        Returns:
            CSS selector string for the Play button, or None.
        """
        for selector in self.PLAY_BUTTON_SELECTORS:
            try:
                button = await row.query_selector(selector)
                if button:
                    # Return a selector that uniquely identifies this button
                    # We'll use the row as context when clicking
                    return selector
            except Exception:
                continue
        return None
