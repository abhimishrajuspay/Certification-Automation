"""Main scraper orchestrator coordinating all scraping components."""

import asyncio
from typing import List, Optional

from playwright.async_api import ElementHandle

from config.settings import Settings
from core.dag_engine import TestSuiteDAG
from core.models import TestCase
from scraper.auditor import Auditor
from scraper.browser import BrowserManager
from scraper.extractor import Extractor
from scraper.navigator import Navigator
from scraper.trigger import Trigger


class Scraper:
    """Main scraper orchestrator for CZ portal certification test cases."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.browser_manager: Optional[BrowserManager] = None
        self.navigator: Optional[Navigator] = None
        self.extractor: Optional[Extractor] = None
        self.trigger: Optional[Trigger] = None
        self.auditor: Optional[Auditor] = None

    async def initialize(self) -> None:
        """Initialize all scraping components."""
        self.browser_manager = BrowserManager(self.settings)
        page = await self.browser_manager.start()

        self.navigator = Navigator(page, self.settings)
        self.extractor = Extractor(page)
        self.trigger = Trigger(page, self.settings)
        self.auditor = Auditor(page, self.settings)

    async def shutdown(self) -> None:
        """Clean up all scraping resources."""
        if self.browser_manager:
            await self.browser_manager.stop()

    async def scrape_all_test_cases(self) -> TestSuiteDAG:
        """Scrape all test cases across all pages and build dependency DAG.

        Returns:
            TestSuiteDAG containing all test cases with resolved dependencies.
        """
        await self.navigator.navigate_to_test_suite()

        total_pages = await self.navigator.get_total_pages()
        all_test_cases: List[TestCase] = []

        for page_num in range(1, total_pages + 1):
            if page_num > 1:
                await self.navigator.navigate_to_page(page_num)

            page_cases = await self.extractor.extract_all_test_cases()
            all_test_cases.extend(page_cases)

            # Rate limiting between pages
            if page_num < total_pages:
                await asyncio.sleep(self.settings.rate_limit_delay_ms / 1000)

        # Build dependency DAG
        dag = TestSuiteDAG()
        dag.build(all_test_cases)

        return dag

    async def execute_test_case(self, tc: TestCase) -> TestCase:
        """Execute a single test case by clicking Play and monitoring.

        Args:
            tc: Test case to execute.

        Returns:
            Updated TestCase with execution results.
        """
        from core.enums import TestStatus
        from datetime import datetime

        tc.status = TestStatus.RUNNING
        tc.started_at = datetime.now()

        try:
            # Find the test case row
            tc_row = await self._find_test_case_row(tc.id)
            if not tc_row:
                raise Exception(f"Test case row not found for {tc.id}")

            # Capture before screenshot
            await self.auditor.capture_before(tc)

            # Click Play
            await self.trigger.click_play(tc_row)

            # Capture during screenshot
            await self.auditor.capture_during(tc)

            # Wait for completion
            completed = await self.trigger.wait_for_completion(
                timeout=self.settings.max_tc_execution_timeout_sec
            )

            if not completed:
                tc.status = TestStatus.FAILED
                tc.execution_error = "Timeout waiting for execution completion"
            else:
                # Status will be determined by validator
                pass

        except Exception as e:
            tc.status = TestStatus.FAILED
            tc.execution_error = str(e)
            await self.auditor.capture_on_failure(tc)

        finally:
            tc.completed_at = datetime.now()

        return tc

    async def _find_test_case_row(self, tc_id: str) -> Optional[ElementHandle]:
        """Find a test case row by ID on the current page.

        Args:
            tc_id: Test case ID.

        Returns:
            Row element or None.
        """
        # Try various selector strategies
        selectors = [
            f'[data-tc-id="{tc_id}"]',
            f'tr:has-text("{tc_id}")',
            f'[data-testid="test-case-{tc_id}"]',
        ]

        for selector in selectors:
            try:
                row = await self.browser_manager.page.query_selector(selector)
                if row:
                    return row
            except Exception:
                continue

        return None

    async def click_logs_icon(self, tc_id: str) -> Optional[str]:
        """Click the Logs icon for a failed test case and capture content.

        Args:
            tc_id: Test case ID.

        Returns:
            Log content or None.
        """
        try:
            row = await self._find_test_case_row(tc_id)
            if not row:
                return None

            # Find logs icon/button
            log_selectors = [
                'button:has-text("Logs")',
                '.btn-logs',
                '[data-action="logs"]',
                '.logs-icon',
                'i[class*="log"]',
            ]

            for selector in log_selectors:
                try:
                    log_button = await row.query_selector(selector)
                    if log_button:
                        await log_button.evaluate("element => element.click()")
                        await asyncio.sleep(0.5)

                        # Try to capture log content from modal/popup
                        log_content = await self._capture_log_content()
                        return log_content
                except Exception:
                    continue

        except Exception as e:
            print(f"Warning: Failed to click logs icon for {tc_id}: {e}")

        return None

    async def _capture_log_content(self) -> Optional[str]:
        """Capture log content from a modal or popup.

        Returns:
            Log text content or None.
        """
        modal_selectors = [
            '.modal-body',
            '.log-content',
            '[data-testid="log-modal"]',
            '.logs-container',
            'pre',
            'code',
        ]

        for selector in modal_selectors:
            try:
                element = await self.browser_manager.page.wait_for_selector(
                    selector, timeout=2000
                )
                if element:
                    text = await element.inner_text()
                    if text:
                        return text.strip()
            except Exception:
                continue

        return None
