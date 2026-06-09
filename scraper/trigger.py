"""Test case trigger mechanism for CZ portal Play buttons."""

import asyncio
from typing import Optional

from playwright.async_api import ElementHandle, Page

from config.settings import Settings
from core.exceptions import TriggerError


class Trigger:
    """Handles clicking the Play button and monitoring execution state."""

    def __init__(self, page: Page, settings: Settings):
        self.page = page
        self.settings = settings

    async def click_play(self, tc_row: ElementHandle, timeout: int = 55) -> bool:
        """Click the Play button for a test case and wait for execution.

        Args:
            tc_row: The test case row element containing the Play button.
            timeout: Maximum time to wait for execution (seconds).

        Returns:
            True if the click was successful and execution started.

        Raises:
            TriggerError: If the Play button cannot be found or clicked.
        """
        try:
            # Find the Play button within the row
            play_button = await self._find_play_button(tc_row)
            if not play_button:
                raise TriggerError("Play button not found in test case row")

            # Scroll into view and click
            await play_button.scroll_into_view_if_needed()
            await asyncio.sleep(0.2)  # Small delay for stability

            # Use JavaScript click to handle event listeners
            await play_button.evaluate("element => element.click()")

            # Wait for spinner or execution indicator
            await self._wait_for_execution_start(timeout=5)

            return True

        except Exception as e:
            if isinstance(e, TriggerError):
                raise
            raise TriggerError(f"Failed to click Play button: {e}")

    async def wait_for_completion(self, timeout: int = 60) -> bool:
        """Wait for test case execution to complete.

        Detects completion by:
        1. Disappearance of spinner/loading indicator
        2. Appearance of status icons (green tick / red cross)
        3. Timeout

        Args:
            timeout: Maximum wait time in seconds.

        Returns:
            True if execution completed within timeout.
        """
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            # Check for completion indicators
            is_complete = await self._check_completion_indicators()
            if is_complete:
                return True

            await asyncio.sleep(0.5)

        return False  # Timeout reached

    async def _find_play_button(
        self, tc_row: ElementHandle
    ) -> Optional[ElementHandle]:
        """Find the Play button within a test case row.

        Args:
            tc_row: Test case row element.

        Returns:
            Play button element or None.
        """
        selectors = [
            'button:has-text("Play")',
            '.btn-play',
            '[data-action="play"]',
            '.play-button',
            'button[class*="play"]',
            'button[class*="run"]',
        ]

        for selector in selectors:
            try:
                button = await tc_row.query_selector(selector)
                if button:
                    return button
            except Exception:
                continue

        return None

    async def _wait_for_execution_start(self, timeout: int = 5) -> None:
        """Wait briefly to confirm execution has started.

        Looks for spinner or loading indicators.

        Args:
            timeout: Maximum wait time in seconds.
        """
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            spinner_selectors = [
                '.spinner',
                '.loading',
                '[data-state="running"]',
                '.animate-spin',
                '.fa-spinner',
            ]

            for selector in spinner_selectors:
                try:
                    spinner = await self.page.wait_for_selector(
                        selector, timeout=500
                    )
                    if spinner and await spinner.is_visible():
                        return  # Execution started
                except Exception:
                    continue

            await asyncio.sleep(0.2)

    async def _check_completion_indicators(self) -> bool:
        """Check if the test case has completed execution.

        Returns:
            True if completion indicators are present.
        """
        # Check for status icons
        status_selectors = [
            '.status-passed',
            '.status-failed',
            '[data-status="passed"]',
            '[data-status="failed"]',
            '.fa-check-circle',
            '.fa-times-circle',
            '.green-tick',
            '.red-cross',
        ]

        for selector in status_selectors:
            try:
                element = await self.page.query_selector(selector)
                if element and await element.is_visible():
                    return True
            except Exception:
                continue

        # Check if spinner disappeared (indicates completion)
        spinner_selectors = [
            '.spinner',
            '.loading',
            '[data-state="running"]',
        ]

        any_spinner_visible = False
        for selector in spinner_selectors:
            try:
                spinner = await self.page.query_selector(selector)
                if spinner and await spinner.is_visible():
                    any_spinner_visible = True
                    break
            except Exception:
                continue

        # If no spinners visible and page is stable, assume completion
        if not any_spinner_visible:
            try:
                await self.page.wait_for_load_state("networkidle", timeout=1000)
                return True
            except Exception:
                pass

        return False
