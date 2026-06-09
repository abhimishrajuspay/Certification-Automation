"""Navigation and pagination handling for CZ portal."""

import asyncio
from typing import List, Optional

from playwright.async_api import Page

from config.settings import Settings
from core.exceptions import NavigationError


class Navigator:
    """Handles page navigation and pagination on the CZ portal."""

    def __init__(self, page: Page, settings: Settings):
        self.page = page
        self.settings = settings

    async def navigate_to_test_suite(self) -> None:
        """Navigate to the test suite page.

        Raises:
            NavigationError: If navigation fails.
        """
        try:
            await self.page.goto(
                self.settings.cz_base_url,
                wait_until="networkidle",
                timeout=30000,
            )
            # Additional wait for any client-side hydration
            await asyncio.sleep(1)
        except Exception as e:
            raise NavigationError(f"Failed to navigate to test suite: {e}")

    async def get_total_pages(self) -> int:
        """Extract total number of pages from pagination controls.

        Returns:
            Total page count (defaults to 1 if not found).
        """
        try:
            # Look for common pagination patterns
            pagination_selectors = [
                '[data-testid="pagination"]',
                '.pagination',
                '.page-nav',
                '//span[contains(text(), "of")]',  # e.g., "Page 1 of 10"
            ]

            for selector in pagination_selectors:
                try:
                    if selector.startswith("//"):
                        element = await self.page.wait_for_selector(
                            f"xpath={selector}", timeout=2000
                        )
                    else:
                        element = await self.page.wait_for_selector(
                            selector, timeout=2000
                        )
                    if element:
                        text = await element.inner_text()
                        # Try to extract number after "of"
                        import re
                        match = re.search(r"of\s+(\d+)", text)
                        if match:
                            return int(match.group(1))
                except Exception:
                    continue

            return 1
        except Exception:
            return 1

    async def navigate_to_page(self, page_number: int) -> bool:
        """Navigate to a specific page number.

        Args:
            page_number: Target page number (1-indexed).

        Returns:
            True if navigation succeeded.

        Raises:
            NavigationError: If navigation fails.
        """
        try:
            # Strategy 1: Look for page number links/buttons
            page_link_selectors = [
                f'a[href*="page={page_number}"]',
                f'button:has-text("{page_number}")',
                f'[data-page="{page_number}"]',
                f'.page-item:has-text("{page_number}")',
            ]

            for selector in page_link_selectors:
                try:
                    link = await self.page.wait_for_selector(
                        selector, timeout=2000
                    )
                    if link:
                        await link.click()
                        await self.page.wait_for_load_state("networkidle")
                        await asyncio.sleep(0.5)
                        return True
                except Exception:
                    continue

            # Strategy 2: Modify URL directly
            current_url = self.page.url
            if "page=" in current_url:
                new_url = current_url.replace(
                    f"page={self._get_current_page()}",
                    f"page={page_number}",
                )
            else:
                separator = "&" if "?" in current_url else "?"
                new_url = f"{current_url}{separator}page={page_number}"

            await self.page.goto(new_url, wait_until="networkidle")
            await asyncio.sleep(0.5)
            return True

        except Exception as e:
            raise NavigationError(f"Failed to navigate to page {page_number}: {e}")

    def _get_current_page(self) -> int:
        """Extract current page number from URL."""
        import re
        from urllib.parse import urlparse, parse_qs

        parsed = urlparse(self.page.url)
        params = parse_qs(parsed.query)
        if "page" in params:
            return int(params["page"][0])
        return 1
