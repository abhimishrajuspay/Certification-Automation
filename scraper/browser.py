"""Browser management for Playwright scraper."""

import asyncio
from pathlib import Path
from typing import Optional

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from config.settings import Settings
from core.exceptions import ScraperError


class BrowserManager:
    """Manages Playwright browser lifecycle and session injection."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._playwright = None

    async def start(self) -> Page:
        """Launch browser and inject authenticated session.

        Returns:
            Playwright Page object ready for navigation.

        Raises:
            ScraperError: If browser launch fails.
        """
        try:
            self._playwright = await async_playwright().start()
            self.browser = await self._playwright.chromium.launch(
                headless=False,  # Show browser for visual monitoring
                args=["--disable-blink-features=AutomationControlled"],
            )

            # Prepare context with session cookie
            cookies = []
            if self.settings.jsessionid:
                cookies.append({
                    "name": "JSESSIONID",
                    "value": self.settings.jsessionid,
                    "domain": self._extract_domain(self.settings.cz_base_url),
                    "path": "/",
                })

            self.context = await self.browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                cookies=cookies,
            )

            # Add anti-detection scripts
            await self.context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)

            self.page = await self.context.new_page()
            return self.page

        except Exception as e:
            raise ScraperError(f"Failed to launch browser: {e}")

    async def stop(self) -> None:
        """Clean up browser resources."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def take_screenshot(self, name: str, path: Path) -> Path:
        """Take a screenshot of the current page.

        Args:
            name: Screenshot identifier.
            path: Full path to save the screenshot.

        Returns:
            Path to the saved screenshot.
        """
        if not self.page:
            raise ScraperError("Browser not started")

        await self.page.screenshot(path=path, full_page=True)
        return path

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for cookie setting."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc
