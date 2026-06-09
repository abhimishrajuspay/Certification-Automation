"""Web dashboard server for human review and live monitoring."""

import asyncio
import json
from pathlib import Path
from typing import Dict, Optional

from jinja2 import Environment, FileSystemLoader

from config.settings import Settings
from core.enums import ReviewAction
from core.state_machine import StateMachine


class DashboardServer:
    """Simple HTTP server for web-based review panel."""

    def __init__(self, settings: Settings, state_machine: Optional[StateMachine] = None):
        self.settings = settings
        self.state_machine = state_machine
        self._server = None
        self._review_callbacks: Dict[str, asyncio.Future] = {}

    async def start(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        """Start the dashboard server.

        Args:
            host: Host to bind to.
            port: Port to listen on.
        """
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/api/status", self._handle_status)
        app.router.add_post("/api/review/{review_id}", self._handle_review)
        app.router.add_static("/artifacts", str(self.settings.artifacts_dir))

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        print(f"Dashboard server running at http://{host}:{port}")

    async def _handle_index(self, request) -> any:
        """Serve the main dashboard page."""
        from aiohttp import web

        template_dir = Path(__file__).parent.parent / "reporting" / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template("dashboard.html")

        if self.state_machine and self.state_machine.result:
            html = template.render(
                result=self.state_machine.result,
                test_cases=self.state_machine.result.test_cases,
                stats=self._compute_stats(),
                duration=self._format_duration(),
            )
        else:
            html = "<html><body><h1>CZ Automation Dashboard</h1><p>No results yet.</p></body></html>"

        return web.Response(text=html, content_type="text/html")

    async def _handle_status(self, request) -> any:
        """Serve current automation status as JSON."""
        from aiohttp import web

        status = {
            "state": self.state_machine.current_state if self.state_machine else "IDLE",
            "total_tests": self.state_machine.result.total if self.state_machine else 0,
            "completed": self.state_machine.result.total - self.state_machine.result.pending if self.state_machine else 0,
        }

        return web.json_response(status)

    async def _handle_review(self, request) -> any:
        """Handle review submission from web dashboard."""
        from aiohttp import web

        review_id = request.match_info["review_id"]
        data = await request.json()
        action = data.get("action", "reject")

        if review_id in self._review_callbacks:
            self._review_callbacks[review_id].set_result(action)

        return web.json_response({"status": "ok"})

    async def wait_for_review(self, review_id: str, timeout: int = 300) -> Optional[ReviewAction]:
        """Wait for review submission.

        Args:
            review_id: Unique review identifier.
            timeout: Maximum wait time.

        Returns:
            ReviewAction or None on timeout.
        """
        future = asyncio.get_event_loop().create_future()
        self._review_callbacks[review_id] = future

        try:
            action_str = await asyncio.wait_for(future, timeout=timeout)
            return ReviewAction(action_str)
        except asyncio.TimeoutError:
            return None
        finally:
            self._review_callbacks.pop(review_id, None)

    def _compute_stats(self) -> dict:
        """Compute statistics for dashboard."""
        if not self.state_machine or not self.state_machine.result:
            return {}

        result = self.state_machine.result
        total = result.total

        return {
            "total": total,
            "passed": result.passed,
            "failed": result.failed,
            "skipped": result.skipped,
            "pass_rate": (result.passed / total * 100) if total > 0 else 0,
            "fail_rate": (result.failed / total * 100) if total > 0 else 0,
        }

    def _format_duration(self) -> str:
        """Format duration for dashboard."""
        if not self.state_machine or not self.state_machine.result:
            return "N/A"

        seconds = self.state_machine.result.duration_seconds
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
