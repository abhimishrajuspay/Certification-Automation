"""Human review gate with web dashboard and CLI fallback."""

import asyncio
import sys
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from core.enums import ReviewAction
from core.models import TestCase

console = Console()


class HumanReviewGate:
    """Manages human review of LLM-generated commands."""

    def __init__(self, enable_dashboard: bool = True):
        self.enable_dashboard = enable_dashboard
        self._pending_reviews: dict = {}

    async def review(
        self,
        tc: TestCase,
        command: Optional[str],
        previous_error: Optional[str] = None,
    ) -> ReviewAction:
        """Present generated command for human review.

        Tries web dashboard first, falls back to CLI prompt.

        Args:
            tc: Test case being reviewed.
            command: Generated curl command.
            previous_error: Previous error if this is a retry.

        Returns:
            ReviewAction enum value.
        """
        # Try web dashboard first
        if self.enable_dashboard:
            action = await self._web_review(tc, command, previous_error)
            if action is not None:
                return action

        # Fallback to CLI
        return await self._cli_review(tc, command, previous_error)

    async def _web_review(
        self,
        tc: TestCase,
        command: Optional[str],
        previous_error: Optional[str] = None,
    ) -> Optional[ReviewAction]:
        """Present review via web dashboard.

        Returns:
            ReviewAction if review completed, None if dashboard unavailable.
        """
        # TODO: Implement web dashboard review panel
        # For now, return None to fall back to CLI
        return None

    async def _cli_review(
        self,
        tc: TestCase,
        command: Optional[str],
        previous_error: Optional[str] = None,
    ) -> ReviewAction:
        """Present review via CLI prompt.

        Args:
            tc: Test case being reviewed.
            command: Generated curl command.
            previous_error: Previous error if this is a retry.

        Returns:
            ReviewAction enum value.
        """
        console.print(Panel.fit(
            f"[bold cyan]Human Review Required[/bold cyan]\n\n"
            f"[yellow]Test Case:[/yellow] {tc.id}\n"
            f"[yellow]Description:[/yellow] {tc.description}\n"
            f"[yellow]Expected Status:[/yellow] {tc.expected_status}",
            title="Review Gate",
        ))

        if previous_error:
            console.print(f"\n[red]Previous Error:[/red]\n{previous_error}\n")

        if command:
            console.print("[green]Generated Command:[/green]")
            console.print(command)
            console.print()

        # Prompt for action
        while True:
            response = Prompt.ask(
                "[bold]Action[/bold]",
                choices=["a", "approve", "r", "reject", "e", "edit", "s", "skip"],
                default="a",
            ).lower()

            if response in ("a", "approve", ""):
                return ReviewAction.APPROVE
            elif response in ("r", "reject"):
                return ReviewAction.REJECT
            elif response in ("e", "edit"):
                edited = Prompt.ask("Enter edited command", default=command or "")
                if edited and edited != command:
                    tc.generated_curl_command = edited
                return ReviewAction.EDIT
            elif response in ("s", "skip"):
                return ReviewAction.REJECT  # Treat skip as reject

    async def wait_for_web_approval(self, review_id: str, timeout: int = 300) -> Optional[ReviewAction]:
        """Wait for web dashboard approval.

        Args:
            review_id: Unique review identifier.
            timeout: Maximum wait time in seconds.

        Returns:
            ReviewAction or None if timeout.
        """
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            if review_id in self._pending_reviews:
                action = self._pending_reviews.pop(review_id)
                return ReviewAction(action)
            await asyncio.sleep(1)

        return None

    def submit_web_review(self, review_id: str, action: str) -> None:
        """Submit review action from web dashboard.

        Args:
            review_id: Review identifier.
            action: Action string (approve, reject, edit).
        """
        self._pending_reviews[review_id] = action
