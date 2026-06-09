"""State machine for orchestrating the CZ certification automation lifecycle."""

import asyncio
from datetime import datetime
from typing import Callable, Dict, List, Optional

from config.settings import Settings
from core.dag_engine import TestSuiteDAG
from core.enums import ExecutionMode, FailureCategory, TestStatus
from core.exceptions import StateMachineError
from core.k8s_bridge import LogFetcher, PortForwardManager
from core.models import TestCase, TestSuiteResult
from llm_agent.client import LiteLLMClient
from llm_agent.context_loader import ContextLoader
from llm_agent.executor import CommandExecutor
from llm_agent.prompt_builder import PromptBuilder
from scraper.orchestrator import Scraper
from validator import Validator


class StateMachine:
    """Orchestrates the complete certification automation lifecycle."""

    STATES = [
        "INIT",
        "SCRAPING",
        "PORT_FORWARD_SETUP",
        "EXECUTING",
        "VALIDATING",
        "REPORTING",
        "DONE",
    ]

    def __init__(self, settings: Settings):
        self.settings = settings
        self.current_state = "INIT"
        self.scraper: Optional[Scraper] = None
        self.port_forward: Optional[PortForwardManager] = None
        self.log_fetcher: Optional[LogFetcher] = None
        self.validator: Optional[Validator] = None
        self.llm_client: Optional[LiteLLMClient] = None
        self.context_loader: Optional[ContextLoader] = None
        self.prompt_builder: Optional[PromptBuilder] = None
        self.executor: Optional[CommandExecutor] = None
        self.result = TestSuiteResult()
        self.dag: Optional[TestSuiteDAG] = None
        self.failed_nodes: set = set()
        self.on_state_change: Optional[Callable[[str], None]] = None

    async def run(self) -> TestSuiteResult:
        """Execute the complete automation lifecycle.

        Returns:
            TestSuiteResult with all test case outcomes.
        """
        try:
            # Phase 0: Initialize
            await self._transition("INIT")
            self._initialize_components()

            # Phase 1: Scrape test cases and build DAG
            await self._transition("SCRAPING")
            self.dag = await self.scraper.scrape_all_test_cases()
            self.result.test_cases = [
                self.dag.get_test_case(tc_id)
                for tc_id in self.dag.get_topological_order()
            ]
            self.result.total = len(self.result.test_cases)

            # Phase 2: Setup port-forward
            await self._transition("PORT_FORWARD_SETUP")
            self.port_forward.start()
            await asyncio.sleep(2)  # Wait for tunnel

            # Phase 3: Execute test cases
            await self._transition("EXECUTING")
            if self.settings.parallel:
                await self._execute_parallel()
            else:
                await self._execute_sequential()

            # Phase 4: Final validation pass
            await self._transition("VALIDATING")
            await self._final_validation()

            # Phase 5: Generate report
            await self._transition("REPORTING")
            await self._generate_report()

            await self._transition("DONE")
            return self.result

        except Exception as e:
            raise StateMachineError(f"Automation failed in state {self.current_state}: {e}")

        finally:
            await self._cleanup()

    def _initialize_components(self) -> None:
        """Initialize all automation components."""
        self.settings.ensure_directories()
        self.scraper = Scraper(self.settings)
        self.port_forward = PortForwardManager(self.settings)
        self.log_fetcher = LogFetcher(self.settings)
        self.llm_client = LiteLLMClient(self.settings)
        self.context_loader = ContextLoader(self.settings)
        self.prompt_builder = PromptBuilder()
        self.executor = CommandExecutor(timeout=30)

    async def _execute_sequential(self) -> None:
        """Execute test cases sequentially in topological order."""
        for tc_id in self.dag.get_topological_order():
            tc = self.dag.get_test_case(tc_id)

            # Skip if blocked by failed dependency
            if self.dag.is_blocked(tc_id, self.failed_nodes):
                tc.status = TestStatus.SKIPPED
                continue

            await self._execute_single_test_case(tc)

            # Rate limiting
            if self.settings.rate_limit_delay_ms > 0:
                await asyncio.sleep(self.settings.rate_limit_delay_ms / 1000)

    async def _execute_parallel(self) -> None:
        """Execute test cases in parallel waves respecting DAG."""
        waves = self.dag.get_execution_waves()

        for wave in waves:
            tasks = []
            for tc_id in wave:
                tc = self.dag.get_test_case(tc_id)
                if not self.dag.is_blocked(tc_id, self.failed_nodes):
                    tasks.append(self._execute_single_test_case(tc))

            # Execute wave with semaphore limiting
            semaphore = asyncio.Semaphore(self.settings.max_parallel_workers)

            async def bounded_execute(tc: TestCase) -> None:
                async with semaphore:
                    await self._execute_single_test_case(tc)

            await asyncio.gather(*[bounded_execute(tc) for tc in tasks])

    async def _execute_single_test_case(self, tc: TestCase) -> None:
        """Execute a single test case with LLM-driven payload generation."""
        from core.exceptions import LLMAgentError

        tc.status = TestStatus.RUNNING
        tc.started_at = datetime.now()

        try:
            # Initialize validator lazily
            if not self.validator:
                await self.scraper.initialize()
                self.validator = Validator(self.scraper.browser_manager.page, self.settings)

            # Health check port-forward before execution
            await self.port_forward.restart_if_stale()

            # Trigger test case on portal
            await self.scraper.execute_test_case(tc)

            # If outbound/chained API call is needed, generate and execute
            if tc.payload_requirements:
                await self._generate_and_execute_payload(tc)

            # Validate results
            validated_tc = await self.validator.validate_test_case(tc)
            tc.status = validated_tc.status
            tc.pod_logs = validated_tc.pod_logs
            tc.failure_category = validated_tc.failure_category

            # Handle failure with LLM self-correction
            if tc.status == TestStatus.FAILED:
                success = await self._attempt_self_correction(tc)
                if not success:
                    self.failed_nodes.add(tc.id)
                    self.dag.skip_subtree(tc.id)

        except Exception as e:
            tc.status = TestStatus.FAILED
            tc.execution_error = str(e)
            self.failed_nodes.add(tc.id)
            self.dag.skip_subtree(tc.id)

        finally:
            tc.completed_at = datetime.now()

    async def _generate_and_execute_payload(self, tc: TestCase) -> None:
        """Generate and execute curl command via LLM."""
        context = self.context_loader.load_context()
        context_str = self.context_loader.format_context_for_prompt(context)

        local_endpoint = f"http://localhost:{self.settings.k8s_local_port}"

        system_prompt, user_prompt = self.prompt_builder.build_initial_prompt(
            tc, context_str, local_endpoint
        )

        # Human review if enabled
        if self.settings.human_review:
            approved = await self._human_review_gate(tc, None)
            if not approved:
                tc.status = TestStatus.SKIPPED
                return

        # Generate curl command
        curl_command = self.llm_client.generate_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_retries=1,
        )

        tc.generated_curl_command = curl_command

        # Extract curl from markdown code block if present
        curl_command = self._extract_curl_from_markdown(curl_command)

        # Execute
        result = self.executor.execute_curl(curl_command)
        tc.execution_output = result.stdout
        tc.execution_error = result.stderr if not result.success else None

    async def _attempt_self_correction(self, tc: TestCase) -> bool:
        """Attempt LLM self-correction for failed test cases.

        Returns:
            True if correction succeeded.
        """
        if not tc.generated_curl_command:
            return False

        for attempt in range(self.settings.max_llm_retries):
            context = self.context_loader.load_context()
            context_str = self.context_loader.format_context_for_prompt(context)

            system_prompt, user_prompt = self.prompt_builder.build_correction_prompt(
                tc,
                context_str,
                tc.generated_curl_command,
                tc.execution_error or "Unknown error",
                tc.pod_logs,
            )

            try:
                corrected_command = self.llm_client.generate_payload(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )

                corrected_command = self._extract_curl_from_markdown(corrected_command)

                # Human review if enabled
                if self.settings.human_review:
                    approved = await self._human_review_gate(tc, corrected_command)
                    if not approved:
                        continue

                result = self.executor.execute_curl(corrected_command)

                # Record attempt
                tc.llm_attempts.append({
                    "command": corrected_command,
                    "output": result.stdout,
                    "error": result.stderr if not result.success else None,
                })

                if result.success:
                    tc.generated_curl_command = corrected_command
                    tc.execution_output = result.stdout
                    tc.execution_error = None

                    # Re-validate
                    validated_tc = await self.validator.validate_test_case(tc)
                    if validated_tc.status == TestStatus.PASSED:
                        tc.status = TestStatus.PASSED
                        return True

            except Exception as e:
                tc.llm_attempts.append({
                    "command": None,
                    "error": str(e),
                })

        return False

    def _extract_curl_from_markdown(self, text: str) -> str:
        """Extract curl command from markdown code block."""
        import re

        # Look for bash or plain code blocks containing curl
        patterns = [
            r"```bash\n(.*?)\n```",
            r"```\n(.*?)\n```",
            r"```shell\n(.*?)\n```",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(1).strip()
                if content.startswith("curl"):
                    return content

        # If no code block, return as-is if it starts with curl
        stripped = text.strip()
        if stripped.startswith("curl"):
            return stripped

        return text

    async def _human_review_gate(self, tc: TestCase, command: Optional[str]) -> bool:
        """Gate for human review of generated commands.

        Args:
            tc: Test case being reviewed.
            command: Command to review (None for initial generation).

        Returns:
            True if approved.
        """
        # TODO: Implement web dashboard approval panel and CLI fallback
        # For now, auto-approve if human review is disabled
        return True

    async def _final_validation(self) -> None:
        """Perform final validation pass on all test cases."""
        self.result.update_counts()
        self.result.completed_at = datetime.now()

    async def _generate_report(self) -> None:
        """Generate HTML dashboard report."""
        from reporting.dashboard import DashboardGenerator

        generator = DashboardGenerator(self.settings)
        output_path = self.settings.artifacts_dir / "report.html"
        generator.generate(self.result, output_path)
        print(f"Report generated: {output_path}")

    async def _cleanup(self) -> None:
        """Clean up all resources."""
        if self.scraper:
            await self.scraper.shutdown()
        if self.port_forward:
            self.port_forward.stop()

    async def _transition(self, new_state: str) -> None:
        """Transition to a new state.

        Args:
            new_state: Target state name.
        """
        print(f"State transition: {self.current_state} -> {new_state}")
        self.current_state = new_state

        if self.on_state_change:
            self.on_state_change(new_state)
