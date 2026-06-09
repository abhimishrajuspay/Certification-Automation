"""Prompt builder for LLM-driven payload generation and self-correction."""

from typing import Dict, List, Optional

from core.models import TestCase


class PromptBuilder:
    """Builds structured prompts for LiteLLM interactions."""

    SYSTEM_PROMPT_TEMPLATE = """You are an expert certification automation engineer specializing in API testing and XML payload generation.

Your task is to generate precise curl commands with correct XML payloads and headers to simulate system responses for CZ (Central Zone) certification test cases.

RULES:
1. Analyze the test case description carefully to understand the expected behavior.
2. Reference the provided XML schemas and sample payloads from the application repository.
3. Generate exact curl commands with correct headers, namespaces, and content types.
4. Include all necessary correlation IDs in the payload.
5. Handle both positive (success) and negative (failure) scenarios correctly.
6. For timeout scenarios, ensure the payload structure matches the expected delayed response.
7. ALWAYS return the curl command in a code block.
8. If you cannot generate the command, explain why.

OUTPUT FORMAT:
Provide the curl command in a fenced code block:
```bash
curl -X POST [URL] \
  -H "Content-Type: application/xml" \
  -H "[other-headers]" \
  -d '[XML_PAYLOAD]'
```
"""

    CORRECTION_SYSTEM_PROMPT = """You are correcting a previously generated curl command that failed during execution.

Review the error details and the original command, then generate a corrected version.

CORRECTION GUIDELINES:
1. Fix the specific error indicated in the failure reason.
2. Ensure XML schema compliance based on the reference XSD.
3. Verify all correlation IDs are correctly placed.
4. Double-check headers and namespaces.
5. If the error indicates a timeout, verify the payload is designed for delayed processing.

OUTPUT FORMAT:
Provide the corrected curl command in a fenced code block:
```bash
curl -X POST [URL] \
  -H "Content-Type: application/xml" \
  -H "[other-headers]" \
  -d '[CORRECTED_XML_PAYLOAD]'
```
"""

    def build_initial_prompt(
        self,
        test_case: TestCase,
        context_str: str,
        local_endpoint: str,
    ) -> tuple[str, str]:
        """Build initial prompt for payload generation.

        Args:
            test_case: Test case to generate payload for.
            context_str: Formatted repository context string.
            local_endpoint: Local endpoint URL for curl command.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE

        user_prompt_parts = [
            f"## Test Case: {test_case.id}",
            f"**Description:** {test_case.description}",
            f"**Expected Status:** {test_case.expected_status}",
        ]

        if test_case.payload_requirements:
            user_prompt_parts.append(
                f"**Payload Requirements:** {test_case.payload_requirements}"
            )

        if test_case.correlation_ids:
            user_prompt_parts.append(
                f"**Correlation IDs:** {', '.join(test_case.correlation_ids)}"
            )

        user_prompt_parts.extend([
            f"",
            f"**Target Endpoint:** {local_endpoint}",
            f"",
            context_str,
            f"",
            f"Please generate the exact curl command to execute this test case.",
            f"Ensure the XML payload is complete and syntactically correct.",
        ])

        user_prompt = "\n".join(user_prompt_parts)
        return system_prompt, user_prompt

    def build_correction_prompt(
        self,
        test_case: TestCase,
        context_str: str,
        previous_command: str,
        error_details: str,
        pod_logs: Optional[str],
    ) -> tuple[str, str]:
        """Build correction prompt for failed execution.

        Args:
            test_case: Test case being retried.
            context_str: Formatted repository context string.
            previous_command: Previously generated curl command.
            error_details: Error message or failure reason.
            pod_logs: Filtered pod logs (optional).

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        system_prompt = self.CORRECTION_SYSTEM_PROMPT

        user_prompt_parts = [
            f"## Test Case: {test_case.id} (Retry)",
            f"**Description:** {test_case.description}",
            f"**Expected Status:** {test_case.expected_status}",
            f"",
            f"**Previous Command:**",
            f"```bash",
            f"{previous_command}",
            f"```",
            f"",
            f"**Error Details:**",
            f"```",
            f"{error_details}",
            f"```",
        ]

        if pod_logs:
            user_prompt_parts.extend([
                f"",
                f"**Pod Logs:**",
                f"```",
                f"{pod_logs}",
                f"```",
            ])

        user_prompt_parts.extend([
            f"",
            context_str,
            f"",
            f"Please generate the corrected curl command.",
            f"Address the specific error and ensure XML validity.",
        ])

        user_prompt = "\n".join(user_prompt_parts)
        return system_prompt, user_prompt

    def build_chained_prompt(
        self,
        current_tc: TestCase,
        previous_results: List[Dict[str, any]],
        context_str: str,
        local_endpoint: str,
    ) -> tuple[str, str]:
        """Build prompt for chained test cases with previous context.

        Args:
            current_tc: Current test case to execute.
            previous_results: Results from previous test cases in the chain.
            context_str: Formatted repository context string.
            local_endpoint: Local endpoint URL.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE

        user_prompt_parts = [
            f"## Chained Test Case: {current_tc.id}",
            f"**Description:** {current_tc.description}",
            f"**Expected Status:** {current_tc.expected_status}",
            f"",
            f"**Previous Test Case Results:**",
        ]

        for prev in previous_results:
            user_prompt_parts.append(f"- {prev['id']}: {prev['status']}")
            if prev.get("extracted_data"):
                user_prompt_parts.append(
                    f"  Extracted Data: {prev['extracted_data']}"
                )

        user_prompt_parts.extend([
            f"",
            f"**Target Endpoint:** {local_endpoint}",
            f"",
            context_str,
            f"",
            f"Generate the curl command considering the chain context.",
            f"Use extracted data from previous test cases where applicable.",
        ])

        user_prompt = "\n".join(user_prompt_parts)
        return system_prompt, user_prompt
