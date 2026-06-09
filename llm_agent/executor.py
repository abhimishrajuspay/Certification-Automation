"""Command executor for running LLM-generated curl commands."""

import subprocess
from dataclasses import dataclass
from typing import Optional

from core.exceptions import ExecutionError


@dataclass
class ExecutionResult:
    """Result of executing a command."""

    success: bool
    stdout: str
    stderr: str
    return_code: int
    command: str

    @property
    def combined_output(self) -> str:
        """Combine stdout and stderr."""
        parts = []
        if self.stdout:
            parts.append(f"STDOUT:\n{self.stdout}")
        if self.stderr:
            parts.append(f"STDERR:\n{self.stderr}")
        return "\n\n".join(parts) if parts else ""


class CommandExecutor:
    """Executes shell commands safely."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def execute(self, command: str) -> ExecutionResult:
        """Execute a shell command.

        Args:
            command: Command string to execute.

        Returns:
            ExecutionResult with output and status.

        Raises:
            ExecutionError: If execution fails catastrophically.
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                command=command,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Command timed out after {self.timeout} seconds",
                return_code=-1,
                command=command,
            )
        except Exception as e:
            raise ExecutionError(f"Command execution failed: {e}")

    def execute_curl(self, curl_command: str) -> ExecutionResult:
        """Execute a curl command with validation.

        Args:
            curl_command: Curl command string.

        Returns:
            ExecutionResult.
        """
        # Basic validation that it's actually a curl command
        if not curl_command.strip().startswith("curl"):
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="Invalid command: Expected curl command",
                return_code=-1,
                command=curl_command,
            )

        return self.execute(curl_command)
