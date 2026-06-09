"""Kubernetes bridge for port-forwarding and log retrieval."""

import asyncio
import subprocess
import time
from typing import List, Optional

import httpx

from config.settings import Settings
from core.exceptions import K8SError, PortForwardError


class PortForwardManager:
    """Manages kubectl port-forward lifecycle with health checking."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._process: Optional[subprocess.Popen] = None
        self._is_active = False

    def start(self) -> None:
        """Start kubectl port-forward subprocess.

        Raises:
            PortForwardError: If port-forward fails to start.
        """
        try:
            # Kill any existing process on the same port
            self._cleanup_existing()

            cmd = [
                "kubectl",
                "port-forward",
                f"pod/{self.settings.k8s_pod}",
                f"{self.settings.k8s_local_port}:{self.settings.k8s_remote_port}",
                "-n",
                self.settings.k8s_namespace,
            ]

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait a moment for tunnel to establish
            time.sleep(2)

            # Verify process is running
            if self._process.poll() is not None:
                stderr = self._process.stderr.read() if self._process.stderr else ""
                raise PortForwardError(f"Port-forward process exited early: {stderr}")

            self._is_active = True

        except FileNotFoundError:
            raise PortForwardError(
                "kubectl not found. Please ensure kubectl is installed and in PATH."
            )
        except Exception as e:
            raise PortForwardError(f"Failed to start port-forward: {e}")

    def stop(self) -> None:
        """Stop the port-forward subprocess."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        self._is_active = False

    async def health_check(self) -> bool:
        """Check if the port-forward tunnel is healthy.

        Returns:
            True if the tunnel is active and responding.
        """
        if not self._is_active or not self._process:
            return False

        # Check if process is still running
        if self._process.poll() is not None:
            self._is_active = False
            return False

        # Try to reach the health endpoint
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self.settings.k8s_health_check_endpoint)
                return response.status_code < 500
        except Exception:
            return False

    async def restart_if_stale(self) -> None:
        """Restart the port-forward if it's not healthy."""
        if not await self.health_check():
            print("Port-forward tunnel stale, restarting...")
            self.stop()
            await asyncio.sleep(1)
            self.start()
            await asyncio.sleep(2)  # Wait for re-establishment

    def _cleanup_existing(self) -> None:
        """Kill any existing kubectl port-forward processes on the same port."""
        try:
            # Find and kill existing port-forward processes
            result = subprocess.run(
                ["pgrep", "-f", f"port-forward.*{self.settings.k8s_local_port}"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for pid in result.stdout.strip().split("\n"):
                    if pid:
                        subprocess.run(["kill", "-9", pid], capture_output=True)
        except Exception:
            pass  # Best effort cleanup


class LogFetcher:
    """Fetches and filters pod logs for correlation IDs."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def fetch_logs(
        self,
        correlation_ids: List[str],
        tail_lines: int = 500,
        since_minutes: int = 5,
    ) -> str:
        """Fetch pod logs filtered by correlation IDs.

        Args:
            correlation_ids: List of correlation IDs to filter by.
            tail_lines: Number of recent log lines to fetch.
            since_minutes: Time window in minutes for log retrieval.

        Returns:
            Filtered log content.

        Raises:
            K8SError: If kubectl logs command fails.
        """
        try:
            # Build kubectl logs command
            cmd = [
                "kubectl",
                "logs",
                f"pod/{self.settings.k8s_pod}",
                "-n",
                self.settings.k8s_namespace,
                f"--tail={tail_lines}",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                raise K8SError(f"kubectl logs failed: {result.stderr}")

            # Filter logs by correlation IDs
            logs = result.stdout
            if correlation_ids:
                filtered_lines = []
                for line in logs.split("\n"):
                    if any(cid in line for cid in correlation_ids):
                        filtered_lines.append(line)
                logs = "\n".join(filtered_lines)

            return logs

        except subprocess.TimeoutExpired:
            raise K8SError("kubectl logs command timed out")
        except FileNotFoundError:
            raise K8SError("kubectl not found. Please ensure kubectl is installed.")
        except Exception as e:
            raise K8SError(f"Failed to fetch logs: {e}")

    def stream_logs(
        self,
        correlation_ids: List[str],
        follow: bool = False,
    ) -> subprocess.Popen:
        """Start streaming pod logs.

        Args:
            correlation_ids: List of correlation IDs to filter by.
            follow: Whether to follow log output (tail -f behavior).

        Returns:
            Subprocess.Popen object for reading log stream.
        """
        cmd = [
            "kubectl",
            "logs",
            f"pod/{self.settings.k8s_pod}",
            "-n",
            self.settings.k8s_namespace,
        ]

        if follow:
            cmd.append("-f")

        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
