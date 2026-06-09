"""Context loader for repository files to feed into LLM prompts."""

import os
from pathlib import Path
from typing import Dict, List

import tiktoken

from config.settings import Settings
from core.exceptions import ContextLoadingError


class ContextLoader:
    """Loads and manages repository context for LLM prompts."""

    ALLOWED_EXTENSIONS = {".xml", ".json", ".yaml", ".yml", ".xsd", ".txt"}

    def __init__(self, settings: Settings):
        self.settings = settings

    def load_context(self) -> Dict[str, str]:
        """Load repository context files into a dictionary.

        Returns:
            Dictionary mapping file paths to their contents.

        Raises:
            ContextLoadingError: If context loading fails.
        """
        if not self.settings.repo_path:
            return {}

        context_files: Dict[str, str] = {}

        try:
            for subdir in self.settings.allowed_context_paths:
                target_dir = self.settings.repo_path / subdir
                if not target_dir.exists():
                    continue

                for root, _, files in os.walk(target_dir):
                    for filename in files:
                        filepath = Path(root) / filename
                        if filepath.suffix.lower() in self.ALLOWED_EXTENSIONS:
                            try:
                                content = filepath.read_text(encoding="utf-8")
                                relative_path = filepath.relative_to(
                                    self.settings.repo_path
                                )
                                context_files[str(relative_path)] = content
                            except Exception as e:
                                print(f"Warning: Could not read {filepath}: {e}")

            return context_files

        except Exception as e:
            raise ContextLoadingError(f"Failed to load context: {e}")

    def estimate_tokens(self, context: Dict[str, str]) -> int:
        """Estimate token count for context files.

        Args:
            context: Dictionary of file paths to contents.

        Returns:
            Estimated token count.
        """
        encoder = tiktoken.get_encoding("cl100k_base")
        total_tokens = 0

        for content in context.values():
            tokens = encoder.encode(content)
            total_tokens += len(tokens)

        return total_tokens

    def trim_context(
        self,
        context: Dict[str, str],
        max_tokens: int = 80000,
    ) -> Dict[str, str]:
        """Trim context to fit within token budget.

        Prioritizes files by extension importance (.xsd > .xml > .json > others).

        Args:
            context: Full context dictionary.
            max_tokens: Maximum allowed tokens.

        Returns:
            Trimmed context dictionary.
        """
        priority_order = {".xsd": 0, ".xml": 1, ".json": 2, ".yaml": 3, ".yml": 3}

        # Sort files by priority
        sorted_files = sorted(
            context.items(),
            key=lambda item: priority_order.get(
                Path(item[0]).suffix.lower(), 99
            ),
        )

        trimmed: Dict[str, str] = {}
        current_tokens = 0
        encoder = tiktoken.get_encoding("cl100k_base")

        for filepath, content in sorted_files:
            content_tokens = len(encoder.encode(content))

            if current_tokens + content_tokens > max_tokens:
                # Try to include a truncated version
                remaining = max_tokens - current_tokens
                if remaining > 100:
                    truncated = encoder.decode(encoder.encode(content)[:remaining])
                    trimmed[filepath] = truncated
                break

            trimmed[filepath] = content
            current_tokens += content_tokens

        return trimmed

    def format_context_for_prompt(self, context: Dict[str, str]) -> str:
        """Format context dictionary into a prompt-friendly string.

        Args:
            context: Dictionary of file paths to contents.

        Returns:
            Formatted string for LLM prompt.
        """
        parts = []
        parts.append("=== REFERENCE DATA FROM APPLICATION REPOSITORY ===\n")

        for filepath, content in context.items():
            parts.append(f"--- FILE: {filepath} ---")
            parts.append(content)
            parts.append("")

        parts.append("=== END REFERENCE DATA ===\n")

        return "\n".join(parts)
