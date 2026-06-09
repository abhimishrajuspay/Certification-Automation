"""LiteLLM client for LLM-driven payload generation."""

import json
from typing import Any, Dict, List, Optional

from litellm import completion

from config.settings import Settings
from core.exceptions import LLMAgentError


class LiteLLMClient:
    """Wrapper around LiteLLM for generating payloads and commands."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def generate_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> str:
        """Generate a payload or command using LiteLLM.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User query with context.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Maximum tokens in response.

        Returns:
            Generated text response.

        Raises:
            LLMAgentError: If generation fails.
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = completion(
                model=self.settings.litellm_model,
                messages=messages,
                api_base=self.settings.litellm_base_url,
                api_key=self.settings.litellm_api_key.get_secret_value(),
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            raise LLMAgentError(f"LiteLLM generation failed: {e}")

    def generate_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4000,
        max_retries: int = 3,
    ) -> str:
        """Generate with automatic retry on failure.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User query with context.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            max_retries: Maximum number of retry attempts.

        Returns:
            Generated text response.

        Raises:
            LLMAgentError: If all retries fail.
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return self.generate_payload(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff

        raise LLMAgentError(
            f"LiteLLM generation failed after {max_retries + 1} attempts: {last_error}"
        )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> str:
        """Send a chat completion request.

        Args:
            messages: List of message dictionaries with role and content.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            Generated text response.

        Raises:
            LLMAgentError: If completion fails.
        """
        try:
            response = completion(
                model=self.settings.litellm_model,
                messages=messages,
                api_base=self.settings.litellm_base_url,
                api_key=self.settings.litellm_api_key.get_secret_value(),
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            raise LLMAgentError(f"Chat completion failed: {e}")
