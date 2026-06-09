"""Pydantic settings configuration for CZ Certification Automation."""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # CZ Portal Configuration
    cz_base_url: str = Field(
        default="",
        description="Base URL of the CZ certification portal",
    )
    jsessionid: Optional[str] = Field(
        default=None,
        description="JSESSIONID cookie value for authenticated session",
    )

    # Repository Context
    repo_path: Optional[Path] = Field(
        default=None,
        description="Absolute path to the application repository",
    )
    allowed_context_paths: List[str] = Field(
        default_factory=lambda: ["schemas/", "fixtures/", "templates/"],
        description="Subdirectories within repo_path to scan for LLM context",
    )

    # LiteLLM Configuration
    litellm_base_url: str = Field(
        default="http://localhost:4000",
        description="LiteLLM API base URL",
    )
    litellm_model: str = Field(
        default="gpt-4o",
        description="Model identifier for LiteLLM",
    )
    litellm_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for LiteLLM",
    )

    # Kubernetes Configuration
    k8s_pod: str = Field(
        default="my-pod",
        description="Kubernetes pod name for port-forwarding",
    )
    k8s_namespace: str = Field(
        default="my-namespace",
        description="Kubernetes namespace",
    )
    k8s_local_port: int = Field(
        default=8080,
        description="Local port for kubectl port-forward",
    )
    k8s_remote_port: int = Field(
        default=8080,
        description="Remote port on the Kubernetes pod",
    )
    k8s_health_check_endpoint: str = Field(
        default="http://localhost:8080/health",
        description="Local health check endpoint to verify port-forward",
    )

    # Execution Configuration
    rate_limit_delay_ms: int = Field(
        default=1000,
        ge=0,
        description="Delay in milliseconds between actions",
    )
    max_llm_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retries for LLM self-correction",
    )
    max_tc_execution_timeout_sec: int = Field(
        default=60,
        ge=1,
        description="Maximum timeout for single test case execution",
    )
    parallel: bool = Field(
        default=False,
        description="Enable parallel test case execution",
    )
    max_parallel_workers: int = Field(
        default=4,
        ge=1,
        description="Maximum parallel workers when parallel mode is enabled",
    )
    human_review: bool = Field(
        default=False,
        description="Enable human review guardrail for LLM-generated commands",
    )

    # Output Configuration
    artifacts_dir: Path = Field(
        default=Path("./artifacts"),
        description="Base directory for artifacts",
    )
    screenshots_dir: Path = Field(
        default=Path("./artifacts/screenshots"),
        description="Directory for screenshots",
    )
    logs_dir: Path = Field(
        default=Path("./artifacts/logs"),
        description="Directory for log files",
    )

    @field_validator("cz_base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("cz_base_url must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("repo_path")
    @classmethod
    def validate_repo_path(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None and not v.exists():
            raise ValueError(f"repo_path does not exist: {v}")
        return v

    @field_validator("allowed_context_paths", mode="before")
    @classmethod
    def parse_allowed_context_paths(cls, v):
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v

    def ensure_directories(self) -> None:
        """Create artifact directories if they don't exist."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Load settings with optional CLI overrides."""
    return Settings()
