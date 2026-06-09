"""CLI entrypoint for CZ Certification Automation."""

import asyncio
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from config.settings import Settings, load_settings
from core.state_machine import StateMachine

console = Console()


def prompt_for_session_cookie() -> str:
    """Prompt user for JSESSIONID cookie."""
    console.print(Panel.fit(
        "[yellow]Session Authentication Required[/yellow]\n"
        "Please enter your JSESSIONID cookie value after completing CAPTCHA and 2FA.",
        title="CZ Portal Login",
    ))
    return click.prompt("JSESSIONID", hide_input=False)


@click.command()
@click.option(
    "--repo-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Path to application repository for LLM context",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to configuration file (overrides .env)",
)
@click.option(
    "--parallel",
    "-p",
    is_flag=True,
    help="Enable parallel test case execution",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=4,
    help="Maximum parallel workers (default: 4)",
)
@click.option(
    "--human-review",
    "-r",
    is_flag=True,
    help="Enable human review guardrail for LLM commands",
)
@click.option(
    "--model",
    "-m",
    type=str,
    help="Override LiteLLM model identifier",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be executed without running",
)
def cli(
    repo_path: Path,
    config: Path,
    parallel: bool,
    workers: int,
    human_review: bool,
    model: str,
    dry_run: bool,
) -> None:
    """CZ Certification Automation Framework.

    Automates execution and validation of certification test cases
    on the CZ (Central Zone) portal.
    """
    console.print(Panel.fit(
        "[bold blue]CZ Certification Automation[/bold blue]\n"
        "Automated test execution and validation framework",
        title="Welcome",
    ))

    # Load settings
    try:
        settings = load_settings()
    except Exception as e:
        console.print(f"[red]Failed to load settings: {e}[/red]")
        sys.exit(1)

    # Apply CLI overrides
    if repo_path:
        settings.repo_path = repo_path
    if parallel:
        settings.parallel = True
    if workers:
        settings.max_parallel_workers = workers
    if human_review:
        settings.human_review = True
    if model:
        settings.litellm_model = model

    # Validate essential settings
    if not settings.cz_base_url:
        console.print("[red]Error: CZ_BASE_URL is required[/red]")
        sys.exit(1)

    # Prompt for session cookie if not provided
    if not settings.jsessionid:
        settings.jsessionid = prompt_for_session_cookie()

    if dry_run:
        console.print("\n[yellow]DRY RUN MODE - Configuration:[/yellow]")
        table = Table(title="Settings")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("CZ Base URL", settings.cz_base_url)
        table.add_row("Repository Path", str(settings.repo_path) if settings.repo_path else "None")
        table.add_row("LiteLLM Model", settings.litellm_model)
        table.add_row("K8s Pod", settings.k8s_pod)
        table.add_row("K8s Namespace", settings.k8s_namespace)
        table.add_row("Parallel", str(settings.parallel))
        table.add_row("Max Workers", str(settings.max_parallel_workers))
        table.add_row("Human Review", str(settings.human_review))
        table.add_row("Rate Limit Delay", f"{settings.rate_limit_delay_ms}ms")

        console.print(table)
        return

    # Run automation
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Starting automation...", total=None)

            async def run_automation():
                sm = StateMachine(settings)

                def on_state_change(state: str):
                    progress.update(task, description=f"State: {state}")

                sm.on_state_change = on_state_change
                result = await sm.run()
                return result

            result = asyncio.run(run_automation())

        # Display results
        console.print("\n[bold green]Automation Complete![/bold green]\n")

        table = Table(title="Results Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")

        total = result.total
        if total > 0:
            table.add_row("Total", str(total), "100.0%")
            table.add_row("Passed", str(result.passed), f"{result.passed/total*100:.1f}%")
            table.add_row("Failed", str(result.failed), f"{result.failed/total*100:.1f}%")
            table.add_row("Skipped", str(result.skipped), f"{result.skipped/total*100:.1f}%")

        console.print(table)

        # Show report location
        report_path = settings.artifacts_dir / "report.html"
        console.print(f"\n[blue]Report saved to:[/blue] {report_path}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Automation interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Automation failed: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    cli()
