import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from oap.envelope import TaskEnvelope
from oap.router import OAPRouter, RoutingError
from oap.adapters.mock import MockAgentAdapter
from oap.adapters.http import HTTPAdapter

app = typer.Typer(
    name="oap",
    help="Open Agent Protocol — route tasks between agents.",
    add_completion=False,
)
console = Console()


def _build_demo_router() -> OAPRouter:
    """A router pre-loaded with mock agents for local testing."""
    router = OAPRouter()
    router.register(
        "research-agent",
        MockAgentAdapter("research-agent", response="Found 5 relevant sources."),
        capabilities=["research", "find", "search", "summarise"],
    )
    router.register(
        "coding-agent",
        MockAgentAdapter("coding-agent", response="Code written and tested."),
        capabilities=["code", "implement", "debug", "refactor"],
    )
    return router


@app.command()
def init(
    goal: str = typer.Argument(..., help="The task goal for this envelope"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save envelope to file"),
):
    """Create a new TaskEnvelope and print it."""
    envelope = TaskEnvelope(goal=goal)
    json_str = envelope.model_dump_json(indent=2)

    console.print(Syntax(json_str, "json", theme="monokai"))

    if output:
        output.write_text(json_str)
        console.print(f"\n[green]Saved to {output}[/green]")


@app.command()
def inspect(
    file: Path = typer.Argument(..., help="Path to a TaskEnvelope JSON file"),
):
    """Inspect a TaskEnvelope file and summarise its contents."""
    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    try:
        envelope = TaskEnvelope.model_validate_json(file.read_text())
    except Exception as e:
        console.print(f"[red]Invalid envelope: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Envelope[/bold] [dim]{envelope.id}[/dim]")
    console.print(f"[bold]Goal:[/bold] {envelope.goal}")
    console.print(f"[bold]Version:[/bold] {envelope.version}")

    table = Table(show_header=True, header_style="bold dim")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Steps taken", str(len(envelope.steps_taken)))
    table.add_row("Memory keys", ", ".join(envelope.memory.keys()) or "—")
    table.add_row("Max cost", str(envelope.constraints.max_cost_usd or "—"))
    table.add_row("Allowed tools", ", ".join(envelope.constraints.allowed_tools or []) or "—")
    table.add_row("Handoff", envelope.handoff.next_agent if envelope.handoff else "—")
    console.print(table)

    if envelope.steps_taken:
        console.print("\n[bold]Steps:[/bold]")
        for i, step in enumerate(envelope.steps_taken, 1):
            console.print(f"  {i}. [cyan]{step.agent_id}[/cyan] → {step.action}")


@app.command()
def validate(
    file: Path = typer.Argument(..., help="Path to a TaskEnvelope JSON file"),
):
    """Validate a TaskEnvelope file against the schema."""
    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    try:
        TaskEnvelope.model_validate_json(file.read_text())
        console.print("[green]Valid envelope.[/green]")
    except Exception as e:
        console.print(f"[red]Invalid:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def route(
    file: Path = typer.Argument(..., help="Path to a TaskEnvelope JSON file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save result envelope to file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show which agent would handle it, without invoking"),
):
    """Route a TaskEnvelope to the appropriate agent."""
    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    try:
        envelope = TaskEnvelope.model_validate_json(file.read_text())
    except Exception as e:
        console.print(f"[red]Invalid envelope: {e}[/red]")
        raise typer.Exit(1)

    router = _build_demo_router()

    try:
        agent_id = router.select_agent(envelope)
    except RoutingError as e:
        console.print(f"[red]Routing failed:[/red] {e}")
        raise typer.Exit(1)

    if dry_run:
        console.print(f"[yellow]Dry run:[/yellow] would route to [cyan]{agent_id}[/cyan]")
        return

    console.print(f"[dim]Routing to[/dim] [cyan]{agent_id}[/cyan]...")
    result = asyncio.run(router.route(envelope))

    json_str = result.model_dump_json(indent=2)
    console.print(Syntax(json_str, "json", theme="monokai"))

    if output:
        output.write_text(json_str)
        console.print(f"\n[green]Saved to {output}[/green]")

@app.command()
def register(
    agent_id: str = typer.Argument(..., help="Unique name for this agent"),
    url: str = typer.Argument(..., help="Base URL of the agent's HTTP server"),
    capabilities: str = typer.Option(..., "--capabilities", "-c", help="Comma-separated capability keywords"),
    file: Path = typer.Argument(..., help="Envelope file to route"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save result envelope to file"),
):
    """Route an envelope to a specific HTTP agent by URL."""
    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    try:
        envelope = TaskEnvelope.model_validate_json(file.read_text())
    except Exception as e:
        console.print(f"[red]Invalid envelope: {e}[/red]")
        raise typer.Exit(1)

    caps = [c.strip() for c in capabilities.split(",")]
    router = OAPRouter()
    router.register(agent_id, HTTPAdapter(agent_id=agent_id, base_url=url), caps)

    console.print(f"[dim]Routing to[/dim] [cyan]{agent_id}[/cyan] at [dim]{url}[/dim]...")
    result = asyncio.run(router.route(envelope))

    json_str = result.model_dump_json(indent=2)
    console.print(Syntax(json_str, "json", theme="monokai"))

    if output:
        output.write_text(json_str)
        console.print(f"\n[green]Saved to {output}[/green]")
        
@app.command()
def agents():
    """List all registered agents and their capabilities."""
    router = _build_demo_router()

    table = Table(show_header=True, header_style="bold dim")
    table.add_column("Agent ID")
    table.add_column("Capabilities")

    for entry in router.list_agents():
        table.add_row(
            f"[cyan]{entry['id']}[/cyan]",
            ", ".join(entry["capabilities"]),
        )

    console.print(table)