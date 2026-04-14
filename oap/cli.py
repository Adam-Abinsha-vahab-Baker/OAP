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
from oap import registry

app = typer.Typer(
    name="oap",
    help="Open Agent Protocol — route tasks between agents.",
    add_completion=False,
)
console = Console()


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
def register(
    agent_id: str = typer.Argument(..., help="Unique name for this agent"),
    url: str = typer.Argument(..., help="Base URL of the agent's HTTP server"),
    capabilities: str = typer.Option(..., "--capabilities", "-c", help="Comma-separated capability keywords"),
):
    """Register an HTTP agent in the local registry (~/.oap/agents.json)."""
    caps = [c.strip() for c in capabilities.split(",")]
    registry.add(agent_id, url, caps)
    console.print(f"[green]Registered[/green] [cyan]{agent_id}[/cyan] → {url}")
    console.print(f"[dim]Capabilities:[/dim] {', '.join(caps)}")


@app.command()
def unregister(
    agent_id: str = typer.Argument(..., help="Agent ID to remove"),
):
    """Remove an agent from the local registry."""
    if registry.remove(agent_id):
        console.print(f"[green]Removed[/green] [cyan]{agent_id}[/cyan]")
    else:
        console.print(f"[red]Agent not found:[/red] {agent_id}")
        raise typer.Exit(1)


@app.command()
def route(
    file: Path = typer.Argument(..., help="Path to a TaskEnvelope JSON file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save result envelope to file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show which agent would handle it, without invoking"),
):
    """Route a TaskEnvelope to the best matching agent in the registry."""
    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    try:
        envelope = TaskEnvelope.model_validate_json(file.read_text())
    except Exception as e:
        console.print(f"[red]Invalid envelope: {e}[/red]")
        raise typer.Exit(1)

    router = registry.load_router()

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
def chain(
    file: Path = typer.Argument(..., help="Path to a TaskEnvelope JSON file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save final envelope to file"),
    max_hops: int = typer.Option(10, "--max-hops", help="Maximum number of agent hops before stopping"),
):
    """Route a TaskEnvelope, automatically following handoffs until the task is complete."""
    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    try:
        envelope = TaskEnvelope.model_validate_json(file.read_text())
    except Exception as e:
        console.print(f"[red]Invalid envelope: {e}[/red]")
        raise typer.Exit(1)

    router = registry.load_router()

    def on_hop(hop: int, agent_id: str) -> None:
        console.print(f"  [dim]hop {hop}:[/dim] [cyan]{agent_id}[/cyan]")

    try:
        console.print(f"[bold]Chaining[/bold] (max {max_hops} hops)...")
        result, visited = asyncio.run(router.chain(envelope, max_hops=max_hops, on_hop=on_hop))
    except RoutingError as e:
        console.print(f"[red]Routing failed:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"\n[green]Done[/green] — {len(visited)} hop(s): {' → '.join(visited)}")

    json_str = result.model_dump_json(indent=2)
    console.print(Syntax(json_str, "json", theme="monokai"))

    if output:
        output.write_text(json_str)
        console.print(f"\n[green]Saved to {output}[/green]")


@app.command()
def agents():
    """List all agents in the local registry."""
    entries = registry.list_all()

    if not entries:
        console.print("[dim]No agents registered. Use [bold]oap register[/bold] to add one.[/dim]")
        return

    table = Table(show_header=True, header_style="bold dim")
    table.add_column("Agent ID")
    table.add_column("URL")
    table.add_column("Capabilities")

    for entry in entries:
        table.add_row(
            f"[cyan]{entry['id']}[/cyan]",
            entry["url"],
            ", ".join(entry["capabilities"]),
        )

    console.print(table)
