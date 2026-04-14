from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from oap.envelope import TaskEnvelope
from oap.router import OAPRouter, RoutingError
from oap import registry
from oap.transport.http import HTTPTransport

app = typer.Typer(
    name="oap",
    help="Open Agent Protocol — route tasks between agents.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _fetch_agent_info(url: str, timeout: float = 5.0) -> dict | None:
    """Hit GET / on an agent URL. Returns parsed JSON or None on failure."""
    transport = HTTPTransport(base_url=url, timeout=timeout)
    try:
        response = await transport.get("/")
        if response.status_code == 200:
            try:
                return response.json()
            except Exception:
                return None
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
        pass
    return None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

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
    capabilities: Optional[str] = typer.Option(None, "--capabilities", "-c", help="Comma-separated capability keywords (fallback if GET / returns none)"),
    timeout: float = typer.Option(60.0, "--timeout", "-t", help="Request timeout in seconds"),
):
    """Register an HTTP agent. Discovers capabilities automatically from GET /."""
    info = asyncio.run(_fetch_agent_info(url, timeout=timeout))

    discovered_caps: list[str] = []
    description = ""
    source = "manual"

    if info and info.get("capabilities"):
        discovered_caps = [c.strip() for c in info["capabilities"]]
        description = info.get("description", "")
        source = "discovered"
        console.print(f"[dim]Discovered from agent:[/dim] capabilities={discovered_caps}")
        if description:
            console.print(f"[dim]Description:[/dim] {description}")
    elif capabilities:
        discovered_caps = [c.strip() for c in capabilities.split(",")]
    else:
        # GET / either failed or returned no capabilities — and no --capabilities given
        if info is None:
            console.print(
                f"[red]Agent at {url} has no GET / endpoint.[/red]\n"
                f"Add one that returns {{agent_id, capabilities, description}} "
                f"or pass [bold]--capabilities[/bold] manually."
            )
        else:
            console.print(
                f"[red]Agent at {url} returned no capabilities.[/red]\n"
                f"Add a GET / endpoint that returns {{agent_id, capabilities, description}} "
                f"or pass [bold]--capabilities[/bold] manually."
            )
        raise typer.Exit(1)

    # Use agent_id from GET / response if present, otherwise use CLI argument
    final_agent_id = (info or {}).get("agent_id") or agent_id

    registry.add(final_agent_id, url, discovered_caps, timeout=timeout, description=description)

    console.print(f"[green]Registered[/green] [cyan]{final_agent_id}[/cyan] → {url} [dim]({source})[/dim]")
    console.print(f"[dim]Capabilities:[/dim] {', '.join(discovered_caps)}")
    console.print(f"[dim]Timeout:[/dim] {timeout}s")


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
    pipeline: Optional[str] = typer.Option(None, "--pipeline", help="Comma-separated list of agent IDs to invoke in order, bypassing capability matching"),
):
    """Route a TaskEnvelope, automatically following handoffs until the task is complete.

    Use --pipeline to force a fixed sequence of agents regardless of handoffs.
    """
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
        if pipeline:
            agent_ids = [a.strip() for a in pipeline.split(",")]
            total = len(agent_ids)

            def on_pipeline_hop(hop: int, tot: int, agent_id: str) -> None:
                console.print(f"  [dim]hop {hop}/{tot}[/dim] → [cyan]{agent_id}[/cyan]")

            console.print(f"[bold]Pipeline[/bold] ({total} agent(s))...")
            result, visited = asyncio.run(
                router.run_pipeline(envelope, agent_ids, on_hop=on_pipeline_hop)
            )
        else:
            def on_hop(hop: int, agent_id: str) -> None:
                console.print(f"  [dim]hop {hop}:[/dim] [cyan]{agent_id}[/cyan]")

            console.print(f"[bold]Chaining[/bold] (max {max_hops} hops)...")
            result, visited = asyncio.run(
                router.chain(envelope, max_hops=max_hops, on_hop=on_hop)
            )
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
def ping():
    """Check reachability of all registered agents. Updates capabilities if changed."""
    entries = registry.list_all()

    if not entries:
        console.print("[dim]No agents registered.[/dim]")
        return

    async def check(entry: dict) -> dict:
        transport = HTTPTransport(base_url=entry["url"], timeout=5.0)
        start = time.monotonic()
        try:
            response = await transport.get("/")
            elapsed = int((time.monotonic() - start) * 1000)

            if response.status_code >= 400:
                return {
                    "id": entry["id"], "status": "no health endpoint",
                    "ms": elapsed, "ok": True,
                    "caps_updated": False, "new_caps": None,
                }

            # Alive — check if capabilities changed
            caps_updated = False
            new_caps = None
            try:
                info = response.json()
                remote_caps = info.get("capabilities")
                remote_desc = info.get("description", "")
                if remote_caps and sorted(remote_caps) != sorted(entry["capabilities"]):
                    registry.add(
                        entry["id"], entry["url"], remote_caps,
                        timeout=entry["timeout"], description=remote_desc,
                    )
                    caps_updated = True
                    new_caps = remote_caps
            except Exception:
                pass

            return {
                "id": entry["id"], "status": "alive",
                "ms": elapsed, "ok": True,
                "caps_updated": caps_updated, "new_caps": new_caps,
            }
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
            elapsed = int((time.monotonic() - start) * 1000)
            return {
                "id": entry["id"], "status": "dead",
                "ms": elapsed, "ok": False,
                "caps_updated": False, "new_caps": None,
            }

    async def _run_all() -> list[dict]:
        return await asyncio.gather(*[check(e) for e in entries])

    results = asyncio.run(_run_all())

    table = Table(show_header=True, header_style="bold dim")
    table.add_column("Agent ID")
    table.add_column("URL")
    table.add_column("Status")
    table.add_column("ms", justify="right")

    any_dead = False
    for res, entry in zip(results, entries):
        if res["status"] == "alive":
            note = " [yellow](capabilities updated)[/yellow]" if res["caps_updated"] else ""
            status_str = f"[green]alive[/green]{note}"
        elif res["status"] == "dead":
            status_str = "[red]dead[/red]"
            any_dead = True
        else:
            status_str = "[dim]no health endpoint[/dim]"

        table.add_row(
            f"[cyan]{res['id']}[/cyan]",
            entry["url"],
            status_str,
            str(res["ms"]),
        )

    console.print(table)

    if any_dead:
        raise typer.Exit(1)


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
    table.add_column("Timeout")
    table.add_column("Description")

    for entry in entries:
        table.add_row(
            f"[cyan]{entry['id']}[/cyan]",
            entry["url"],
            ", ".join(entry["capabilities"]),
            f"{entry['timeout']}s",
            entry.get("description", "") or "—",
        )

    console.print(table)
