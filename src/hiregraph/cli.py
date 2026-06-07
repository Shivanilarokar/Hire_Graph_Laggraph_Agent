"""Command-line runner for the canned demo.

On a fresh clone with keys set this compiles the graph, writes the PNG, runs the
four canned scenarios (strong / borderline / mismatch / PDF) - auto-approving at the
borderline interrupt - and prints a per-node trace plus a final scoreboard.
"""

from __future__ import annotations

import time
import uuid
import warnings
from pathlib import Path

from langgraph.types import Command
from rich.console import Console
from rich.table import Table

from hiregraph.graph import compile_graph, render_graph_png
from hiregraph.logging_config import get_logger, setup_logging

# LangGraph warns when it round-trips our Pydantic schemas through the in-memory
# checkpointer. The data deserializes correctly; silence the noise for the demo.
warnings.filterwarnings("ignore", message=".*unregistered type.*")

log = get_logger("hiregraph.cli")
console = Console()
_DATA = Path("sample_data")

SCENARIOS = [
    ("Priya (strong senior)", _DATA / "resumes/resume_priya.md", _DATA / "jds/jd_senior_backend.md"),
    ("Eitan (borderline mid)", _DATA / "resumes/resume_eitan.md", _DATA / "jds/jd_senior_backend.md"),
    ("Mira (frontend mismatch)", _DATA / "resumes/resume_mira.md", _DATA / "jds/jd_senior_backend.md"),
    ("Shivani (PDF resume)", _DATA / "resumes/Shivani_Resume.pdf", _DATA / "jds/jd_senior_backend.md"),
]


def _run(app, name: str, resume: Path, jd: Path) -> dict:
    """Invoke one scenario, auto-approving through any borderline interrupt."""
    config = {"configurable": {"thread_id": f"{name}-{uuid.uuid4().hex[:8]}"}}
    start = time.perf_counter()
    result = app.invoke({"resume_path": str(resume), "jd_path": str(jd)}, config)
    snapshot = app.get_state(config)
    paused = bool(snapshot.next)
    while snapshot.next:  # paused at human_review -> auto-approve and resume
        console.print(f"  [yellow]interrupt[/]: {name} paused for human review -> auto-approving")
        result = app.invoke(Command(resume={"approved": True}), config)
        snapshot = app.get_state(config)
    result["_elapsed_ms"] = round((time.perf_counter() - start) * 1000, 1)
    result["_paused"] = paused
    result["_resume"] = resume.name
    result["_jd"] = jd.name
    return result


def _print_trace(name: str, result: dict) -> None:
    console.print(f"\n[bold cyan]{name}[/]")
    for entry in result.get("audit_trail", []):
        console.print(f"  - {entry['node']:<22}{entry['verdict']:<24}{entry['elapsed_ms']:>8.1f} ms")


def _scoreboard(rows: list[tuple[str, dict]]) -> None:
    """Scoreboard in the expected_outcomes.md shape: Resume, JD, seniority,
    recommendation, and whether the run paused for human review."""
    table = Table(title="HireGraph Scoreboard", show_lines=True)
    for col in ("Resume", "JD", "Seniority", "Final", "Recommendation", "Pause for review?"):
        table.add_column(col)
    for _name, r in rows:
        cls = r.get("classification") or {}
        rec = r.get("recommendation", "-")
        color = {"advance": "green", "borderline": "yellow", "reject": "red"}.get(rec, "white")
        pause = "[yellow]yes[/]" if r.get("_paused") else "no"
        table.add_row(
            r.get("_resume", "-"),
            r.get("_jd", "-"),
            str(cls.get("seniority", "-")),
            str(r.get("final_score", "-")),
            f"[{color}]{rec}[/]",
            pause,
        )
    console.print(table)


def main() -> None:
    """Entry point used by ``main.py`` and the ``hiregraph`` console script."""
    setup_logging()
    console.rule("[bold]HireGraph - LangGraph Smart Hiring Assistant")
    png = render_graph_png()
    console.print(f"Compiled graph rendered -> [green]{png}[/]")

    app = compile_graph()
    rows: list[tuple[str, dict]] = []
    for name, resume, jd in SCENARIOS:
        try:
            result = _run(app, name, resume, jd)
        except Exception as exc:  # one bad scenario shouldn't sink the whole demo
            log.error("scenario %s failed: %s", name, exc)
            console.print(f"[red]{name} failed:[/] {exc}")
            continue
        rows.append((name, result))
        _print_trace(name, result)

    console.print()
    if rows:
        _scoreboard(rows)
    else:
        console.print(
            "[yellow]No scenarios completed. In real mode this usually means no "
            "internet / API access. Check your connection, or run offline with "
            "HIREGRAPH_USE_MOCKS=true.[/]"
        )


if __name__ == "__main__":
    main()
