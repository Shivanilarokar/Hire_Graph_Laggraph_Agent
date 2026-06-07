"""Interactive CLI support for running one resume against one job description."""

from __future__ import annotations

import argparse
import uuid
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langgraph.types import Command

from hiregraph.config import get_settings
from hiregraph.graph import compile_graph

InputFn = Callable[[str], str]
PrintFn = Callable[[str], None]

DEFAULT_RESUME = "sample_data/resumes/resume_priya.md"
DEFAULT_JD = "sample_data/jds/jd_senior_backend.md"


def prompt_for_decision(
    review: dict[str, Any],
    *,
    input_fn: InputFn = input,
    print_fn: PrintFn = print,
) -> bool:
    """Display an interrupt payload and wait for an approve/reject decision."""
    print_fn("")
    print_fn("=" * 64)
    print_fn("HUMAN REVIEW REQUIRED - LANGGRAPH EXECUTION IS PAUSED")
    print_fn("=" * 64)
    print_fn(f"Candidate:      {review.get('candidate') or 'Unknown'}")
    print_fn(f"Final score:    {review.get('final_score')}")
    print_fn(f"Recommendation: {review.get('recommendation') or 'borderline'}")
    if review.get("action"):
        print_fn(f"Action:         {review['action']}")

    while True:
        answer = input_fn("Decision [A]pprove / [R]eject: ").strip().lower()
        if answer in {"a", "approve", "approved", "y", "yes"}:
            print_fn("Human decision recorded: APPROVE")
            return True
        if answer in {"r", "reject", "rejected", "n", "no"}:
            print_fn("Human decision recorded: REJECT")
            return False
        print_fn("Please enter A or R.")


def _interrupt_payload(output: dict[str, Any], snapshot: Any) -> dict[str, Any]:
    """Extract the value supplied to LangGraph interrupt()."""
    interrupts = output.get("__interrupt__") or getattr(snapshot, "interrupts", ())
    if interrupts:
        value = getattr(interrupts[0], "value", interrupts[0])
        if isinstance(value, dict):
            return value
    values = getattr(snapshot, "values", {}) or {}
    return {
        "candidate": values.get("candidate_name"),
        "recommendation": values.get("recommendation"),
        "final_score": values.get("final_score"),
        "action": "approve to advance this borderline candidate, or reject.",
    }


def run_resume(
    app: Any,
    resume: str,
    jd: str,
    *,
    thread_id: str | None = None,
    auto_approve: bool = False,
    input_fn: InputFn = input,
    print_fn: PrintFn = print,
) -> dict[str, Any]:
    """Run one graph thread, pausing for terminal input at human_review."""
    run_thread_id = thread_id or f"resume-{Path(resume).stem}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": run_thread_id}}
    print_fn(f"thread_id: {run_thread_id}")

    snapshot = app.get_state(config)
    if snapshot.next:
        print_fn("Resuming existing paused thread from its checkpoint.")
        output = dict(snapshot.values)
    else:
        output = app.invoke({"resume_path": resume, "jd_path": jd}, config)
        snapshot = app.get_state(config)
    paused = bool(snapshot.next)

    while snapshot.next:
        if "human_review" not in snapshot.next:
            raise RuntimeError(f"Graph paused at unexpected node(s): {snapshot.next}")

        review = _interrupt_payload(output, snapshot)
        if auto_approve:
            print_fn("")
            print_fn("[interrupt] human review paused the graph -> auto-approving")
            approved = True
        else:
            approved = prompt_for_decision(review, input_fn=input_fn, print_fn=print_fn)

        output = app.invoke(Command(resume={"approved": approved}), config)
        snapshot = app.get_state(config)

    result = dict(output)
    result["_paused"] = paused
    result["_thread_id"] = run_thread_id
    return result


def _print_mode(print_fn: PrintFn) -> None:
    settings = get_settings()
    print_fn(
        "mode: "
        f"{'MOCK (offline)' if settings.llm_is_mock else 'REAL (OpenAI)'}"
        f" | tavily: {'mock' if settings.tavily_is_mock else 'REAL'}"
        f" | github: {'mock' if settings.use_mocks else 'REAL'}"
        f" | email: {'mock' if settings.email_is_mock else 'REAL'}"
    )


def _print_result(result: dict[str, Any], print_fn: PrintFn) -> None:
    classification = result.get("classification") or {}
    scores = result.get("completed_scores") or []
    print_fn("")
    print_fn(f"  candidate:      {result.get('candidate_name')}")
    print_fn(
        "  seniority:      "
        f"{classification.get('seniority')} ({classification.get('role_family')})"
    )
    print_fn(
        "  skill workers:  "
        f"{sum(1 for score in scores if score['kind'] == 'skill')} fanned out via Send"
    )
    print_fn(f"  final_score:    {result.get('final_score')}")
    print_fn(
        "  recommendation: "
        f"{result.get('recommendation')}   | paused for review: {result.get('_paused')}"
    )
    print_fn(f"  human_decision: {result.get('human_decision') or 'not required'}")
    print_fn(
        "  research:       "
        f"{(result.get('research_findings') or {}).get('summary', '')[:140]}"
    )
    print_fn(
        f"  sent_status:    {result.get('sent_status')} "
        f"| terminal: {result.get('terminal_status')}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one resume through HireGraph and handle human review in the terminal."
    )
    parser.add_argument("resume", nargs="?", default=DEFAULT_RESUME)
    parser.add_argument("jd", nargs="?", default=DEFAULT_JD)
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve a borderline interrupt instead of prompting.",
    )
    parser.add_argument(
        "--thread-id",
        help="Use a specific LangGraph checkpoint thread ID instead of generating one.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Single-resume command entry point."""
    warnings.filterwarnings("ignore")
    args = build_parser().parse_args(argv)
    _print_mode(print)
    print(f"resume: {args.resume}\njd:     {args.jd}\n")

    thread_id = args.thread_id or f"resume-{Path(args.resume).stem}-{uuid.uuid4().hex[:8]}"
    try:
        result = run_resume(
            compile_graph(),
            args.resume,
            args.jd,
            thread_id=thread_id,
            auto_approve=args.auto_approve,
        )
    except (EOFError, KeyboardInterrupt):
        print(f"\nReview cancelled. The graph remains paused under thread_id: {thread_id}")
        return 130

    _print_result(result, print)
    return 0
