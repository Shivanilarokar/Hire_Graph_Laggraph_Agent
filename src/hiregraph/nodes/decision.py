"""The decision half of the graph: score -> route -> draft -> review -> send.

``aggregate_scores`` combines every completed score into a single ``final_score``,
then ``route_recommendation`` sends the candidate down one of three paths:
  - advance    -> draft_email
  - borderline -> human_review (pauses for a person) -> draft_email or draft_rejection
  - reject     -> draft_rejection -> send_email_update_ats

``draft_email`` and ``critic_loop`` trade off until the email is good enough (or
the attempt cap is hit), then ``send_email_update_ats`` delivers it. A handled
delivery failure routes to ``compensate``, which records the partial failure for
reconciliation. Every path converges on ``finalize`` before END.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END
from langgraph.types import Command, interrupt

from hiregraph.audit import audited
from hiregraph.config import get_settings
from hiregraph.llm import get_llm
from hiregraph.logging_config import get_logger
from hiregraph.prompts import build_critique_prompt, build_email_prompt
from hiregraph.schemas import CriticFeedback
from hiregraph.services import ATSError, EmailSendError, log_to_ats
from hiregraph.services import send_email as _send_email
from hiregraph.state import HireGraphState

log = get_logger("hiregraph.decision")

# Per-seniority (advance, borderline) bar. The seniority the classifier returns
# picks the row, so a senior is held to a higher bar than a junior.
_BARS = {"junior": (60, 45), "mid": (70, 45), "senior": (74, 50), "executive": (82, 62)}
# final_score = weighted blend by score kind (skills mean + the four scorers).
_WEIGHTS = {"skill": 0.35, "experience": 0.30, "education": 0.10, "signal": 0.15, "research": 0.10}


@audited("aggregate_scores")
def aggregate_scores(state: HireGraphState) -> Command[Literal["route_recommendation"]]:
    """Fan-in (barrier + reducer): combine every completed_scores entry - all
    skill-worker scores plus experience, education, signals, research - into one
    final_score and a seniority-aware recommendation."""
    by_kind: dict[str, list[int]] = {}
    for s in state.get("completed_scores") or []:
        by_kind.setdefault(s["kind"], []).append(s["score"])

    means = {k: (round(sum(v) / len(v)) if v else 0) for k, v in by_kind.items()}
    final = round(sum(_WEIGHTS.get(k, 0) * means.get(k, 0) for k in _WEIGHTS))

    seniority = (state.get("classification") or {}).get("seniority", "mid")
    adv, bor = _BARS.get(seniority, (70, 45))
    rec = "advance" if final >= adv else "borderline" if final >= bor else "reject"

    name = state.get("candidate_name", "Candidate")
    summary = f"{name} scored {final}/100 ({seniority}) -> {rec}."
    log.info(
        "SCORECARD  %s -> final %d/100 | %s | skill=%d exp=%d edu=%d signal=%d research=%d",
        name, final, rec.upper(),
        means.get("skill", 0), means.get("experience", 0), means.get("education", 0),
        means.get("signal", 0), means.get("research", 0),
    )
    return Command(
        update={"final_score": final, "recommendation": rec, "scorecard_summary": summary},
        goto="route_recommendation",
    )


@audited("route_recommendation")
def route_recommendation(
    state: HireGraphState,
) -> Command[Literal["draft_email", "human_review", "draft_rejection"]]:
    """Branch on the recommendation: advance -> draft, borderline -> human_review
    (interrupt), reject -> rejection email."""
    rec = state.get("recommendation")
    goto = (
        "draft_email" if rec == "advance"
        else "human_review" if rec == "borderline"
        else "draft_rejection"
    )
    log.info("DECISION   %s -> %s", str(rec).upper(), goto)
    return Command(goto=goto)


@audited("draft_email")
def draft_email(state: HireGraphState) -> Command[Literal["critic_loop"]]:
    """Draft a personalized advance/borderline email; draft_attempts increments."""
    content = get_llm().invoke(build_email_prompt(state)).content
    return Command(
        update={"email_draft": content, "draft_attempts": state.get("draft_attempts", 0) + 1},
        goto="critic_loop",
    )


@audited("critic_loop")
def critic_loop(
    state: HireGraphState,
) -> Command[Literal["draft_email", "send_email_update_ats"]]:
    """The critic grades the draft; if it isn't good enough and we're still under
    the attempt cap, loop back to draft_email to revise, otherwise send. (Borderline
    candidates were already gated by human_review before drafting.)"""
    settings = get_settings()
    critique: CriticFeedback = get_llm().with_structured_output(CriticFeedback).invoke(
        build_critique_prompt(state)
    )
    fb = [critique.model_dump()]
    if not critique.approved and state.get("draft_attempts", 0) < settings.email_max_revisions:
        return Command(update={"critic_feedback": fb}, goto="draft_email")
    return Command(update={"critic_feedback": fb}, goto="send_email_update_ats")


@audited("human_review")
def human_review(state: HireGraphState) -> Command[Literal["draft_email", "draft_rejection"]]:
    """Borderline candidates pause here for a human decision. ``interrupt()`` saves
    state and hands control back to the caller; approving drafts the advance email,
    rejecting drafts a rejection. The same thread_id resumes via Command(resume=...)."""
    decision = interrupt(
        {
            "candidate": state.get("candidate_name", ""),
            "recommendation": state.get("recommendation", ""),
            "final_score": state.get("final_score"),
            "action": "approve to advance this borderline candidate, or reject.",
        }
    )
    approved = decision.get("approved") if isinstance(decision, dict) else bool(decision)
    if not approved:
        return Command(update={"human_decision": "reject"}, goto="draft_rejection")
    return Command(update={"human_decision": "approve"}, goto="draft_email")


@audited("draft_rejection")
def draft_rejection(state: HireGraphState) -> Command[Literal["send_email_update_ats"]]:
    """Draft a kind rejection email, then send it through the shared delivery path."""
    rejection_state = {**state, "recommendation": "reject"}
    content = get_llm().invoke(build_email_prompt(rejection_state)).content
    return Command(
        update={"email_draft": content, "recommendation": "reject"},
        goto="send_email_update_ats",
    )


@audited("send_email_update_ats")
def send_email_update_ats(state: HireGraphState) -> Command[Literal["finalize", "compensate"]]:
    """Send the email and write the ATS - two side effects in one step.

    Typed delivery failures become explicit graph state and route to compensate.
    This node does not claim transactional rollback across SMTP and the ATS.
    """
    decision = {
        "candidate": state.get("candidate_name"),
        "recommendation": state.get("recommendation"),
        "final_score": state.get("final_score"),
    }
    try:
        _send_email(
            state.get("candidate_email") or "candidate@example.com",
            f"Your application - {state.get('recommendation', 'update')}",
            state.get("email_draft", ""),
        )
        ats_id = log_to_ats(decision, force_failure=state.get("force_ats_failure", False))
    except (EmailSendError, ATSError) as exc:
        return Command(
            update={"sent_status": "failed", "compensation_log": [f"Delivery failed: {exc}"]},
            goto="compensate",
        )
    return Command(update={"sent_status": "sent", "ats_id": ats_id}, goto="finalize")


@audited("compensate")
def compensate(state: HireGraphState) -> Command[Literal["finalize"]]:
    """Record a handled side-effect failure, then reach finalize cleanly."""
    return Command(
        update={
            "compensation_log": [
                "Delivery requires reconciliation; external side effects were not rolled back."
            ],
            "terminal_status": "compensated",
        },
        goto="finalize",
    )


@audited("finalize")
def finalize(state: HireGraphState) -> Command[Literal["__end__"]]:
    """Single terminal node: settle the terminal status and end (writes audit)."""
    return Command(update={"terminal_status": state.get("terminal_status") or "completed"}, goto=END)
