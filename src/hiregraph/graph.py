"""Graph assembly for the HireGraph workflow.

START -> ingest_resume_and_jd -> classify_seniority -> plan_required_skills
  -> (A) assign_skill_workers conditional edge: one Send("skill_worker") per skill
  -> (B) static fan-out: experience_scorer | education_scorer | signal_scorer | research_agent
  -> aggregate_scores  (reducer on completed_scores + barrier)
  -> recommendation?:  advance -> draft_email
                       borderline -> human_review (interrupt)
                       reject -> draft_rejection
  draft_email <-> critic_loop -> send_email_update_ats
  human_review -> draft_email | draft_rejection
  draft_rejection -> send_email_update_ats
  send_email_update_ats -> compensate (handled failure) | finalize (sent)
  compensate -> finalize -> END
"""

from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph

from hiregraph import nodes as N
from hiregraph.config import get_settings
from hiregraph.logging_config import get_logger
from hiregraph.state import HireGraphState

log = get_logger("hiregraph.graph")

# Pattern B: the fixed set of scorers that always run (static parallelization).
_PARALLEL = ["experience_scorer", "education_scorer", "signal_scorer", "research_agent"]


def build_graph() -> StateGraph:
    """Wire every node and the minimal static edges; return the uncompiled graph."""
    g = StateGraph(HireGraphState)

    # --- Intake spine (ingest runs the parse -> normalize -> extract chain) --
    g.add_node("ingest_resume_and_jd", N.ingest_resume_and_jd)
    g.add_node("classify_seniority", N.classify_seniority)
    g.add_node("plan_required_skills", N.plan_required_skills)

    # --- The per-skill worker + the fixed scorers ----------------------------
    g.add_node("skill_worker", N.skill_worker)
    g.add_node("experience_scorer", N.experience_scorer)
    g.add_node("education_scorer", N.education_scorer)
    g.add_node("signal_scorer", N.signal_scorer)
    g.add_node("research_agent", N.research_agent)

    # --- Aggregate (fan-in reducer + barrier) + recommendation router --------
    g.add_node("aggregate_scores", N.aggregate_scores)
    g.add_node("route_recommendation", N.route_recommendation)

    # --- Email draft + critic revision loop + human review -------------------
    g.add_node("draft_email", N.draft_email)
    g.add_node("critic_loop", N.critic_loop)
    g.add_node("human_review", N.human_review)
    g.add_node("draft_rejection", N.draft_rejection)

    # --- Send + compensation record + finalize ------------------------------
    g.add_node("send_email_update_ats", N.send_email_update_ats)
    g.add_node("compensate", N.compensate)
    g.add_node("finalize", N.finalize)

    # --- Fan-out (A: Send conditional edge, B: static edges) + fan-in --------
    g.add_edge(START, "ingest_resume_and_jd")
    g.add_conditional_edges("plan_required_skills", N.assign_skill_workers, ["skill_worker"])
    g.add_edge("skill_worker", "aggregate_scores")
    for name in _PARALLEL:
        g.add_edge("plan_required_skills", name)
        g.add_edge(name, "aggregate_scores")

    return g


_REDIS_CM = None  # keep the RedisSaver context manager alive for the process


def get_checkpointer():
    """Return the configured checkpointer (Redis when wired, else in-memory)."""
    settings = get_settings()
    if settings.checkpointer == "redis" and settings.redis_url:
        try:
            from langgraph.checkpoint.redis import RedisSaver

            global _REDIS_CM
            _REDIS_CM = RedisSaver.from_conn_string(settings.redis_url)
            saver = _REDIS_CM.__enter__()
            saver.setup()
            log.info("checkpointer: Redis (%s)", settings.redis_url.split("@")[-1])
            return saver
        except Exception as exc:  # pragma: no cover - infra fallback
            log.warning("Redis checkpointer unavailable (%s); using in-memory", exc)
    log.info("checkpointer: in-memory")
    return InMemorySaver()


def compile_graph(checkpointer=None):
    """Compile with a checkpointer so interrupts resume (Redis or in-memory)."""
    return build_graph().compile(checkpointer=checkpointer or get_checkpointer())


def _architecture_mermaid() -> str:
    """Return the styled Mermaid representation of the compiled workflow."""
    return """graph TD
    START([START]):::start --> ingest["ingest_resume_and_jd<br/>data: parse-normalize-extract chain"]:::data
    ingest --> classify["classify_seniority<br/>LLM, with_structured_output"]:::llm
    classify --> plan["plan_required_skills<br/>LLM, with_structured_output"]:::llm

    subgraph OW["Orchestrator and worker"]
      sw["skill_worker xN<br/>skill #1, #2, #3 ..."]:::llm
    end
    subgraph PAR["Parallelization with reducer"]
      exps["experience_scorer<br/>LLM"]:::llm
      edus["education_scorer<br/>LLM"]:::llm
      sigs["signal_scorer<br/>LLM"]:::llm
      research["research_agent<br/>agent + tools"]:::agent
    end
    plan == assign_skill_workers - one Send per skill ==> sw
    plan -.-> exps
    plan -.-> edus
    plan -.-> sigs
    plan -.-> research
    research -. tavily_search / github_lookup .-> research

    sw --> agg["aggregate_scores<br/>reducer: Annotated list, operator.add"]:::data
    exps --> agg
    edus --> agg
    sigs --> agg
    research --> agg

    agg --> rec{"recommendation?"}:::decision
    rec -- advance --> draft["draft_email<br/>LLM"]:::llm
    rec -- borderline --> review["human_review<br/>LangGraph interrupt + checkpoint"]:::human
    rec -- reject --> rej["draft_rejection<br/>LLM"]:::llm
    review -. paused thread .-> person["Recruiter decision<br/>terminal A/R or API"]:::human
    person -- Command resume: approved --> draft
    person -- Command resume: rejected --> rej
    draft --> critic["critic_loop<br/>evaluator + optimizer, max 3"]:::llm
    critic -. not yet good enough .-> draft
    critic -- accepted --> send["send_email + update_ats<br/>handled failures route to compensation"]:::action
    rej -- rejection email --> send

    subgraph SAGA["Saga / compensation"]
      send -. handled delivery failure .-> comp["compensate<br/>record failed side effects"]:::action
    end
    send -- sent --> fin["finalize<br/>write audit trail"]:::data
    comp -- compensated --> fin
    fin --> ENDN([END]):::terminal

    classDef start fill:#d5e8d4,stroke:#2f5b22,color:#000
    classDef terminal fill:#f8cecc,stroke:#6b1d1d,color:#000
    classDef data fill:#e1d5e7,stroke:#3f2b6c,color:#000
    classDef llm fill:#fff2cc,stroke:#9a7d0a,color:#000
    classDef action fill:#ffe6cc,stroke:#a0522d,color:#000
    classDef human fill:#f5d0d0,stroke:#8b0000,color:#000
    classDef agent fill:#cce5ff,stroke:#1a5fb4,color:#000
    classDef decision fill:#fff2cc,stroke:#9a7d0a,color:#000
"""


def render_graph_png(path: str = "graph_out/graph.png") -> str:
    """Write the styled architecture diagram as both a PNG and its mermaid source."""
    from langchain_core.runnables.graph_mermaid import draw_mermaid_png

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mermaid = _architecture_mermaid()
    out.with_suffix(".mmd").write_text(mermaid, encoding="utf-8")
    try:
        out.write_bytes(draw_mermaid_png(mermaid, max_retries=5, retry_delay=2.0))
        log.info("graph rendered -> %s (+ .mmd)", out)
        return str(out)
    except Exception as exc:  # offline -> the .mmd is already written
        log.warning("PNG render failed (%s); mermaid source at %s", exc, out.with_suffix(".mmd"))
        return str(out.with_suffix(".mmd"))
