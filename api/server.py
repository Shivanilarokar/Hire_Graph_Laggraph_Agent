"""FastAPI backend for the HireGraph UI.

Endpoints
  GET  /api/health             -> liveness
  GET  /api/samples            -> available sample résumés + JDs
  POST /api/run                -> start a run; returns thread_id + status + state
  POST /api/resume/{thread_id} -> resume a paused (borderline) run with a decision
  GET  /api/graph.png          -> the compiled graph image

The compiled graph is module-level with the configured checkpointer (Redis or
in-memory) so a paused run resumes by ``thread_id`` across HTTP requests.

Run:  uv run uvicorn api.server:app --reload --port 8000
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langgraph.types import Command
from pydantic import BaseModel

from hiregraph.graph import compile_graph, render_graph_png
from hiregraph.logging_config import get_logger, setup_logging

setup_logging()
log = get_logger("hiregraph.api")

app = FastAPI(title="HireGraph API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_GRAPH = compile_graph()
_DATA = Path("sample_data")
render_graph_png()  # ensure graph_out/graph.png exists for the UI


class RunRequest(BaseModel):
    resume_path: str | None = None
    jd_path: str | None = None
    resume_text: str | None = None
    jd_text: str | None = None


class ResumeRequest(BaseModel):
    approved: bool = True


def _serialize(state: dict) -> dict:
    """Project the LangGraph state into a small JSON payload for the UI."""
    scores = state.get("completed_scores") or []
    by_kind: dict[str, list[int]] = {}
    for s in scores:
        by_kind.setdefault(s["kind"], []).append(s["score"])
    dims = {k: round(sum(v) / len(v)) for k, v in by_kind.items() if v}
    return {
        "candidate_name": state.get("candidate_name"),
        "github_username": state.get("github_username"),
        "classification": state.get("classification"),
        "recommendation": state.get("recommendation"),
        "final_score": state.get("final_score"),
        "dimensions": dims,
        "skill_scores": [s for s in scores if s.get("kind") == "skill"],
        "email_draft": state.get("email_draft"),
        "research": (state.get("research_findings") or {}).get("summary"),
        "sent_status": state.get("sent_status"),
        "terminal_status": state.get("terminal_status"),
        "audit_trail": state.get("audit_trail", []),
    }


def _interrupt_value(state: dict):
    intr = state.get("__interrupt__")
    return intr[0].value if intr else None


def _write_temp(text: str, suffix: str) -> str:
    f = Path(tempfile.gettempdir()) / f"hiregraph_{uuid.uuid4().hex[:8]}{suffix}"
    f.write_text(text, encoding="utf-8")
    return str(f)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/samples")
def samples() -> dict:
    return {
        "resumes": sorted(
            p.name for p in (_DATA / "resumes").glob("*") if p.suffix in (".md", ".pdf")
        ),
        "jds": sorted(p.name for p in (_DATA / "jds").glob("*.md")),
    }


@app.post("/api/run")
def run(req: RunRequest) -> dict:
    resume = str(_DATA / "resumes" / req.resume_path) if req.resume_path else None
    jd = str(_DATA / "jds" / req.jd_path) if req.jd_path else None
    if req.resume_text:
        resume = _write_temp(req.resume_text, ".md")
    if req.jd_text:
        jd = _write_temp(req.jd_text, ".md")
    if not resume or not jd:
        raise HTTPException(400, "Provide resume_path/jd_path or resume_text/jd_text")

    thread_id = uuid.uuid4().hex[:12]
    cfg = {"configurable": {"thread_id": thread_id}}
    log.info("run %s : %s vs %s", thread_id, Path(resume).name, Path(jd).name)
    try:
        state = _GRAPH.invoke({"resume_path": resume, "jd_path": jd}, cfg)
    except Exception as exc:
        log.error("run %s failed: %s", thread_id, exc)
        raise HTTPException(500, f"Run failed: {exc}") from exc
    paused = bool(_GRAPH.get_state(cfg).next)
    return {
        "thread_id": thread_id,
        "status": "paused" if paused else "done",
        "interrupt": _interrupt_value(state),
        "state": _serialize(state),
    }


@app.post("/api/resume/{thread_id}")
def resume(thread_id: str, req: ResumeRequest) -> dict:
    cfg = {"configurable": {"thread_id": thread_id}}
    log.info("resume %s approved=%s", thread_id, req.approved)
    try:
        state = _GRAPH.invoke(Command(resume={"approved": req.approved}), cfg)
    except Exception as exc:
        log.error("resume %s failed: %s", thread_id, exc)
        raise HTTPException(500, f"Resume failed: {exc}") from exc
    paused = bool(_GRAPH.get_state(cfg).next)
    return {
        "thread_id": thread_id,
        "status": "paused" if paused else "done",
        "interrupt": _interrupt_value(state),
        "state": _serialize(state),
    }


@app.get("/api/graph.png")
def graph_png() -> FileResponse:
    p = Path("graph_out/graph.png")
    if not p.exists():
        render_graph_png()
    return FileResponse(p, media_type="image/png")
