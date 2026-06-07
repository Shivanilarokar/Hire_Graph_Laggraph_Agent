# HireGraph — Architecture

**HireGraph is a real, production-shaped LangGraph application that exercises every
concept from Class 2 in one coherent system:** state design, routing,
parallelization, the orchestrator-and-worker pattern, the evaluator-and-optimizer
loop, agents with tools, retries, human-in-the-loop, and the saga pattern.

The business it models is recruiting triage. A senior recruiter drowning in
applications wants an assistant that reads a résumé and a job description, decides
the candidate's level, scores them across independent dimensions, produces an
explainable scorecard, drafts a personalized email, lets a human approve the
borderline cases, records the decision, and rolls back cleanly if a downstream
step fails. Every one of those steps is a LangGraph node; every transition is a
LangGraph edge or `Command`. The orchestration is **100% LangGraph** — no LCEL
pipelines and no `asyncio.gather` doing control flow — and the whole thing runs
end to end on a keyless clone thanks to deterministic mocks.

This document explains it top to bottom: the high-level shape, the state object,
every node and edge, the two parallel patterns, the reliability machinery, and how
each Class 2 concept is realized in code. The Python source and tests are
authoritative; this doc tracks them.

---

## Table of contents

1. [Every Class 2 concept → where it lives](#1-every-class-2-concept--where-it-lives)
2. [The shape of the graph](#2-the-shape-of-the-graph)
3. [Repository layout](#3-repository-layout)
4. [State design](#4-state-design)
5. [Control flow, node by node](#5-control-flow-node-by-node)
6. [Parallelization: two fan-outs, one fan-in](#6-parallelization-two-fan-outs-one-fan-in)
7. [Orchestrator and worker (Send)](#7-orchestrator-and-worker-send)
8. [Agent with tools (research sub-graph)](#8-agent-with-tools-research-sub-graph)
9. [Routing](#9-routing)
10. [Evaluator and optimizer (the critic loop)](#10-evaluator-and-optimizer-the-critic-loop)
11. [Human in the loop (interrupt + checkpointer)](#11-human-in-the-loop-interrupt--checkpointer)
12. [Retries and the saga](#12-retries-and-the-saga)
13. [Services and the mock strategy](#13-services-and-the-mock-strategy)
14. [Configuration](#14-configuration)
15. [Observability](#15-observability)
16. [Scoring math](#16-scoring-math)
17. [A full trace (Eitan)](#17-a-full-trace-eitan)
18. [Runtime surfaces (CLI, API, UI)](#18-runtime-surfaces-cli-api-ui)
19. [Testing](#19-testing)
20. [Notes and limitations](#20-notes-and-limitations)

---

## 1. Every Class 2 concept → where it lives

This is the heart of the design. Each pattern below is implemented as a real,
non-contrived part of the pipeline — point at the node, the edge, or the line.

| Class 2 concept | Where it lives in HireGraph |
|---|---|
| **State design** (TypedDict, raw data, reducers) | `state.py` → `HireGraphState`; `completed_scores`, `audit_trail`, `messages` use reducers |
| **Routing** | `route_recommendation` (the `recommendation?` diamond) branches advance / borderline / reject; `classify_seniority` sets the seniority that picks the bar |
| **Parallelization with a reducer** | `experience/education/signal_scorer` + workers all write `completed_scores` (`Annotated[list, operator.add]`) and join at `aggregate_scores` |
| **Orchestrator and worker (Send)** | `plan_required_skills` plans the skills; `assign_skill_workers` returns `[Send("skill_worker", …)]`, one worker per skill, sized at runtime |
| **Evaluator and optimizer** | `draft_email ⇄ critic_loop`, bounded by `draft_attempts` ≤ `EMAIL_MAX_REVISIONS` |
| **Agent with tools + ToolNode** | `research_agent` — a compiled `llm ⇄ ToolNode` sub-graph binding `github_lookup` + `web_search` |
| **Retries (RetryPolicy + retry_on)** | `research_agent` (`TavilyError`, `GitHubError`) and `send_email_update_ats` (`EmailSendError`, `ConnectionError`) |
| **Human in the loop** | `human_review` calls `interrupt()`; resumes by `thread_id` via `Command(resume=…)` |
| **Saga / compensation** | `send_email_update_ats` failure → `compensate` (rollback + alert) → `finalize` |
| **Structured output** | every LLM call uses `with_structured_output(...)` against a Pydantic model in `schemas.py` |
| **Tool calling** | two `@tool` functions in `tools.py` wrapping the services |
| **Short-term memory / MessagesState** | the research sub-graph runs on `MessagesState` with `add_messages` |
| **Prompt chaining** | inside `ingest_resume_and_jd`: `parse_resume → normalize_skills → extract_experience` |
| **LLM-recoverable loopback** | the research `ToolNode(handle_tool_errors=True)` turns a failed tool call into an observation the agent reasons about |

---

## 2. The shape of the graph

A **linear intake spine**, a **fan-out / fan-in parallel block**, and a
**decide-and-act tail**. The committed image (`graph_out/graph.png`, generated from
`graph.py`) mirrors `docs/architecture/hiregraph_architecture.drawio`.

```
START
  └─ ingest_resume_and_jd          data: parse → normalize → extract chain
       └─ classify_seniority        LLM, structured output
            └─ plan_required_skills  LLM, structured output  (the orchestrator)
                 │
   bold ────────►│ assign_skill_workers  ── one Send per skill ──►  skill_worker × N
   dotted ───────►│ experience_scorer  education_scorer  signal_scorer  research_agent
                 ▼            (every branch appends to completed_scores)
            aggregate_scores          reducer + barrier → final_score, recommendation
                 ▼
            recommendation?           (route_recommendation)
     advance ────┤ borderline ────────┤ reject
        ▼        ▼                     ▼
   draft_email  human_review(interrupt)  draft_rejection
      ▲ ▼         │ approved → draft_email      ▼
   critic_loop    │ rejected → draft_rejection  log_rejection
      │ accepted  ▼                             ▼
      └──► send_email + update_ats ──(sent)──► finalize ──► END
                  │ retries exhausted             ▲
                  └──► compensate ────────────────┘
```

**Routing happens inside nodes.** Almost every transition is a `Command(goto=...)`
returned by a node, so the graph visualizes correctly and the wiring stays
readable. The only static edges are the parallel fan-out/fan-in into
`aggregate_scores`.

---

## 3. Repository layout

```
src/hiregraph/
  state.py          The one TypedDict that travels through the graph
  schemas.py        Pydantic models for every structured LLM call
  prompts.py        Pure functions: raw state -> prompt string
  llm.py            get_llm() factory + the deterministic MockChatModel
  services.py       External I/O (read, tavily, github, email, ATS) + mock fallback
  tools.py          @tool wrappers the research agent binds
  audit.py          @audited decorator: timing, verdict, logging, audit trail
  config.py         Typed settings from environment / .env
  logging_config.py Central logging setup
  graph.py          build_graph(), compile_graph(), render_graph_png()
  cli.py            The demo runner + scoreboard
  nodes/
    intake.py       ingest_resume_and_jd (+ the parse/normalize/extract chain)
    classify.py     classify_seniority
    scoring.py      plan_required_skills, assign_skill_workers, skill_worker, 3 scorers
    research.py     research_agent (its own llm<->ToolNode sub-graph)
    decision.py     aggregate_scores, route_recommendation, draft/critic/review,
                    draft_rejection, log_rejection, send_email_update_ats,
                    compensate, finalize
api/server.py       FastAPI backend for the UI
ui/                 React + Vite + Tailwind front-end
scripts/test_resume.py  Run a single résumé and print the verdict
tests/              state / node / end-to-end tests
graph_out/          Committed graph.png + graph.mmd
sample_data/        Résumés (.md, .pdf) and JDs
```

Each module has one job. Nodes never build prompts inline (that's `prompts.py`)
and never call external APIs directly (that's `services.py`), which keeps every
node small, single-purpose, and testable.

---

## 4. State design

Everything flows through one `TypedDict`, `HireGraphState` (`state.py`), holding
**raw data only** — structured-output results are stored as plain dicts
(`model_dump()`), so the whole state is JSON-serializable and can be persisted by
the Redis checkpointer. Prompts are never stored; they're rebuilt on demand.

| Field | Type | Written by | Read by |
|---|---|---|---|
| `candidate_id` | str | ingest | — |
| `resume_path`, `jd_path` | str | caller | ingest |
| `resume_text`, `jd_text` | str | ingest | prompts |
| `candidate_name`, `candidate_email`, `github_username` | str | ingest | classify, email, research |
| `normalized_skills` | list[str] | ingest | scorers |
| `years_experience` | float | ingest | classify, scorers |
| `classification` | dict | classify_seniority | aggregate, prompts |
| `required_skills` | list[dict] | plan_required_skills | assign_skill_workers |
| `completed_scores` | `Annotated[list, operator.add]` | every worker + scorer | aggregate_scores |
| `final_score`, `recommendation` | int / Literal | aggregate_scores | route, email, send |
| `email_draft` | str | draft_email / draft_rejection / human_review | critic, send |
| `critic_feedback` | `Annotated[list, operator.add]` | critic_loop | draft_email (next pass) |
| `draft_attempts` | int | draft_email | critic_loop |
| `human_decision` | Literal | human_review | — |
| `sent_status`, `ats_id` | Literal / str | send / log_rejection | finalize, tests |
| `terminal_status` | Literal | finalize / compensate | tests, scoreboard |
| `compensation_log` | `Annotated[list, operator.add]` | send (failure) / compensate | — |
| `audit_trail` | `Annotated[list, operator.add]` | every node (via `@audited`) | CLI trace |
| `messages` | `Annotated[list, add_messages]` | research sub-graph | research sub-graph |

**Why the parallel writes are safe.** Four keys are `Annotated` with a reducer so
concurrent writes *merge* instead of overwriting:

- `completed_scores` → `operator.add` — the fan-in. Every parallel branch returns
  `{"completed_scores": [one_dict]}`; LangGraph concatenates them into one list.
- `audit_trail`, `compensation_log` → `operator.add`.
- `messages` → `add_messages` (inside the research sub-graph).

Without these reducers, two branches finishing in the same superstep would clobber
each other. With them, fan-out is trivial.

---

## 5. Control flow, node by node

Every node is wrapped by `@audited("name")`, which times it, records a
`{node, verdict, elapsed_ms}` row into `audit_trail`, logs it, and re-raises real
exceptions while letting LangGraph's control-flow signals through. Most nodes
return `Command[Literal[...]]`, so routing is visible to LangGraph.

| Node | Kind | What it does | Routes to |
|---|---|---|---|
| `ingest_resume_and_jd` | data | Reads résumé + JD, then runs the **parse → normalize → extract** chain (3 sequential structured LLM calls) → name, skills, years | `classify_seniority` |
| `classify_seniority` | LLM | `with_structured_output(Classification)` → seniority + role_family + confidence | `plan_required_skills` |
| `plan_required_skills` | LLM | `with_structured_output(RequiredSkills)` → the skills to score (the orchestrator's plan) | (edges) |
| `assign_skill_workers` | edge fn | Conditional edge returning `[Send("skill_worker", …)]`, one per planned skill | `skill_worker × N` |
| `skill_worker` | LLM | Scores the **single** skill it was Sent; appends `{kind:"skill"}` | `aggregate_scores` |
| `experience_scorer` | LLM | Scores experience; appends `{kind:"experience"}` | `aggregate_scores` |
| `education_scorer` | LLM | Scores education; appends `{kind:"education"}` | `aggregate_scores` |
| `signal_scorer` | LLM | Scores public signals; appends `{kind:"signal"}` | `aggregate_scores` |
| `research_agent` | agent | Tool-using sub-graph; appends `{kind:"research"}` + findings summary | `aggregate_scores` |
| `aggregate_scores` | data | Blends `completed_scores` into `final_score` + recommendation | `route_recommendation` |
| `route_recommendation` | router | Branches on the recommendation | `draft_email` / `human_review` / `draft_rejection` |
| `draft_email` | LLM | Drafts the advance/borderline email; bumps `draft_attempts` | `critic_loop` |
| `critic_loop` | LLM | Grades the draft; revise or send | `draft_email` / `send_email_update_ats` |
| `human_review` | interrupt | Pauses for a person on borderline candidates | `draft_email` / `draft_rejection` |
| `draft_rejection` | LLM | Drafts a kind rejection email | `log_rejection` |
| `log_rejection` | action | Records the rejection (no email sent) | `finalize` |
| `send_email_update_ats` | action | Sends email + writes ATS (two side effects) | `finalize` / `compensate` |
| `compensate` | action | Saga rollback + alert | `finalize` |
| `finalize` | data | Settles terminal status, writes the audit trail | `END` |

---

## 6. Parallelization: two fan-outs, one fan-in

`plan_required_skills` fans out in **two different ways at once** — drawn as a bold
path and a dotted band in the diagram — and both converge on a single reducer.

- **Dynamic fan-out (orchestrator + worker), bold.** `assign_skill_workers` returns
  one `Send("skill_worker", …)` per planned skill. The worker count is decided at
  runtime from data: three skills → three workers, ten → ten.
- **Static fan-out (fixed parallelization), dotted.** Plain static edges to a known
  set: `experience_scorer`, `education_scorer`, `signal_scorer`, `research_agent`.

Both kinds write `{"completed_scores": [ {kind,label,score} ]}` and point at
`aggregate_scores`. Because `completed_scores` has the `operator.add` reducer,
LangGraph concatenates every partial write; because `aggregate_scores` is a join,
LangGraph applies an automatic **barrier** so it doesn't fire until *every*
incoming branch has finished. Fan-out (Send + static edges) and fan-in (reducer +
barrier) are the matched pair the whole top half is built around.

```python
# graph.py
g.add_conditional_edges("plan_required_skills", N.assign_skill_workers, ["skill_worker"])
g.add_edge("skill_worker", "aggregate_scores")
for name in ["experience_scorer", "education_scorer", "signal_scorer", "research_agent"]:
    g.add_edge("plan_required_skills", name)
    g.add_edge(name, "aggregate_scores")
```

---

## 7. Orchestrator and worker (Send)

The orchestrator is just an LLM deciding *how much work there is* before any work
happens, and `assign_skill_workers` translating that decision into `Send` calls:

```python
# scoring.py
def plan_required_skills(state) -> dict:                  # orchestrator: the plan
    result = get_llm().with_structured_output(RequiredSkills).invoke(build_plan_skills_prompt(state))
    return {"required_skills": [s.model_dump() for s in result.skills[:5]] or [{"name": "General Backend"}]}

def assign_skill_workers(state) -> list[Send]:            # one worker per skill
    resume = state.get("resume_text", "")
    skills = state.get("required_skills") or [{"name": "General Backend"}]
    return [Send("skill_worker", {"skill": s["name"], "resume_text": resume}) for s in skills]
```

Each worker sees only its own `Send` payload, so its prompt stays small and it is
independently retryable/observable. This is map-reduce: the planner sizes the map,
the reducer joins it.

---

## 8. Agent with tools (research sub-graph)

`research_agent` is the only agent node. It is a **self-contained compiled
sub-graph** on `MessagesState`:

```
START → llm  ⇄  tools (ToolNode)        # tools_condition loops llm→tools or stops at END
```

- `llm` binds `RESEARCH_TOOLS` (`github_lookup`, `web_search`) and decides whether
  to call a tool or stop.
- `ToolNode(RESEARCH_TOOLS, handle_tool_errors=True)` executes them; if a tool
  raises (e.g. a 404 from a missing GitHub handle), the error is fed back as an
  observation the model reasons about instead of crashing — the **recoverable
  loopback**. The agent adapts and continues.
- `tools_condition` is the prebuilt router that loops until the model stops
  calling tools.

Encapsulating it as a sub-graph means that, from the main graph, it is just **one
ordinary parallel branch** (fixed-width fan-in), even though internally it may take
many turns. It writes back only a short `research_findings` summary and a research
score — the message objects stay inside the sub-graph, keeping the main state
JSON-serializable. The tools (`tools.py`) are thin `@tool` wrappers over
`services.py`, so they inherit the same real/mock behavior.

---

## 9. Routing

Two routing decisions:

- **Seniority** (`classify_seniority`) produces a typed `Classification`; the
  seniority it returns selects the hiring bar used downstream.
- **Recommendation** (`route_recommendation`, the `recommendation?` diamond) reads
  `recommendation` and sends the run to `draft_email` (advance), `human_review`
  (borderline), or `draft_rejection` (reject) via `Command(goto=...)`.

---

## 10. Evaluator and optimizer (the critic loop)

`draft_email` writes a draft and bumps `draft_attempts`; `critic_loop` grades it
with `with_structured_output(CriticFeedback)`. If the critic is unsatisfied and we
are still under `EMAIL_MAX_REVISIONS` (3), it loops back to `draft_email` to revise
— the previous critique sits in `critic_feedback`, which the next draft prompt
reads. Otherwise it proceeds to send. The attempt counter is the guardrail that
keeps the optimizer from spinning forever.

---

## 11. Human in the loop (interrupt + checkpointer)

For borderline candidates, `human_review` calls `interrupt({...})`, which
**persists state to the checkpointer and returns control to the caller**. The run
resumes only when the caller invokes the graph again with
`Command(resume={"approved": …})` on the **same `thread_id`**: approving routes to
`draft_email`, rejecting to `draft_rejection`. Because state is checkpointed, the
pause survives across separate invocations — and with the Redis checkpointer,
across a full process restart. `compile_graph()` wires `RedisSaver` when
`CHECKPOINTER=redis`, else `InMemorySaver`, falling back to memory if Redis is
unreachable so the app never crashes on a misconfig.

---

## 12. Retries and the saga

**RetryPolicy** is attached to the two nodes that touch flaky external services,
with a narrow `retry_on` so only transient errors retry:

```python
g.add_node("research_agent", N.research_agent,
           retry_policy=RetryPolicy(retry_on=(TavilyError, GitHubError), max_attempts=3))
g.add_node("send_email_update_ats", N.send_email_update_ats,
           retry_policy=RetryPolicy(retry_on=(EmailSendError, ConnectionError),
                                    max_attempts=3, initial_interval=0.5))
```

**Saga / compensation.** `send_email_update_ats` performs **two** side effects —
send the email *and* write the ATS. If a failure outlives the retries, aborting
would leave an inconsistent world (email sent but ATS not updated). Instead the
node routes to `compensate`, which rolls the partial work back, appends a
`compensation_log`, marks `terminal_status="compensated"`, and continues to
`finalize`. Every terminal path — `sent`, `compensated`, and the rejection
`logged` path — converges on `finalize`, so an audit record is **always** written
before `END`.

---

## 13. Services and the mock strategy

`services.py` is the only module that performs external I/O. Each function checks
its own key **at the call site** and degrades to a deterministic stub when the key
is missing or `USE_MOCKS=true`:

| Function | Real | Mock fallback |
|---|---|---|
| `read_document` | PyMuPDF (`.pdf`) / file read (`.md`) | — (local only) |
| `tavily_search` | Tavily API | canned "(mock) public mention…" |
| `github_lookup` | GitHub REST | `{public_repos: 12, followers: 30, mock: True}` |
| `send_email` | Mailtrap sandbox SMTP | print-only, `{sent: True, mock: True}` |
| `log_to_ats` | in-memory ATS id | same (+ a `force_failure` hook for the saga) |

Recoverable failures raise **typed exceptions** (`TavilyError`, `GitHubError`,
`EmailSendError`, `ATSError`) so `RetryPolicy(retry_on=...)` can target exactly
those. A partially configured `.env` uses real APIs where it can and mocks the
rest, with no code change. The **LLM** mirrors this in `llm.py`: `get_llm()`
returns a real `ChatOpenAI` when keyed, or a `MockChatModel` producing
deterministic structured outputs (tuned to reproduce `expected_outcomes.md`) with
zero network calls.

---

## 14. Configuration

`config.py` exposes a cached `Settings` object from the environment / `.env`
(`.env.example` lists every variable). It is wiring only — no prompts, no state.

| Variable | Default | Meaning |
|---|---|---|
| `HIREGRAPH_USE_MOCKS` (alias `USE_MOCKS`) | `false` | force every service to its mock |
| `OPENAI_API_KEY` / `HIRE_MODEL` | — / `gpt-4o-mini` | LLM provider + model |
| `TAVILY_API_KEY`, `GITHUB_TOKEN` | — | research tools |
| `MAILTRAP_USER` / `MAILTRAP_PASS` | — | sandbox email |
| `CHECKPOINTER` / `REDIS_URL` | `memory` / — | checkpointer choice |
| `EMAIL_MAX_REVISIONS` | `3` | critic loop cap |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | — | tracing |

`llm_is_mock`, `tavily_is_mock`, `email_is_mock` are derived: a service mocks when
`USE_MOCKS` is on **or** its key is absent.

---

## 15. Observability

- **Audit trail.** `@audited` appends one `{node, verdict, elapsed_ms}` row per
  node into `audit_trail`; the CLI prints it as a per-candidate trace.
- **Logging.** `logging_config.py` configures one root logger (level via
  `LOG_LEVEL`). Node detail is `DEBUG`; decisions and service calls are `INFO`
  (`SENIORITY …`, `PLAN …`, `SCORECARD …`, `DECISION …`, `tavily search: …`,
  `github lookup: …`); failures log concisely (traceback only at `DEBUG`).
- **LangSmith.** With `LANGSMITH_TRACING=true`, every run is traced and node names
  appear in the trace tree.

---

## 16. Scoring math

`aggregate_scores` groups `completed_scores` by `kind`, averages each kind, and
blends them:

```
final_score = 0.35·mean(skill) + 0.30·experience + 0.10·education
            + 0.15·signal + 0.10·research
```

The bar is **seniority-specific** (`_BARS`) — the seniority the classifier returned
selects the `(advance, borderline)` thresholds:

| Seniority | advance ≥ | borderline ≥ | else |
|---|---|---|---|
| junior | 60 | 45 | reject |
| mid | 70 | 45 | reject |
| senior | 74 | 50 | reject |
| executive | 82 | 62 | reject |

The borderline band is intentionally wide so an under-leveled-but-competent
candidate lands in human review rather than an automatic reject. Expected verdicts
vs `jd_senior_backend`: Priya → senior / advance / no pause; Eitan → mid /
borderline / **pauses**; Mira → senior(frontend) / reject / no pause.

---

## 17. A full trace (Eitan)

1. `ingest_resume_and_jd` reads `resume_eitan.md`; the parse→normalize→extract
   chain yields "Eitan Bergmann", skills, ~2 years.
2. `classify_seniority` → `{seniority: "mid", role_family: "backend"}`.
3. `plan_required_skills` → e.g. `[Python, PostgreSQL, Kafka, Redis, REST]`.
4. Fan-out: 5 `skill_worker` Sends **plus** experience/education/signal/research,
   all in one superstep; each appends to `completed_scores`.
5. `aggregate_scores` blends them → `final_score ≈ 56`; mid bar `(70, 45)` →
   `45 ≤ 56 < 70` → **borderline**.
6. `route_recommendation` → `human_review`.
7. `human_review` calls `interrupt(...)`; the run **pauses**. The CLI/UI resumes
   with `Command(resume={"approved": True})`.
8. Approved → `draft_email` → `critic_loop` (revise up to 3) → `send_email_update_ats`.
9. Email sent + ATS written → `finalize` (`terminal_status="completed"`) → `END`.

---

## 18. Runtime surfaces (CLI, API, UI)

The same compiled graph is exposed three ways:

- **`main.py` / `cli.py`** — compiles the graph, writes `graph_out/graph.png`, runs
  the canned scenarios (3 `.md` + 1 `.pdf`), auto-approves at the borderline
  interrupt, and prints a per-node trace plus a scoreboard.
- **`scripts/test_resume.py`** — runs one résumé (`.md` or `.pdf`) against a JD and
  prints the verdict; in real mode the research tool calls show as
  `tavily search: …` / `github lookup: …` log lines.
- **`api/server.py`** — FastAPI: `POST /api/run` starts a thread, `POST
  /api/resume/{thread_id}` resumes a paused borderline run, `GET /api/graph.png`
  serves the diagram; the React/Vite/Tailwind UI in `ui/` drives it.

---

## 19. Testing

`tests/` (run with `uv run pytest -q`) covers the three required categories with
the LLM and services mocked (`conftest.py` forces `HIREGRAPH_USE_MOCKS=true`):

- **State** — build a `HireGraphState`; exercise the `completed_scores` reducer.
- **Node** — `classify_seniority` returns a `Command` routing to
  `plan_required_skills`; `aggregate_scores` blends `completed_scores` into a
  `final_score` + recommendation.
- **End-to-end** — the borderline candidate pauses at the interrupt, resumes by
  `thread_id`, and reaches a clean terminal state; the reject candidate sends a
  rejection email and updates the ATS.

---

## 20. Notes and limitations

- **The saga path is wired but not triggered by default.** `compensate` is only
  reachable when `send_email_update_ats` fails after its retries, which today only
  happens if `state["force_ats_failure"]` is `True` (the `log_to_ats(force_failure=…)`
  hook). No scenario sets it, so the rollback branch — correctly wired — isn't hit
  in a normal run. Set `force_ats_failure=True` on the input to demo it.
- **`scorecard_summary` is write-only** — set in `aggregate_scores` for
  readability, not yet read anywhere (a candidate for the UI or removal).
- **The recoverable loopback lives in the research agent.** The intake chain is
  linear; the "tool error becomes data the agent reasons about" behavior comes
  from `handle_tool_errors=True`.
- **Determinism.** With mocks on, outputs are fully deterministic and reproduce
  `expected_outcomes.md`; with real models the exact scores vary but the
  advance / borderline / reject verdicts are stable.

---

*The Python source and tests are authoritative. This document describes the system
as built; when in doubt, read the code in `src/hiregraph/`.*
