# Runtime outcomes

These are the actual results from running `uv run python main.py` with the real
OpenAI LLM, Tavily, GitHub API, Redis checkpointer, and Mailtrap configuration.

| Resume | JD | Expected seniority classification | Expected recommendation | Should pause for human review? |
| :--- | :--- | :--- | :--- | :---: |
| `resume_priya.md` | `jd_senior_backend.md` | senior | advance | no |
| `resume_eitan.md` | `jd_senior_backend.md` | junior | borderline | yes |
| `resume_mira.md` | `jd_senior_backend.md` | senior (frontend) | reject | no |
| `Shivani_Resume.pdf` | `jd_senior_backend.md` | mid (data) | borderline | yes |

The runtime signals show Priya as the strong senior backend candidate, Eitan as
the borderline candidate that triggers the `interrupt()` path, Mira as the
frontend skill mismatch that exercises rejection and rejection-email delivery,
and Shivani as the PDF intake candidate that also triggers human review.

