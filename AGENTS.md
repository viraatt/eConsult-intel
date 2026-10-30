# AGENTS.md — read this first

This file is the entry point for any AI coding agent (Claude, Codex, Gemini, Cursor, Copilot, etc.) picking up this project. The owner switches agents when credits run out, so **everything you need to continue is in this repo, not in anyone's chat history.**

## 1. What this project is (30 seconds)

**eConsultation Intelligence** — an NLP pipeline that analyses public comments on draft legislation (modelled on the Ministry of Corporate Affairs eConsultation module, SIH problem statement 25035). It goes beyond whole-comment sentiment with four modules:

| Module | Job |
|---|---|
| M1 Clause-Matcher | Link each comment to the bill clause it is about (SBERT embeddings + cosine similarity) |
| M2 Stance Detector | FAVOR / AGAINST / NONE toward that clause (fine-tuned BERT, two-stage training) |
| M3 Campaign Flagger | Flag near-duplicate / templated comments (exact vs MinHash vs SBERT) and score likely AI-written text |
| M4 Summariser + Dashboard | Clause-linked, traceable summaries on a Streamlit dashboard, behind a FastAPI backend |

It is a **solo B.Tech CSE final-year project** (owner: Virat Mishra, MPGI Kanpur), to be shown at a project review/expo and written up as a conference paper. The owner is **a beginner in ML**: explain what you do and why, in plain language, and keep each step small enough to understand.

## 2. Read order for a new agent

1. This file
2. `docs/PROGRESS.md` — what is done, what is in flight, what is next, open issues. **Source of truth for status.**
3. `docs/ARCHITECTURE.md` — system design, data contracts, Colab/local split, evaluation design
4. `docs/BUILD_PLAN.md` — the ordered steps, each with environment and a "done when" test
5. `docs/DECISIONS.md` — why things are the way they are (do not silently reverse these)

## 2b. Tooling and workflow (current)

- Training happens first, in **Google Colab by copy-paste cells** (`notebooks/COLAB_STAGE1.md`, later a Stage 2 equivalent). The owner pastes cells and reports outputs; do not assume a notebook can be executed for them.
- The rest of the build is done in **Antigravity** and **Cline** (and possibly other agents). Whatever the tool, read this file and `docs/PROGRESS.md` before touching anything. If the tool has a custom-instructions or rules field, it should say: "Read AGENTS.md first."

## 3. Working rules

1. **One step at a time.** Take the next unchecked step in `docs/BUILD_PLAN.md`, finish it, verify it, then stop and report. Do not run ahead into later phases.
2. **Explain as you go.** The owner must be able to defend every choice to a panel. After each step give: what you built, why, how to check it, what it showed.
3. **Never fabricate results.** No invented metrics, no placeholder numbers presented as real, no "should achieve ~X%". If a number is not from a run you can point to, it does not go in a doc, slide or paper. Record real numbers in `docs/PROGRESS.md` → *Results log* with the command/notebook that produced them.
4. **Be honest about the data.** The comments are **synthetic** (real MCA comments are confidential and not public). Always label them as synthetic in code, docs, figures and the paper. Never describe results as "on real MCA data".
5. **Guard against circularity.** Synthetic data is LLM-written and LLM-labelled, so scores on it are optimistic. Always also report a non-circular number (SemEval-2016 held-out test) and the owner's hand-labelled gold set (`data/eval/gold.csv`).
6. **Do not reverse a logged decision** (`docs/DECISIONS.md`) without adding a new entry that explains why.
7. **Verify citations.** Only cite papers whose existence and details you have checked. One earlier citation in this project turned out to be unverifiable and was removed — do not repeat that.
8. **Secrets.** API keys live in `.env` (local) or Colab Secrets. Never commit them, never print them, never paste them in notebooks. `.env.example` lists the names.
9. **Keep the paper's integrity.** Write paper text only when the owner asks, in the owner's voice, paraphrasing sources (see paper section in `docs/BUILD_PLAN.md`). Do not copy text from sources.
10. **Update `docs/PROGRESS.md` at the end of every working session** (template at the bottom of that file) so the next agent can resume cold.

## 4. Where code runs

| Runs on **Google Colab (free T4 GPU)** | Runs **locally in VS Code (CPU is enough)** |
|---|---|
| Stage 1 fine-tune on SemEval-2016 | Repo, environment, tests, git |
| Stage 2 domain adaptation on synthetic data | Synthetic data generation (LLM API calls) |
| Anything needing a GPU for training | Clause matcher, dedupe, AI-text detector, summariser (inference only) |
| Saving trained weights to Drive / HF Hub | FastAPI backend, Streamlit dashboard |
| | Evaluation scripts, figures, paper |

Colab is for **training only**. The demo app never runs on Colab. Code that lives in `src/econsult/` is shared: notebooks `git clone` the repo and import from it, so there is one copy of the logic. Trained weights move Colab → Drive/HF Hub → local `models/` (gitignored).

## 5. Repo map

```
AGENTS.md                this file
CLAUDE.md                one-line pointer to this file
.env.example             names of required secrets (no values)
.gitignore
requirements-local.txt   VS Code / app dependencies
requirements-colab.txt   training dependencies
docs/                    PROGRESS, ARCHITECTURE, BUILD_PLAN, DECISIONS
src/econsult/            shared library (importable from notebooks and app)
notebooks/               Colab notebooks (training + eval that needs GPU)
app/                     FastAPI backend + Streamlit dashboard
data/raw/                downloaded public data (SemEval) — gitignored
data/synthetic/          bill + synthetic comments (small; committed)
data/eval/               gold set + frozen splits (committed)
models/                  trained weights — gitignored
tests/                   pytest
paper/                   paper drafts, figures, bib
```

Status of the skeleton: folders exist; **no application code has been written yet** (planning phase).

## 6. Claims the deck makes — the build must make them true (or the deck must be changed)

The presentation deck already tells the panel the following. Treat them as requirements, or flag the mismatch in `docs/PROGRESS.md` → *Open issues*:

- Clause-level stance detection (not whole-comment sentiment)
- Semantic duplicate/campaign detection that catches *paraphrased* campaigns, compared against exact-match
- LLM-generated-text flagging
- Clause-linked summaries traceable to source comments
- Runs on free-tier infrastructure, no special hardware
- Hindi/regional support is **"planned"**, not delivered — keep it off the critical path
- Honest status: the build starts after the proposal; nothing is claimed as done until it is

## 7. Definition of done for the whole project

A reviewer can: open the dashboard on the synthetic testbed, see per-clause stance breakdowns, click into a clause and see comments with stance tags, see a detected campaign cluster, read a clause summary with links back to source comments, and see a results table with real numbers from `docs/results.md` — all reproducible from the commands in `docs/BUILD_PLAN.md`.
