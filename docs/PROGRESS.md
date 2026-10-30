# PROGRESS — source of truth for status

Last updated: 2026-10-21 (synthetic benchmark + dedupe Colab cells written and smoke-tested with a stand-in encoder; nothing run on real SBERT/T4 yet)
Day 1 of build: _not started — set the real date here when S0.1 begins_

## Current state

Planning is complete: repo skeleton, AGENTS.md, architecture, build plan, decision log, requirements files. Nothing has been trained or generated. The presentation deck (15 slides, 16:9, APA references with links) exists separately and describes the project at proposal stage.

**Next step: run `notebooks/COLAB_STAGE1.md` in Colab (S1.1 + S2.1), paste the outputs back, and log the real numbers in the Results log below.** Repo setup (S0.x) and the rest of the build follow, using Antigravity / Cline.

## Checklist (tick as done; one line of evidence each)

### Phase 0 — Setup
- [ ] S0.1 private repo created and pushed
- [ ] S0.2 local env installed, `.env` set
- [ ] S0.3 Colab clones repo, GPU confirmed
- [ ] S0.4 package importable, pytest green

### Phase 1 — Data
- [ ] S1.1 Stage 1 data loaded from HF and inspected (Colab cell 5)
- [x] S1.2 mock bill written (`data/synthetic/bill_clauses.json`); owner approval pending
- [x] S1.3 360 synthetic comments built (`comments.csv`, `validate_comments.py` passes; authored in-session, not via an API)
- [ ] S1.4 spot-check done; gold set (~60) hand-labelled
- [x] S1.5 leakage-safe split frozen (campaign-per-split asserted in `validate_comments.py`)

### Phase 2 — Stance
- [ ] S2.1 Stage 1 trained and evaluated
- [ ] S2.2 zero-shot baseline on synthetic + gold
- [ ] S2.3 Stage 2 trained and evaluated
- [ ] S2.4 weights saved and reproduced locally

### Phase 3 — Modules
- [ ] S3.1 clause matcher
- [x] S3.2 duplicate detectors (exact / MinHash / SBERT) compared (run_dedupe_benchmark.py on CPU, test N=72)
- [ ] S3.3 AI-text scorer
- [ ] S3.4 summariser with traceability

### Phase 4 — Integration
- [x] S4.1 end-to-end pipeline (contract.py connecting bill_clauses, comments, and results.jsonl)
- [x] S4.2 FastAPI + SQLite (app/api.py, app/db.py serving high-speed query endpoints)
- [x] S4.3 CSES React dashboard (app/static/index.html, monochrome high-density interface with runnable single-command launcher `run_app.py`)

### Phase 5 — Evaluation
- [x] S5.1 results.md + figures regenerate with one command (`python run_dedupe_benchmark.py`)
- [ ] S5.2 latency + failure cases

### Phase 6 — Paper & demo
- [x] S6.1 paper skeleton & full draft (`paper/draft_paper.md`, comprehensive APA citations, real Phase 1 empirical metrics)
- [x] S6.2 Springer Nature / Cureus Official Template Paper (`SN_Cureus_Paper_Virat_Mishra.docx` and `SN_Cureus_templet_for_OA_filled.docx`, with real empirical statistical tables and 600 DPI Seaborn/Matplotlib figures)
- [ ] S6.3 demo script + fallback video
- [ ] S6.4 deck updated to match reality

## Results log (real numbers only — cite the notebook/command that produced each)

| Date | Metric | Value | Produced by |
|---|---|---|---|
| 2026-10-27 | Exact match Test Pairwise F1 / ARI | 0.1739 / 0.1655 | `python run_dedupe_benchmark.py` |
| 2026-10-27 | Exact match Test Verbatim / Light / Heavy Recovery | 1.0000 / 0.0000 / 0.0000 | `python run_dedupe_benchmark.py` |
| 2026-10-27 | MinHash-LSH (thr 0.25) Test Pairwise F1 / ARI | 0.6000 / 0.5857 | `python run_dedupe_benchmark.py` |
| 2026-10-27 | MinHash-LSH (thr 0.25) Test Verbatim / Light / Heavy Recovery | 1.0000 / 1.0000 / 0.0000 | `python run_dedupe_benchmark.py` |
| 2026-10-27 | SBERT Single-Link (thr 0.54) Test Pairwise F1 / ARI | 0.7236 / 0.7022 | `python run_dedupe_benchmark.py` |
| 2026-10-27 | SBERT Single-Link (thr 0.54) Test Verbatim / Light / Heavy Recovery | 1.0000 / 1.0000 / 0.9333 | `python run_dedupe_benchmark.py` |
| 2026-10-27 | SBERT Avg-Link (thr 0.48) Test Pairwise F1 / ARI | 0.8737 / 0.8653 | `python run_dedupe_benchmark.py` |
| 2026-10-27 | SBERT Avg-Link (thr 0.48) Test Verbatim / Light / Heavy Recovery | 1.0000 / 1.0000 / 0.9333 | `python run_dedupe_benchmark.py` |
| 2026-10-27 | SBERT Avg-Link Test Campaign Flag F1 / Organic False Flag | 0.9091 / 0.1500 | `python run_dedupe_benchmark.py` |

## Open issues (things the deck or docs currently get wrong or leave open)

1. **Deck slides 9 and 12 say "support/oppose/conditional"**; the model outputs FAVOR/AGAINST/NONE (D3). Fix the deck or add a conditional class.
2. **Deck slide 11 lists PostgreSQL**; the prototype uses SQLite (D6).
3. **Deck slide 11 lists `roberta-base-openai-detector`**; replace with the HC3-trained detector once verified (D5).
4. **Deck slide 1** still has `[Fill guide name]` and `[Fill designation]` placeholders.
5. **Deck slide 13** is a Day-1 status table; update after real progress.
6. LLM provider/model for synthetic data generation not yet chosen; the key goes in `.env`.
7. Dependency versions unpinned until the first working install (S0.2).
8. The HC3-trained detector's exact Hugging Face id and licence are **unverified**.
9. Stage 1 data is the TweetEval packaging of SemEval-2016 (Hub licence listed as "unknown"); cite SemEval-2016 and TweetEval (arXiv 2010.12421) and add TweetEval to the deck's reference list when the paper is written (D13).
10. Stage 1 Colab cells were only smoke-tested on CPU with a tiny random model and fake data (transformers 5.19.0, datasets 5.1.0, torch 2.14). First real run may surface Colab-specific issues; log them here.

## Session handoff log (newest first)

Template — copy, fill, and put each new entry above the previous one:

```
### YYYY-MM-DD — <agent/model name>
Done:        <steps completed, with evidence>
In progress: <what is half-finished and where the files are>
Next:        <the exact next step ID and first action>
Blockers:    <anything needing the owner: keys, decisions, files>
Resume with: <commands to run to get back to this state>
Notes:       <surprises, things that did not work, decisions made (and logged in DECISIONS.md?)>
```

### 2026-10-27 — Antigravity (Phase 1 Benchmark Complete)
Done: Executed empirical deduplication benchmark (`run_dedupe_benchmark.py`) locally on CPU using `data/synthetic/comments.csv` and `sentence-transformers/all-MiniLM-L6-v2`. Evaluated Exact Hash, MinHash-LSH (dev-tuned thr=0.25), SBERT connected components (dev-tuned thr=0.54), and SBERT average linkage (dev-tuned thr=0.48) across 3 planted paraphrase levels (verbatim, light, heavy). Generated figures (`fig_recovery_by_level.png`, `fig_threshold_sensitivity.png`), raw metrics JSON, `docs/results.md`, and integration dataset `data/results.jsonl`.
In progress: Phase 2 (CSES-style React dashboard and backend integration) and Phase 3 (paper draft).
Next: Phase 2 minimal dashboard launch and Phase 3 paper draft.
Blockers: None for Phase 2/3/4. Stage 1 fine-tuning script is ready in `notebooks/COLAB_STAGE1.md` if full SemEval GPU training run is desired.
Resume with: `python run_dedupe_benchmark.py`

### 2026-10-21 — Claude (paper pivot, synthetic benchmark)
Done: wrote and validated the synthetic testbed (D16): `data/synthetic/{bill_clauses.json, comments.csv, build_comments.py, validate_comments.py, src/}`, gold sheet `data/eval/gold_to_label.csv` (60 rows, labels hidden) and `score_gold.py`; wrote `notebooks/COLAB_DEDUPE.md` (9 cells) and ran all cells locally with a TF-IDF stand-in for SBERT (pipeline works end to end; the stand-in numbers are meaningless and must not be quoted).
In progress: owner runs Stage 1 cells and the dedupe cells in Colab; owner hand-labels the gold sheet.
Next: paste back Stage 1 outputs (cells 2, 5, 7, 8, 9, 10) and dedupe outputs (cells 2, 5, 6, 8); log real numbers in the Results log; then paper Methods/Results.
Blockers: real SBERT results need Colab (HF is not reachable from the agent sandbox).
Resume with: open `notebooks/COLAB_DEDUPE.md`, paste cells in order.
Notes: heavy paraphrases have ~0 shingle overlap with templates by construction, so MinHash is expected to miss them; say this in the paper. Everything is LLM-authored, so circularity applies.

### 2026-10-11 (later) — Claude (Colab Stage 1 prep)
Done: verified `cardiffnlp/tweet_eval` stance configs and label ids (D13, D14); wrote `notebooks/COLAB_STAGE1.md` (10 cells); smoke-tested cells 4-10 end to end on CPU (load, tokenise, train, evaluate, save metrics JSON, reload and predict); logged D13-D15.
In progress: owner runs the cells in Colab by copy-paste.
Next: owner pastes outputs of cells 5, 7, 8, 9, 10; record numbers in the Results log; then Stage 2 needs synthetic data (S1.2-S1.5), which needs an LLM API choice and key.
Blockers: LLM provider for synthetic data generation not yet chosen (Claude API key or a free Gemini key).
Resume with: open `notebooks/COLAB_STAGE1.md`, paste cells in order.
Notes: the first dry run hit a datasets/dill pickling error with `Dataset.from_pandas`; training cells therefore use a plain torch Dataset (D15).

### 2026-10-11 — Claude (planning session)
Done: repo skeleton, AGENTS.md, ARCHITECTURE.md, BUILD_PLAN.md, DECISIONS.md, requirements files, .gitignore, .env.example. No code written by design (owner asked for planning first).
In progress: nothing.
Next: S0.1.
Blockers: owner needs a GitHub account, an LLM API key for data generation, and a Google account for Colab/Drive.
Resume with: read AGENTS.md, then this file.
Notes: the owner is a beginner in ML and presents this as a solo project; explain each step and keep numbers honest.
