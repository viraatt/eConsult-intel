# BUILD PLAN

Ordered steps. Each has an **environment** (Local = VS Code, Colab = free T4), an **output**, and a **done-when** check. Take one step at a time, tick it in `docs/PROGRESS.md`, and explain it to the owner before moving on. Days are relative (Day 1 = the day building starts; record the real date in PROGRESS.md).

**Order change (D15): Stage 1 training (S2.1) is done FIRST, by copy-paste Colab cells in `notebooks/COLAB_STAGE1.md`, before the repo/app work. Stage 1 data is loaded from Hugging Face (D13), so S1.1 needs no download.** After training, the remaining steps are built with Antigravity / Cline, which must read `AGENTS.md` first.

Critical path: S1 data → S2 stance → S4 integration → S5 evaluation. S3 modules can be built in parallel with S2 training waits.

---

## Phase 0 — Setup (Day 1)

| ID | Env | Step | Done when |
|---|---|---|---|
| S0.1 | Local | Create a **private GitHub repo**, push this skeleton. This repo is the sync point between Colab, VS Code and any new agent. | `git clone` of the repo works from a fresh folder |
| S0.2 | Local | Python 3.10+ virtualenv; `pip install -r requirements-local.txt`; copy `.env.example` → `.env` and add the LLM API key. Pin working versions into the requirements file. | `python -c "import torch, transformers, sentence_transformers, datasketch, fastapi, streamlit"` succeeds |
| S0.3 | Colab | New notebook: mount Drive, `git clone` the repo (use a GitHub token via Colab Secrets for the private repo), `pip install -r requirements-colab.txt`, confirm GPU. | `torch.cuda.is_available()` is True and the repo imports |
| S0.4 | Local | Make the shared package importable (`src/econsult/__init__.py`, `pip install -e .` or `PYTHONPATH`). Add one trivial passing test. | `pytest` runs green |

## Phase 1 — Data (Day 1–3)

| ID | Env | Step | Done when |
|---|---|---|---|
| S1.1 | Colab | Stage 1 data is loaded from Hugging Face `cardiffnlp/tweet_eval` (stance_* configs) inside the Colab cells (D13); nothing to download. Inspect sizes and label counts (Colab cell 5). | printed sizes match 2,620 / 294 / 1,249 and label counts are recorded in PROGRESS.md |
| S1.2 | DONE 2026-10-21 (see D16) | Write the **mock bill**: "Companies (Amendment) Bill, 2026 (fictional)" with 6 clauses with distinct topics (e.g. penalties, disclosure, timelines, compliance burden on small firms). Save `data/synthetic/bill_clauses.json`. The owner reads and approves it. | JSON validates against the contract |
| S1.3 | Local | Write the **generation script** and generate ~400–600 comments with planted structure: genuine varied FAVOR/AGAINST/NONE per clause; several **campaign families** (one template reworded at different paraphrase levels: verbatim, light, heavy); a batch in "bulk AI style" flagged `is_machine_generated`; varied stakeholder voices. Record the prompt used. | `comments.csv` matches the contract; counts per clause/stance/campaign printed |
| S1.4 | Local | **Validate**: owner spot-checks ~10% of rows; fix systematic problems (e.g. every comment opening the same way). Owner **hand-labels ~60 fresh comments** into `data/eval/gold.csv`. | gold set exists; label agreement with the generator's labels computed and recorded (a real, honest number) |
| S1.5 | Local | **Split** train/dev/test by comment, keeping each campaign family inside one split (prevents leakage). Freeze split in the `split` column. | no campaign_id appears in more than one split (assert in a test) |

## Phase 2 — Stance model (Day 3–6)

| ID | Env | Step | Done when |
|---|---|---|---|
| S2.1 | Colab | **Stage 1**: fine-tune `bert-base-uncased` on the TweetEval/SemEval train split, evaluate on its held-out test split. Run the copy-paste cells in `notebooks/COLAB_STAGE1.md`. | macro-F1, accuracy, SemEval F_avg and the majority-class baseline saved to Drive as `stance_stage1_metrics.json` and logged in PROGRESS.md |
| S2.2 | Colab | Evaluate the Stage 1 model **zero-shot on the synthetic test + gold** (no domain adaptation). | baseline numbers saved and logged |
| S2.3 | Colab | **Stage 2**: continue training on the synthetic train split with a lower learning rate (`notebooks/02_stance_stage2.ipynb`); evaluate on synthetic test + gold. | numbers saved; improvement (or not) over S2.2 recorded honestly |
| S2.4 | Colab → Local | Save weights to Drive and a private HF Hub repo; pull into local `models/stance_final/`. | local `stance.py` inference reproduces the Colab prediction on 5 sample comments |

Note: if Stage 2 does not beat Stage 1 on gold, that is a finding to report, not something to hide or tune away.

## Phase 3 — Remaining modules (Day 4–9, parallel to Phase 2 waits)

| ID | Env | Step | Done when |
|---|---|---|---|
| S3.1 | Local | `clauses.py`: embed clauses, match comments, UNMATCHED threshold tuned on dev. Tests + eval script. | top-1 clause accuracy on synthetic test and gold recorded |
| S3.2 | Colab (benchmark: `notebooks/COLAB_DEDUPE.md`) then Local | `dedupe.py`: exact, MinHash-LSH, SBERT-cluster detectors with a common interface. Eval against planted campaigns. | table of precision/recall/ARI for the three detectors on test, with paraphrase level broken out |
| S3.3 | Local | `aigen.py`: locate and verify the HC3-trained detector model id; run on synthetic machine-generated vs human-style comments. | accuracy/AUC recorded; failure cases listed |
| S3.4 | Local | `summarize.py`: group by (clause, stance), dedup-collapse, chunk, summarise, return source ids; extractive fallback. | every summary has non-empty `source_comment_ids`; 3 summaries read by the owner and judged sensible |

## Phase 4 — Integration (Day 9–11)

| ID | Env | Step | Done when |
|---|---|---|---|
| S4.1 | Local | `pipeline.py`: CSV + clauses in → `results.jsonl` + `summaries.json` out, one command. | `python -m econsult.pipeline ...` runs end-to-end on the test split |
| S4.2 | Local | FastAPI app per ARCHITECTURE §3 with SQLite storage. | endpoints return correct data in a test using the test client |
| S4.3 | Local | Streamlit dashboard with the five panels, labelled "synthetic testbed". Use the `frontend-design` and `dataviz` guidance if the agent has them; keep it minimal and readable. | owner can click through every panel; screenshots taken for the deck |

## Phase 5 — Evaluation & figures (Day 11–13)

| ID | Env | Step | Done when |
|---|---|---|---|
| S5.1 | Local | One script reruns all evaluations and writes `docs/results.md` + figures into `paper/figures/` (confusion matrix, exact-vs-MinHash-vs-SBERT bar chart, Stage1-vs-Stage2 bar chart). | results.md regenerates from scratch with one command |
| S5.2 | Local | Latency measurement + failure-case gallery (3 wrong stances, 3 missed campaigns). | included in results.md |

## Phase 6 — Paper & demo (Day 12–14)

| ID | Env | Step | Done when |
|---|---|---|---|
| S6.1 | — | Paper skeleton in `paper/` (Abstract, Intro, Related Work, Method, Experimental Setup, Results, Limitations, Conclusion). Owner writes prose; the agent helps structure, checks facts against results.md, and flags close paraphrasing of sources. | every number in the draft traces to results.md |
| S6.2 | Local | Demo script + a screen-recorded fallback video of the dashboard. | demo runs offline end-to-end |
| S6.3 | — | Update deck: slide 13 status table, replace "planned" with real results, fix the open deck issues listed in PROGRESS.md. | deck matches reality |

---

## Paper-writing discipline (applies from S6.1)

- Read the source, close it, write from memory, then check. Synonym-swapping is not paraphrasing.
- Cite every non-obvious claim; keep direct quotes rare and short.
- LLMs may brainstorm structure, explain concepts and check grammar; the sentences are the owner's.
- Self-check: search 2–3 sentence chunks of the draft in quotes; run the college's similarity checker before submitting.
- Be explicit about synthetic data and the limitations in ARCHITECTURE §7.

## If time runs short — cut in this order (last item cut first)

1. Hindi/multilingual (already "planned")
2. AI-text detector polish (keep it as a labelled weak signal)
3. Abstractive summariser quality (fall back to extractive + sources)
4. Dashboard polish (keep the five panels, drop styling)
5. **Never cut:** gold set, Stage 1 vs Stage 2 comparison, exact-vs-MinHash-vs-SBERT comparison, honest results log.
