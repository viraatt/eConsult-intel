# ARCHITECTURE & SYSTEM DESIGN

## 1. Scope

Input: a draft bill (list of clauses) and a batch of public comments.
Output: per-comment annotations (clause, stance, campaign flag, AI-text score) and per-clause summaries with links back to source comments, shown on a dashboard.

Out of scope for the prototype: live ingestion from the real MCA21 portal, authentication, multi-tenant deployment, real-time streaming. Hindi/regional-language support is a **stretch goal** (IndicTrans2 in front of the English pipeline) and is off the critical path.

## 2. Pipeline

```
 bill_clauses.json ─────────────────────────────┐
                                                ▼
 comments.csv ──► [Ingest/validate] ──► M1 Clause-Matcher ──► M2 Stance Detector ─┐
                        │                 (SBERT cosine)        (fine-tuned BERT)  │
                        │                                                          ▼
                        ├──► M3a Duplicate/Campaign Flagger ─────────────────► results.jsonl
                        │      (exact | MinHash | SBERT cluster)                   │
                        └──► M3b AI-text scorer (HC3-trained detector)             ▼
                                                                   M4 Summariser (per clause × stance)
                                                                                   │
                                                                                   ▼
                                                         FastAPI  ◄────────►  Streamlit dashboard
```

M2 and M3 are independent of each other and may run in either order; both need M1's clause assignment only if grouping by clause (M3a clusters across all comments, then reports cluster size per clause).

## 3. Components

### M1 Clause-Matcher (`src/econsult/clauses.py`)
- Model: `sentence-transformers/all-MiniLM-L6-v2` (English). Multilingual variant `paraphrase-multilingual-MiniLM-L12-v2` is a stretch option.
- Method: embed every clause once; embed each comment; assign the clause with the highest cosine similarity. If the best similarity is below a threshold, assign `clause_id = "UNMATCHED"` (general comment).
- Threshold: starting guess 0.30, **to be tuned on the dev split**, not hard-coded from belief.
- Runs on CPU.

### M2 Stance Detector (`src/econsult/stance.py`; training in notebooks)
- Model: `bert-base-uncased` (swap to `roberta-base` only via a logged decision).
- Labels: **FAVOR / AGAINST / NONE** (identical to SemEval-2016 Task 6, see DECISIONS D3). Label ids: **0 NONE, 1 AGAINST, 2 FAVOR** (D14).
- Input format: a sentence **pair** `tokenizer(clause_text, comment_text)` (target first), i.e. `[CLS] clause [SEP] comment [SEP]`. Never a hand-built string.
- Training: Stage 1 on SemEval-2016 Task 6A via the TweetEval packaging (general skill, D13) → Stage 2 on synthetic eConsultation train split (domain adaptation). Stage 1 is run first, in Colab (`notebooks/COLAB_STAGE1.md`).
- Reported numbers: macro-F1 for (a) Stage 1 on SemEval test, (b) Stage 1 zero-shot on synthetic test, (c) Stage 2 on synthetic test, (d) Stage 2 on the owner's gold set. The gap between (b) and (c) is the domain-adaptation ablation.
- Needs a GPU to train (Colab); inference runs on CPU.

### M3a Duplicate / Campaign Flagger (`src/econsult/dedupe.py`)
Three detectors, implemented side by side so the paper can compare them:
1. **Exact**: normalised-text hash match (the baseline the Pew/FCC-style analysis effectively uses).
2. **MinHash-LSH** (`datasketch`): lexical near-duplicates (shingled tokens).
3. **SBERT semantic**: embeddings + cosine threshold + clustering (agglomerative or DBSCAN) to catch *paraphrased* templates.
- Output per comment: `dup_cluster_id`, `dup_cluster_size`, `is_campaign` (cluster size ≥ configurable minimum, default 3).
- Thresholds tuned on dev split. Prototype scale (≤ a few thousand comments) allows plain pairwise cosine; MinHash is the scale-out story.

### M3b AI-generated-text scorer (`src/econsult/aigen.py`)
- Candidate model: a RoBERTa classifier trained on the HC3 corpus (Human-ChatGPT Comparison Corpus). **Verify the exact Hugging Face model id exists and its licence before use** (see DECISIONS D5 — an earlier note named a GPT-2-era detector, which is the wrong target for modern LLM text).
- Output: `llm_prob` in [0,1]. Treated as a **weak signal / flag for human review**, never an accusation. Expect modest accuracy; report it honestly, including failure cases.

### M4 Summariser (`src/econsult/summarize.py`)
- Model: `facebook/bart-large-cnn` (inference only; CPU is workable for demo-sized batches, GPU optional).
- Method: group comments by (clause, stance); collapse each campaign cluster to one representative with a count; chunk to the model's input limit; summarise; **always return the list of source `comment_id`s** so every summary sentence set is traceable.
- Known weakness: BART-CNN is news-tuned and can paraphrase loosely or hallucinate. Mitigation: show source comments beside every summary; keep an extractive fallback (top-k most central comments by embedding centrality).

### API + Dashboard (`app/`)
- FastAPI (`app/api.py`): `POST /ingest` (comments, optional bill), `GET /clauses`, `GET /clauses/{id}/comments`, `GET /clauses/{id}/summary`, `GET /campaigns`, `GET /stats`. A one-way adapter: an ERP can POST comments in and read results out without any change to its internals (this is the "ERP integration" claim; it is a design property, not a built integration).
- Storage: SQLite file for the prototype (see DECISIONS D6). PostgreSQL is the production note on the slides.
- Streamlit (`app/dashboard.py`) panels: (1) clause overview strip with stance mini-bars; (2) clause drill-down with comment list, stance tags, campaign flags; (3) campaign panel (e.g. "N near-identical comments, similarity X"); (4) summary panel with source links; (5) top-line stats bar. The dashboard must be labelled as running on the **synthetic testbed**.

## 4. Data contracts

### `data/synthetic/bill_clauses.json`
```json
[{"clause_id": "C1", "title": "string", "text": "string"}]
```

### `data/synthetic/comments.csv`  (synthetic, ground truth included)
| column | type | meaning |
|---|---|---|
| comment_id | str | unique, e.g. `cm_0001` |
| text | str | comment body |
| language | str | `en` (others stretch) |
| stakeholder_type | str | `citizen` / `industry` / `legal` / `academic` |
| true_clause_id | str | ground-truth clause (or `GENERAL`) |
| true_stance | str | `FAVOR` / `AGAINST` / `NONE` |
| campaign_id | str/empty | empty for organic comments; `K1`..`K8` for planted campaign families |
| paraphrase_level | str | `organic` / `verbatim` / `light` / `heavy` (how a campaign member relates to its template) |
| is_machine_generated | bool | generated in "bulk AI style" for the detector test |
| split | str | `train` / `dev` / `test` (campaign families stay within one split to avoid leakage) |

### `data/eval/gold.csv`
`comment_id, clause_id, text, gold_stance, generator_stance`: 60 organic test-split comments **hand-labelled by the owner** (sheet: `gold_to_label.csv`, scorer: `score_gold.py`), never used for training.

### `results.jsonl` (pipeline output, one object per comment)
```json
{"comment_id": "cm_0001", "clause_id": "C3", "clause_sim": 0.61,
 "stance": "AGAINST", "stance_conf": 0.88,
 "dup_cluster_id": "k7", "dup_cluster_size": 14, "is_campaign": true,
 "llm_prob": 0.12}
```

### `summaries.json`
```json
[{"clause_id": "C3", "stance": "AGAINST", "summary": "string",
  "source_comment_ids": ["cm_0001", "cm_0044"], "n_comments": 31, "n_after_dedup": 12}]
```

Schemas are enforced in `src/econsult/schemas.py` (pydantic or dataclasses) once code begins.

## 5. Environments and artefact flow

| Concern | Colab | Local (VS Code) |
|---|---|---|
| Code source | `git clone` of the repo at start of each notebook | working copy |
| Training | yes (T4 GPU) | no |
| Inference / app | no | yes |
| Secrets | Colab Secrets | `.env` |
| Data in | clone gives `data/synthetic`, `data/eval`; SemEval CSV from Drive | `data/raw` downloaded by hand |
| Weights out | save to Drive **and** push to private HF Hub repo | pull from HF Hub / Drive into `models/` |

Colab sessions are ephemeral: every notebook must be re-runnable from a clean runtime and must write its outputs (weights, metrics JSON) to Drive before ending.

## 6. Evaluation design

| Module | Metric | Dataset | Notes |
|---|---|---|---|
| M2 | macro-F1, accuracy, confusion matrix | SemEval test (non-circular), synthetic test, gold set | report all three; synthetic alone is optimistic |
| M1 | top-1 clause accuracy, UNMATCHED precision | synthetic test + gold | threshold tuned on dev |
| M3a | precision/recall of campaign membership, Adjusted Rand Index of clusters | synthetic test (planted campaigns with paraphrase levels) | **headline ablation: exact vs MinHash vs SBERT** |
| M3b | accuracy / ROC-AUC | synthetic `is_machine_generated` + HC3 held-out | report limits |
| M4 | traceability coverage (every summary has sources), ROUGE vs a few hand-written reference summaries, qualitative examples | small hand-checked sample | do not over-claim ROUGE |
| System | end-to-end latency on N comments on CPU | synthetic | one honest number |

## 7. Known limitations (state these in the paper)

1. Synthetic data: LLM-written and LLM-labelled → circularity and style bias. Mitigated by SemEval numbers and the hand-labelled gold set; not eliminated.
2. SemEval is tweets about unrelated topics; transfer to legislative text is assumed and then measured, not guaranteed.
3. Three-way stance only; no reasons, intensity, sarcasm or conditional stance.
4. AI-text detection is unreliable and degrades as generators improve.
5. Single English-language prototype; Hindi/code-mixed is future work.
6. Summaries from a news-tuned model can drift; mitigated by showing sources.
7. Real MCA comments are confidential; nothing here has been tested on them.
