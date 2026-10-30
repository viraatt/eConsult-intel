# eConsultation Intelligence (`eConsult-Intel`)

> **NLP Pipeline for Legislative Consultations: Clause-Level Stance Detection, Astroturfing/Campaign Deduplication, and Policy Intelligence.**  
> *Modelled on the Ministry of Corporate Affairs (MCA) eConsultation System (SIH PS 25035).*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange.svg)](https://sbert.net/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Summary

Public consultation platforms on draft legislation (such as India's MCA eConsultation portal) face critical challenges during notice-and-comment periods:
1. **Lobbying & Astroturfing Campaigns:** Interest groups coordinate mass submissions ranging from verbatim copy-paste to heavily paraphrased templates designed to distort consensus.
2. **Surface Sentiment Inadequacy:** General sentiment analysis fails on statutory text where stakeholders agree with a bill overall but strongly oppose specific clauses.
3. **Traceability Deficit:** Policymakers need clause-linked, auditable feedback rather than high-level black-box summaries.

**eConsult-Intel** addresses these challenges with an end-to-end NLP pipeline providing clause-level stance detection, a 3-tier campaign deduplication engine capable of catching heavy semantic paraphrases, and a lightweight, high-performance policy dashboard.

---

## 🏛️ Pipeline Architecture

```
                                 [ Citizen & Stakeholder Comments ]
                                                │
                                                ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   eConsult-Intel Pipeline                                   │
 ├─────────────────────────┬─────────────────────────────┬─────────────────────────────────────┤
 │   M1: Clause Matcher    │   M2: Stance Classifier     │  M3: Multi-Tier Campaign Detector   │
 │                         │                             │                                     │
 │  • SBERT Dense Cosine   │  • Clause-specific Stance   │  • Tier 1: Exact Hash (MD5/SHA256)  │
 │    Similarity           │    (FAVOR / AGAINST / NONE) │  • Tier 2: MinHash-LSH (3-shingles) │
 │  • Maps feedback to     │  • Domain fine-tuned        │  • Tier 3: SBERT + Average Linkage  │
 │    statutory clauses    │    Transformer baseline     │    (Catches heavy semantic rephrases)
 └─────────────────────────┴─────────────────────────────┴─────────────────────────────────────┘
                                                │
                                                ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │                           M4: Policy Intelligence Delivery                                  │
 ├──────────────────────────────────────────────────────┬──────────────────────────────────────┤
 │  FastAPI Analytics Engine (app/api.py)               │  High-Density Policy Dashboard       │
 │  • SQLite relational query layer                     │  • Clause stance breakdowns          │
 │  • Low-latency aggregation & filtering               │  • Campaign cluster inspection       │
 │  • Traceable comment exploration                     │  • Organic vs. campaign distribution │
 └──────────────────────────────────────────────────────┴──────────────────────────────────────┘
```

---

## 🔬 Empirical Benchmark: Duplicate & Campaign Detection

Evaluated on a controlled 360-comment benchmark (*Companies Amendment Bill 2026*: 240 organic comments across 6 clauses + 120 planted campaign submissions spanning 8 distinct campaigns with verbatim, light, and heavy rewrite tiers).

*Thresholds tuned strictly on DEV split; tested once on unseen TEST split (`N=72`).*

| Detection Method | Dev Thr | Pairwise F1 | Adjusted Rand Index (ARI) | Campaign Flag F1 | Heavy Paraphrase Recovery | Organic False Flag Rate | Latency |
|---|---|---|---|---|---|---|---|
| **Exact Hash Match** | None | 0.1739 | 0.1655 | 0.5000 | **0.0%** *(Total failure)* | 0.0% | < 1 ms |
| **MinHash-LSH (3-shingle)** | 0.25 | 0.6000 | 0.5857 | 0.8000 | **0.0%** *(Fails semantic rewrite)* | 0.0% | ~64 ms |
| **SBERT + Connected Components** | 0.54 | 0.7236 | 0.7022 | 0.8627 | **93.3%** | 21.7% | < 1 ms |
| **SBERT + Average Linkage** | 0.48 | **0.8737** | **0.8653** | **0.9091** | **93.3%** *(Robust)* | **15.0%** | ~1 ms |

### Key Benchmark Takeaways
- **Exact Hash Matching fails under trivial edits:** Completely collapses (0% recovery) when templates undergo minor or heavy wording modifications.
- **MinHash-LSH captures surface edits only:** Recovers 100% of light edits (1–3 word swaps) but drops to 0% recovery on heavy semantic paraphrases.
- **Dense SBERT with Average Linkage provides state-of-the-art resilience:** Recovers **93.3%** of heavily paraphrased campaign comments while avoiding single-link chaining, achieving **0.8737 Pairwise F1** and **0.8653 ARI**.

Visual evaluation plots and metrics are generated in `docs/artifacts/`:
- `fig1_cureus_recovery.png`: Recovery across verbatim, light, and heavy paraphrase levels.
- `fig2_cureus_sensitivity.png`: Precision-Recall and F1 threshold sensitivity curves.
- `fig3_cureus_metrics_heatmap.png`: Cross-method metric comparison matrix.

---

## 📂 Repository Layout

```
├── app/
│   ├── api.py                  # FastAPI REST endpoints for dashboard analytics
│   └── static/
│       └── index.html          # High-density, zero-dependency policy dashboard UI
├── data/
│   ├── eval/                   # Validation scripts & evaluation splits
│   │   ├── gold_to_label.csv   # Blind evaluation set
│   │   └── score_gold.py       # Inter-annotator & Cohen's kappa scoring
│   ├── synthetic/              # Synthetic legislative testbed (MCA simulated)
│   │   ├── bill_clauses.json   # Simulated bill clauses & definitions
│   │   ├── comments.csv        # 360-comment benchmark with controlled labels
│   │   ├── build_comments.py   # Comment dataset synthesis logic
│   │   └── validate_comments.py# Split leakage & balance validation tests
│   ├── econsult.db             # Pre-built SQLite database for instant demo serving
│   └── results.jsonl           # Pre-computed pipeline inference results
├── docs/                       # Engineering & Architectural Specifications
│   ├── ARCHITECTURE.md         # System design, data contracts & component interfaces
│   ├── BUILD_PLAN.md           # Step-by-step phased execution roadmap
│   ├── DECISIONS.md            # Architectural Decision Records (ADRs D1-D16)
│   ├── PROGRESS.md             # Project verification checklist & live results log
│   ├── REFERENCES_VERIFIED.md  # Verified peer-reviewed literature references
│   ├── results.md              # Detailed empirical benchmark tables & findings
│   └── artifacts/              # High-resolution benchmark figures & metric JSONs
├── notebooks/                  # Colab Training & Reproduction Walkthroughs
│   ├── COLAB_STAGE1.md         # Step-by-step SemEval-2016 stance fine-tuning
│   └── COLAB_DEDUPE.md         # Deduplication benchmark execution notebook
├── src/
│   └── econsult/               # Core importable Python package
│       ├── __init__.py
│       ├── contract.py         # Unified data contract & SQLite ingestion pipeline
│       └── dedupe.py           # Exact, MinHash-LSH, and SBERT clustering engine
├── tests/                      # Unit and integration test suites
├── run_app.py                  # Single-command launcher for FastAPI backend & UI
├── run_dedupe_benchmark.py     # CPU-executable deduplication evaluation script
├── requirements-local.txt      # Local app & CPU benchmark dependencies
├── requirements-colab.txt      # GPU training dependencies for Colab
├── AGENTS.md                   # Machine-readable onboarding guide for coding agents
└── CLAUDE.md                   # Agent pointer
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment Setup
Clone the repository and install the dependencies:
```bash
git clone https://github.com/viraatt/eConsult-intel.git
cd eConsult-intel

# Optional: Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements-local.txt
```

### 2. Launch the Policy Intelligence Dashboard
Run the all-in-one local server:
```bash
python run_app.py
```
Open your browser at **`http://127.0.0.1:8000`** to interact with:
- **Statutory Clause Explorer:** Clause-by-clause sentiment and stance breakdown (`FAVOR` / `AGAINST` / `NONE`).
- **Campaign Inspector:** Grouped clusters showing template origins and paraphrase levels.
- **Raw Traceability Table:** Filterable comment-level evidence feed.

### 3. Re-run Deduplication Benchmark
To re-evaluate Exact matching, MinHash-LSH, and SBERT on the test split:
```bash
python run_dedupe_benchmark.py
```
This regenerates metric tables in `docs/results.md` and plots in `docs/artifacts/`.

---

## ⚙️ REST API Endpoints

The FastAPI backend exposes the following endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | `GET` | Health check and API service status |
| `/api/stats` | `GET` | Global consultation statistics, campaign ratios, and stance counts |
| `/api/clauses` | `GET` | Clause directory with stance percentages and total submissions |
| `/api/comments` | `GET` | Filter comments by clause, stance, campaign status, or search query |
| `/api/campaigns`| `GET` | Discovered campaign clusters with paraphrase breakdown |

---

## 🔒 Data & Ethics Disclaimer

All legislative consultation comments included in `data/synthetic/` are **synthetic** and generated for algorithmic validation. Real consultation filings submitted to the Ministry of Corporate Affairs are confidential and non-public. Synthetic generation followed strict leakage-isolation rules to ensure realistic evaluation without circularity or contamination.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
