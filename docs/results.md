# Empirical Benchmark Results: Duplicate & Campaign Detection

**Generated on:** 2026-10-27 14:25:45  
**Testbed:** `data/synthetic/comments.csv` (Synthetic Companies Bill 2026, N=360: 240 organic + 120 planted campaign comments across 8 families)  
**Hardware:** Local CPU  
**Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense embeddings)  
**Tuning protocol:** Thresholds tuned strictly on DEV split (N=72, 2 campaign families) by maximizing Pairwise F1. Test split (N=72, 3 campaign families) evaluated once without retuning.

---

## 1. Primary Benchmark Metrics (Unseen Test Split)

| Method | Dev Threshold | Pairwise Precision | Pairwise Recall | Pairwise F1 | Adjusted Rand Index (ARI) | Cluster Flag P | Cluster Flag R | Cluster Flag F1 | Organic False Flag Rate | Clusters | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Exact Hash Match** | None | 1.0000 | 0.0952 | 0.1739 | 0.1655 | 1.0000 | 0.3333 | 0.5000 | 0.0000 | 93 | 0.0006 |
| **MinHash-LSH (3-shingle)** | 0.25 | 1.0000 | 0.4286 | 0.6000 | 0.5857 | 1.0000 | 0.6667 | 0.8000 | 0.0000 | 78 | 0.0640 |
| **SBERT + Connected Components** | 0.54 | 0.5822 | 0.9556 | 0.7236 | 0.7022 | 0.7719 | 0.9778 | 0.8627 | 0.2167 | 48 | 0.0006 |
| **SBERT + Average Linkage** | 0.48 | 0.8048 | 0.9556 | 0.8737 | 0.8653 | 0.8333 | 1.0000 | 0.9091 | 0.1500 | 49 | 0.0013 |

---

## 2. Recovery by Paraphrase Level (Test Split)

*Recovery definition: Fraction of planted campaign comments correctly clustered with $\ge 2$ other members of their own planted campaign.*

| Method | Verbatim Recovery (0 edits) | Light Paraphrase Recovery (1-3 word edits) | Heavy Paraphrase Recovery (Full semantic rephrase) | Overall Recovery |
|---|---|---|---|---|
| **Exact Hash Match** | 1.0000 | 0.0000 | 0.0000 | 0.3333 |
| **MinHash-LSH** | 1.0000 | 1.0000 | 0.0000 | 0.6667 |
| **SBERT + Single Link (CC)** | 1.0000 | 1.0000 | 0.9333 | 0.9778 |
| **SBERT + Average Linkage** | 1.0000 | 1.0000 | 0.9333 | 0.9778 |

---

## 3. Key Findings

1. **Exact match completely collapses under paraphrase:** Exact matching scores an ARI of 0.1655 and achieves 0.0% recovery on heavy paraphrases. Any surface edit completely evades exact duplicate detection.
2. **MinHash-LSH catches light edits but fails on semantic paraphrasing:** MinHash recovers 100.0% of light edits, but heavy paraphrasing drops its recovery to 0.0%.
3. **Sentence-BERT maintains robust semantic recovery:** SBERT with average linkage achieves 0.8737 pairwise F1, an ARI of 0.8653, and successfully recovers 93.3% of heavy paraphrases while keeping organic false flags at 15.0%.
4. **Linkage Criterion:** Average-linkage agglomerative clustering avoids single-linkage chaining effects, yielding clean bounded clusters.

---
*Note: As declared throughout, all test comments are synthetic (authored by an LLM to simulate public consultation dynamics under controlled paraphrase levels).*
