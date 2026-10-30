# Colab: exact vs MinHash vs SBERT duplicate detection (synthetic benchmark)
Copy each cell into Colab, in order. Runtime: CPU is enough (a T4 only makes the SBERT encoding a few seconds faster).
**Before you start:** have `data/synthetic/comments.csv` from the repo zip on your computer. Cell 2 asks you to upload it.
**What this answers:** do exact-match methods miss paraphrased comment campaigns, and do MinHash and SBERT recover them? Thresholds are tuned on dev, and test is run once. All data are synthetic (LLM-authored, planted campaigns): say so everywhere you report these numbers.
**Paste back after running:** the printed output of cells 2, 5, 6 and 8, and tell me if any cell errors.

## Cell 1 — Install

```python
!pip -q install sentence-transformers datasketch
import sklearn, datasketch, sentence_transformers
print("sklearn", sklearn.__version__, "| datasketch", datasketch.__version__, "| sbert", sentence_transformers.__version__)
```

## Cell 2 — Load the synthetic comments

```python
import os, re, json, time, itertools, collections
import numpy as np, pandas as pd
from google.colab import files

SEED = 7
np.random.seed(SEED)
OUT_DIR = os.environ.get("ECON_OUT", "/content/drive/MyDrive/econsult")
try:
    from google.colab import drive
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")
except Exception as e:
    print("Drive not mounted, saving locally instead:", e)
    OUT_DIR = "/content/dedupe_out"
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists("comments.csv"):
    print("Choose comments.csv (from data/synthetic/ in the repo zip) ...")
    files.upload()
df = pd.read_csv("comments.csv", keep_default_na=False)
df["campaign_id"] = df["campaign_id"].astype(str)
print(df.shape)
print(df.groupby(["split", "paraphrase_level"]).size().unstack(fill_value=0))
print("campaign families per split:", df[df.campaign_id != ""].groupby("split").campaign_id.nunique().to_dict())
```

## Cell 3 — Define the three detectors and embed

```python
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(t).lower())).strip()

def components(n, pairs):
    """Connected components via union-find. pairs: iterable of (i, j)."""
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = [find(i) for i in range(n)]
    ids = {}
    return np.array([ids.setdefault(r, len(ids)) for r in roots])

# ---- Detector 1: exact match after normalisation (lowercase, strip punctuation/whitespace)
def exact_labels(texts, thr=None):
    ids = {}
    return np.array([ids.setdefault(norm(t), len(ids)) for t in texts])

# ---- Detector 2: MinHash + LSH on word 3-shingles
def shingles(t, k=3):
    w = norm(t).split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}

def minhash_labels(texts, thr, num_perm=128):
    mhs = []
    for t in texts:
        m = MinHash(num_perm=num_perm, seed=1)
        for s in shingles(t):
            m.update(s.encode("utf8"))
        mhs.append(m)
    lsh = MinHashLSH(threshold=thr, num_perm=num_perm)
    for i, m in enumerate(mhs):
        lsh.insert(str(i), m)
    pairs = []
    for i, m in enumerate(mhs):
        for c in lsh.query(m):
            j = int(c)
            if j > i and mhs[i].jaccard(mhs[j]) >= thr:   # verify the LSH candidate
                pairs.append((i, j))
    return components(len(texts), pairs)

# ---- Detector 3: SBERT embeddings. Two clustering rules on cosine similarity.
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def encode(texts):
    return sbert.encode([str(t) for t in texts], normalize_embeddings=True, show_progress_bar=False)

def sbert_cc_labels(emb, thr):
    """Link every pair with cosine >= thr, take connected components (single linkage)."""
    S = emb @ emb.T
    ii, jj = np.where(np.triu(S >= thr, 1))
    return components(len(emb), zip(ii.tolist(), jj.tolist()))

def sbert_avg_labels(emb, thr):
    """Average-linkage agglomerative clustering; stops merging below cosine thr (less chaining)."""
    if len(emb) < 2:
        return np.zeros(len(emb), dtype=int)
    return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                   distance_threshold=1 - thr).fit_predict(emb)

EMB = encode(df.text.tolist())
print("embeddings:", EMB.shape)
```

## Cell 4 — Metrics and sanity check

```python
from sklearn.metrics import adjusted_rand_score
MIN_CLUSTER = 3   # a cluster of >= 3 comments is flagged as a possible campaign

def evaluate(sub, labels):
    """sub: rows of one split; labels: predicted cluster id per row."""
    truth = sub.campaign_id.to_numpy(dtype=object)
    is_c = truth != ""
    n = len(sub)
    iu = np.triu_indices(n, 1)
    same_pred = (labels[:, None] == labels[None, :])[iu]
    same_true = ((truth[:, None] == truth[None, :]) & is_c[:, None])[iu]
    tp = int((same_pred & same_true).sum()); fp = int((same_pred & ~same_true).sum()); fn = int((~same_pred & same_true).sum())
    prf = lambda tp, fp, fn: (tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0,
                              2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    pP, pR, pF = prf(tp, fp, fn)
    sizes = pd.Series(labels).map(pd.Series(labels).value_counts()).values
    flagged = sizes >= MIN_CLUSTER
    fP, fR, fF = prf(int((flagged & is_c).sum()), int((flagged & ~is_c).sum()), int((~flagged & is_c).sum()))
    true_lab = [t if t else f"organic_{i}" for i, t in enumerate(truth)]
    ari = adjusted_rand_score(true_lab, labels)
    # recovered: member shares its predicted cluster with >= 3 members of its OWN campaign (incl. itself)
    key = pd.DataFrame({"lab": labels, "camp": truth})
    own = key.groupby(["lab", "camp"])["camp"].transform("size").values
    recovered = is_c & (own >= MIN_CLUSTER)
    lv = sub.paraphrase_level.to_numpy(dtype=object)
    out = dict(pair_P=pP, pair_R=pR, pair_F1=pF, ARI=ari, flag_P=fP, flag_R=fR, flag_F1=fF,
               organic_false_flag=float(flagged[~is_c].mean()) if (~is_c).any() else float("nan"))
    for l in ("verbatim", "light", "heavy"):
        out[f"recov_{l}"] = float(recovered[lv == l].mean()) if (lv == l).any() else float("nan")
    return out

def run(method, sub_idx, thr=None):
    sub = df.loc[sub_idx]
    texts = sub.text.tolist(); emb = EMB[sub_idx]
    t0 = time.perf_counter()
    if method == "exact":      lab = exact_labels(texts)
    elif method == "minhash":  lab = minhash_labels(texts, thr)
    elif method == "sbert_cc": lab = sbert_cc_labels(emb, thr)
    elif method == "sbert_avg": lab = sbert_avg_labels(emb, thr)
    secs = time.perf_counter() - t0
    r = evaluate(sub, lab); r["seconds"] = secs; r["n_clusters"] = int(len(set(lab)))
    return r

idx = {s: df.index[df.split == s].to_numpy() for s in ("train", "dev", "test")}
# Sanity checks on the training split (nothing is tuned here)
print("exact on train:", {k: round(v, 3) for k, v in run("exact", idx["train"]).items() if k in ("pair_F1", "recov_verbatim", "recov_heavy", "organic_false_flag")})
```

## Cell 5 — Tune thresholds on the dev split only

```python
# Tune thresholds on DEV ONLY: maximise pairwise F1; if several thresholds tie, take the middle of the tied range (a plateau centre is more stable than its edge).
GRIDS = {"minhash": np.round(np.arange(0.10, 0.96, 0.05), 3),
         "sbert_cc": np.round(np.arange(0.30, 0.98, 0.02), 3),
         "sbert_avg": np.round(np.arange(0.30, 0.98, 0.02), 3)}
dev_curves, BEST = {}, {}
for m, grid in GRIDS.items():
    rows = [dict(thr=float(t), **run(m, idx["dev"], float(t))) for t in grid]
    dev_curves[m] = pd.DataFrame(rows)
    d = dev_curves[m]
    tied = d[np.isclose(d.pair_F1, d.pair_F1.max())].sort_values("thr").reset_index(drop=True)
    best = tied.iloc[len(tied) // 2]
    BEST[m] = float(best.thr)
    print(f"{m:9s} best dev threshold {BEST[m]:.2f}  dev pair-F1 {best.pair_F1:.3f}  (P {best.pair_P:.3f} R {best.pair_R:.3f})")
print("exact: no threshold")
```

## Cell 6 — Evaluate once on the test split

```python
# TEST evaluation with the dev-tuned thresholds. Run this once; do not retune after looking at it.
test_rows = {}
for m in ("exact", "minhash", "sbert_cc", "sbert_avg"):
    test_rows[m] = run(m, idx["test"], BEST.get(m))
T = pd.DataFrame(test_rows).T
cols = ["pair_P", "pair_R", "pair_F1", "ARI", "flag_P", "flag_R", "flag_F1", "organic_false_flag", "n_clusters", "seconds"]
print(f"TEST split: {len(idx['test'])} comments ({int((df.loc[idx['test']].campaign_id != '').sum())} in campaigns, 3 families)")
print("thresholds used:", BEST)
print(T[cols].astype(float).round(3).to_string())
print()
print("Recovery by paraphrase level (share of campaign members grouped with >=2 others of their own campaign):")
print(T[["recov_verbatim", "recov_light", "recov_heavy"]].astype(float).round(3).to_string())
```

## Cell 7 — Figures

```python
import matplotlib
import matplotlib.pyplot as plt
names = {"exact": "Exact match", "minhash": "MinHash (3-shingle)", "sbert_cc": "SBERT + connected comp.", "sbert_avg": "SBERT + avg-linkage"}
levels = ["verbatim", "light", "heavy"]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
w = 0.2
for k, m in enumerate(names):
    vals = [float(T.loc[m, f"recov_{l}"]) for l in levels]
    ax.bar(np.arange(3) + (k - 1.5) * w, vals, w, label=names[m])
ax.set_xticks(range(3)); ax.set_xticklabels(["Verbatim", "Light edit", "Heavy paraphrase"])
ax.set_ylim(0, 1.05); ax.set_ylabel("Campaign members recovered"); ax.legend(fontsize=8, loc="lower left")
ax.set_title("Recovery by paraphrase level (synthetic test split)")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_recovery_by_level.png"), dpi=200); plt.show()

# Sensitivity: pairwise F1 vs threshold on dev (used for tuning) and test (NOT used for tuning)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
test_curves = {}
for ax, m in zip(axes, ("minhash", "sbert_cc", "sbert_avg")):
    tc = pd.DataFrame([dict(thr=float(t), **run(m, idx["test"], float(t))) for t in GRIDS[m]])
    test_curves[m] = tc
    ax.plot(dev_curves[m].thr, dev_curves[m].pair_F1, label="dev (tuned here)")
    ax.plot(tc.thr, tc.pair_F1, "--", label="test (not tuned)")
    ax.axvline(BEST[m], color="grey", lw=0.8); ax.set_title(names[m], fontsize=9); ax.set_xlabel("threshold")
axes[0].set_ylabel("pairwise F1"); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig_threshold_sensitivity.png"), dpi=200); plt.show()
```

## Cell 8 — Error analysis (read this; it becomes the discussion)

```python
# Error analysis on TEST at the dev-tuned thresholds. Read these: they are the discussion section.
sub = df.loc[idx["test"]].copy()
cut = lambda s, n=150: (s[:n] + "...") if len(s) > n else s
for m in ("minhash", "sbert_cc"):
    lab = {"minhash": minhash_labels(sub.text.tolist(), BEST["minhash"]),
           "sbert_cc": sbert_cc_labels(EMB[idx["test"]], BEST["sbert_cc"])}[m]
    sub["lab"] = lab
    sub["size"] = sub.groupby("lab")["lab"].transform("size")
    print("=" * 90, "\n", m, "threshold", BEST[m])
    fp = sub[(sub.campaign_id == "") & (sub["size"] >= MIN_CLUSTER)]
    print(f"organic comments falsely flagged: {len(fp)} of {int((sub.campaign_id == '').sum())}")
    for _, r in fp.head(4).iterrows():
        mix = sub[sub.lab == r.lab].campaign_id.replace("", "organic").value_counts().to_dict()
        print(f"  [{r.comment_id}] {cut(r.text)}\n      cluster mix: {mix}")
    miss = sub[(sub.campaign_id != "") & (sub.paraphrase_level == "heavy")]
    own = sub.groupby(["lab", "campaign_id"])["campaign_id"].transform("size")
    miss = miss[own.loc[miss.index] < MIN_CLUSTER]
    print(f"heavy-paraphrase campaign members not recovered: {len(miss)} of {int((sub.paraphrase_level == 'heavy').sum())}")
    for _, r in miss.head(3).iterrows():
        print(f"  [{r.comment_id}, {r.campaign_id}] {cut(r.text)}")
```

## Cell 9 — Save results

```python
out = dict(
    note="SYNTHETIC benchmark: all comments (organic and campaign) were authored by an LLM; campaigns are planted. Results measure detector behaviour on this construction, not on real consultations.",
    seed=SEED, sbert_model="sentence-transformers/all-MiniLM-L6-v2", min_cluster=MIN_CLUSTER,
    n_per_split={s: int(len(i)) for s, i in idx.items()},
    tuned_on="dev (pairwise F1)", thresholds=BEST,
    test=json.loads(T.astype(float).to_json(orient="index")),
    dev_curves={m: json.loads(d.to_json(orient="records")) for m, d in dev_curves.items()},
    test_curves={m: json.loads(d.to_json(orient="records")) for m, d in test_curves.items()},
)
with open(os.path.join(OUT_DIR, "dedupe_metrics.json"), "w") as f:
    json.dump(out, f, indent=2)
print("saved:", sorted(os.listdir(OUT_DIR)))
```
