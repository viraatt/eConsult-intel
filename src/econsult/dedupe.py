"""
M3a Duplicate / Campaign Flagger.
Implements:
1. Exact match (normalized text hash)
2. MinHash-LSH (word 3-shingles)
3. Sentence-BERT semantic clustering (connected components & average linkage)
"""

import re
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score


def norm(t: str) -> str:
    """Lowercase and strip punctuation and excessive whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(t).lower())).strip()


def components(n: int, pairs) -> np.ndarray:
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


def exact_labels(texts: list[str]) -> np.ndarray:
    """Normalized exact match hash clustering."""
    ids = {}
    return np.array([ids.setdefault(norm(t), len(ids)) for t in texts])


def shingles(t: str, k: int = 3) -> set[str]:
    """Extract word k-shingles."""
    w = norm(t).split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def minhash_labels(texts: list[str], thr: float, num_perm: int = 128) -> np.ndarray:
    """MinHash-LSH with Jaccard verification."""
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
            if j > i and mhs[i].jaccard(mhs[j]) >= thr:
                pairs.append((i, j))
    return components(len(texts), pairs)


class SBERTClusterer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            [str(t) for t in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size
        )

    @staticmethod
    def connected_components(emb: np.ndarray, thr: float) -> np.ndarray:
        S = emb @ emb.T
        ii, jj = np.where(np.triu(S >= thr, 1))
        return components(len(emb), zip(ii.tolist(), jj.tolist()))

    @staticmethod
    def average_linkage(emb: np.ndarray, thr: float) -> np.ndarray:
        if len(emb) < 2:
            return np.zeros(len(emb), dtype=int)
        return AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=1.0 - thr
        ).fit_predict(emb)


def evaluate_clustering(sub: pd.DataFrame, labels: np.ndarray, min_cluster: int = 3) -> dict:
    """
    Evaluate duplicate / campaign detection.
    sub: DataFrame with columns 'campaign_id' and 'paraphrase_level'
    labels: cluster ID array for each row in sub
    """
    truth = sub["campaign_id"].to_numpy(dtype=object)
    is_c = truth != ""
    n = len(sub)
    iu = np.triu_indices(n, 1)

    same_pred = (labels[:, None] == labels[None, :])[iu]
    same_true = ((truth[:, None] == truth[None, :]) & is_c[:, None])[iu]

    tp = int((same_pred & same_true).sum())
    fp = int((same_pred & ~same_true).sum())
    fn = int((~same_pred & same_true).sum())

    prf = lambda tp, fp, fn: (
        tp / (tp + fp) if tp + fp else 0.0,
        tp / (tp + fn) if tp + fn else 0.0,
        2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    )
    pP, pR, pF = prf(tp, fp, fn)

    sizes = pd.Series(labels).map(pd.Series(labels).value_counts()).values
    flagged = sizes >= min_cluster
    fP, fR, fF = prf(
        int((flagged & is_c).sum()),
        int((flagged & ~is_c).sum()),
        int((~flagged & is_c).sum())
    )

    true_lab = [t if t else f"organic_{i}" for i, t in enumerate(truth)]
    ari = adjusted_rand_score(true_lab, labels)

    key = pd.DataFrame({"lab": labels, "camp": truth})
    own = key.groupby(["lab", "camp"])["camp"].transform("size").values
    recovered = is_c & (own >= min_cluster)
    lv = sub["paraphrase_level"].to_numpy(dtype=object)

    out = {
        "pair_P": float(pP),
        "pair_R": float(pR),
        "pair_F1": float(pF),
        "ARI": float(ari),
        "flag_P": float(fP),
        "flag_R": float(fR),
        "flag_F1": float(fF),
        "organic_false_flag": float(flagged[~is_c].mean()) if (~is_c).any() else float("nan")
    }

    for l in ("verbatim", "light", "heavy"):
        out[f"recov_{l}"] = float(recovered[lv == l].mean()) if (lv == l).any() else float("nan")

    return out
