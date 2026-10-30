"""Assemble comments.csv from the authored source lists. Deterministic (seed 7). SYNTHETIC data."""
import csv, json, random, re, itertools, collections, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from organic_c1 import C1; from organic_c2 import C2; from organic_c3 import C3
from organic_c4 import C4; from organic_c5 import C5; from organic_c6 import C6
from campaigns import CAMPAIGNS
HERE = os.path.dirname(os.path.abspath(__file__)); rnd = random.Random(7)
SPLIT_OF_K = {"K1":"train","K2":"train","K3":"dev","K4":"dev","K5":"dev","K6":"test","K7":"test","K8":"test"}
rows = []
for i, L in enumerate([C1, C2, C3, C4, C5, C6], 1):
    by = collections.defaultdict(list)
    for s, st, t in L: by[s].append((st, t))
    for s, items in by.items():
        rnd.shuffle(items); n = len(items)
        n_tr, n_dv = (7, 4) if n == 14 else (5, 3)   # per clause: train 19 / dev 11 / test 10
        for j, (st, t) in enumerate(items):
            sp = "train" if j < n_tr else "dev" if j < n_tr + n_dv else "test"
            rows.append(dict(text=t, stakeholder_type=st, true_clause_id=f"C{i}", true_stance=s,
                             campaign_id="", paraphrase_level="organic", split=sp))
def verbatim_variants(t):
    return [t, t + " ", t.lower(), t.replace(". ", ".  "), t.rstrip(".")]
for k, v in CAMPAIGNS.items():
    stk = itertools.cycle(["citizen", "industry"])
    for lvl, texts in (("verbatim", verbatim_variants(v["template"])), ("light", v["light"]), ("heavy", v["heavy"])):
        for t in texts:
            rows.append(dict(text=t, stakeholder_type=next(stk), true_clause_id=v["clause"], true_stance=v["stance"],
                             campaign_id=k, paraphrase_level=lvl, split=SPLIT_OF_K[k]))
rnd.shuffle(rows)
for n, r in enumerate(rows, 1):
    r["comment_id"] = f"cm_{n:04d}"; r["language"] = "en"; r["is_machine_generated"] = "False"
cols = ["comment_id","text","language","stakeholder_type","true_clause_id","true_stance","campaign_id",
        "paraphrase_level","is_machine_generated","split"]
with open(os.path.join(HERE, "comments.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows([{c: r[c] for c in cols} for r in rows])
# gold sheet: 60 organic TEST comments, labels hidden, for the owner to hand-label
gold = [r for r in rows if r["split"] == "test" and r["paraphrase_level"] == "organic"]
gold.sort(key=lambda r: r["comment_id"])
with open(os.path.join(HERE, "..", "eval", "gold_to_label.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["comment_id","clause_id","clause_title","text","your_stance"])
    titles = {c["clause_id"]: c["title"] for c in json.load(open(os.path.join(HERE, "bill_clauses.json")))}
    for r in gold: w.writerow([r["comment_id"], r["true_clause_id"], titles[r["true_clause_id"]], r["text"], ""])
print("rows", len(rows), "gold", len(gold))
