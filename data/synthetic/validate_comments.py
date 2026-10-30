"""Sanity checks for comments.csv (no third-party deps)."""
import csv, re, collections, itertools, os
H = os.path.dirname(os.path.abspath(__file__))
R = list(csv.DictReader(open(os.path.join(H, "comments.csv"), encoding="utf-8")))
norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()
sh = lambda t: {" ".join(w[i:i+3]) for w in [norm(t).split()] for i in range(max(1, len(w)-2))}
jac = lambda a, b: len(a & b) / len(a | b)
print("rows", len(R), "ids unique", len({r["comment_id"] for r in R}) == len(R))
print("split", collections.Counter(r["split"] for r in R))
print("level", collections.Counter(r["paraphrase_level"] for r in R))
print("stance", collections.Counter(r["true_stance"] for r in R))
# 1 campaign never straddles splits
cs = collections.defaultdict(set)
for r in R:
    if r["campaign_id"]: cs[r["campaign_id"]].add(r["split"])
assert all(len(v) == 1 for v in cs.values()), "campaign leaks across splits"
# 2 organic exact-duplicates none
org = [r for r in R if r["paraphrase_level"] == "organic"]
assert len({norm(r["text"]) for r in org}) == len(org), "organic exact dupes"
# 3 verbatim copies collapse to one normalised form per family
for k in cs:
    v = {norm(r["text"]) for r in R if r["campaign_id"] == k and r["paraphrase_level"] == "verbatim"}
    assert len(v) == 1, (k, len(v))
# 4 light/heavy are NOT exact matches of template
for k in cs:
    t = next(norm(r["text"]) for r in R if r["campaign_id"] == k and r["paraphrase_level"] == "verbatim")
    assert all(norm(r["text"]) != t for r in R if r["campaign_id"] == k and r["paraphrase_level"] in ("light", "heavy"))
# 5 organic text should not look like a campaign (max Jaccard organic-vs-anything not in campaign)
S = {r["comment_id"]: sh(r["text"]) for r in R}
mx = max(jac(S[a["comment_id"]], S[b["comment_id"]]) for a, b in itertools.combinations(org, 2))
print("max 3-shingle Jaccard among organic pairs: %.3f" % mx)
tm = {k: S[next(r["comment_id"] for r in R if r["campaign_id"] == k and r["paraphrase_level"] == "verbatim")] for k in cs}
for lvl in ("verbatim", "light", "heavy"):
    vals = [jac(tm[r["campaign_id"]], S[r["comment_id"]]) for r in R if r["paraphrase_level"] == lvl]
    print(f"{lvl:8s} Jaccard to template  min %.2f  mean %.2f  max %.2f" % (min(vals), sum(vals)/len(vals), max(vals)))
# 6 opening-phrase monotony check
first = collections.Counter(" ".join(norm(r["text"]).split()[:2]) for r in org)
print("most common organic openers:", first.most_common(4))
print("OK")
