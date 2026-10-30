"""Compare the owner's hand labels with the generator's labels. Usage: python score_gold.py gold_to_label.csv
Fill the 'your_stance' column with FAVOR / AGAINST / NONE (or F / A / N) BEFORE looking at comments.csv.
Writes gold.csv (owner labels = ground truth for evaluation)."""
import csv, sys, os, collections
here = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "gold_to_label.csv")
M = {"F": "FAVOR", "A": "AGAINST", "N": "NONE", "FAVOR": "FAVOR", "AGAINST": "AGAINST", "NONE": "NONE"}
gen = {r["comment_id"]: r for r in csv.DictReader(open(os.path.join(here, "..", "synthetic", "comments.csv"), encoding="utf-8"))}
rows = list(csv.DictReader(open(src, encoding="utf-8")))
bad = [r["comment_id"] for r in rows if r["your_stance"].strip().upper() not in M]
if bad: sys.exit(f"{len(bad)} rows still unlabelled or invalid, e.g. {bad[:3]}")
pairs = [(M[r["your_stance"].strip().upper()], gen[r["comment_id"]]["true_stance"]) for r in rows]
n = len(pairs); agree = sum(a == b for a, b in pairs) / n
labs = ["FAVOR", "AGAINST", "NONE"]
pa = {l: sum(a == l for a, _ in pairs) / n for l in labs}; pb = {l: sum(b == l for _, b in pairs) / n for l in labs}
pe = sum(pa[l] * pb[l] for l in labs); kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
print(f"n={n}  raw agreement={agree:.3f}  Cohen's kappa={kappa:.3f}")
cm = collections.Counter(pairs)
print("rows = YOUR label, cols = generator label"); print("%-8s" % "" + "".join("%9s" % l for l in labs))
for a in labs: print("%-8s" % a + "".join("%9d" % cm[(a, b)] for b in labs))
with open(os.path.join(here, "gold.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["comment_id", "clause_id", "text", "gold_stance", "generator_stance"])
    for r in rows: w.writerow([r["comment_id"], r["clause_id"], r["text"], M[r["your_stance"].strip().upper()], gen[r["comment_id"]]["true_stance"]])
print("wrote gold.csv")
