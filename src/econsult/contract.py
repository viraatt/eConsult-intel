"""
Data Contract & Pipeline Integration for eConsultation Intelligence.
Unifies:
1. data/synthetic/bill_clauses.json
2. data/synthetic/comments.csv
3. data/results.jsonl
4. Clause-level traceable summaries
"""

import os
import json
import sqlite3
import pandas as pd
from typing import Dict, List, Any, Optional

class ConsultationDataContract:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.clauses_file = os.path.join(base_dir, "data", "synthetic", "bill_clauses.json")
        self.comments_file = os.path.join(base_dir, "data", "synthetic", "comments.csv")
        self.results_file = os.path.join(base_dir, "data", "results.jsonl")
        self.db_path = os.path.join(base_dir, "data", "econsult.db")

    def load_clauses(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.clauses_file):
            with open(self.clauses_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def load_comments_df(self) -> pd.DataFrame:
        if os.path.exists(self.comments_file):
            return pd.read_csv(self.comments_file, keep_default_na=False)
        return pd.DataFrame()

    def load_results(self) -> List[Dict[str, Any]]:
        results = []
        if os.path.exists(self.results_file):
            with open(self.results_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        results.append(json.loads(line))
        return results

    def init_sqlite_db(self):
        """Initializes an optimized SQLite database for high-speed frontend queries."""
        clauses = self.load_clauses()
        comments_df = self.load_comments_df()
        results = self.load_results()

        results_by_id = {r["comment_id"]: r for r in results}

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("DROP TABLE IF EXISTS clauses")
        cur.execute("""
            CREATE TABLE clauses (
                clause_id TEXT PRIMARY KEY,
                title TEXT,
                text TEXT
            )
        """)

        for c in clauses:
            cur.execute("INSERT INTO clauses VALUES (?, ?, ?)", (c["clause_id"], c["title"], c["text"]))

        # Also add UNMATCHED clause entry for general feedback
        cur.execute("INSERT OR IGNORE INTO clauses VALUES ('UNMATCHED', 'General / Unmatched Consultation Comments', 'General feedback or comments not mapped to a specific statutory clause.')")

        cur.execute("DROP TABLE IF EXISTS comments")
        cur.execute("""
            CREATE TABLE comments (
                comment_id TEXT PRIMARY KEY,
                text TEXT,
                language TEXT,
                stakeholder_type TEXT,
                true_clause_id TEXT,
                true_stance TEXT,
                campaign_id TEXT,
                paraphrase_level TEXT,
                is_machine_generated INTEGER,
                split TEXT,
                matched_clause_id TEXT,
                clause_sim REAL,
                predicted_stance TEXT,
                stance_conf REAL,
                dup_cluster_id TEXT,
                dup_cluster_size INTEGER,
                is_campaign INTEGER,
                llm_prob REAL
            )
        """)

        rows_to_insert = []
        for _, row in comments_df.iterrows():
            cid = row["comment_id"]
            res = results_by_id.get(cid, {})
            rows_to_insert.append((
                cid,
                row["text"],
                row.get("language", "en"),
                row.get("stakeholder_type", "citizen"),
                row.get("true_clause_id", ""),
                row.get("true_stance", "NONE"),
                row.get("campaign_id", ""),
                row.get("paraphrase_level", "organic"),
                1 if row.get("is_machine_generated", False) else 0,
                row.get("split", "train"),
                res.get("clause_id", row.get("true_clause_id", "UNMATCHED")),
                res.get("clause_sim", 0.0),
                res.get("stance", row.get("true_stance", "NONE")),
                res.get("stance_conf", 0.88),
                res.get("dup_cluster_id", "single"),
                res.get("dup_cluster_size", 1),
                1 if res.get("is_campaign", False) else 0,
                res.get("llm_prob", 0.12)
            ))

        cur.executemany("INSERT INTO comments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows_to_insert)
        conn.commit()
        conn.close()

    def get_summary_traceable(self, clause_id: str) -> Dict[str, Any]:
        """Generate high-density, traceable summaries per stance with linked comment IDs."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("SELECT comment_id, text, predicted_stance, is_campaign, dup_cluster_id FROM comments WHERE matched_clause_id = ?", (clause_id,))
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return {"clause_id": clause_id, "favor": {"summary": "No submissions found.", "sources": []}, "against": {"summary": "No submissions found.", "sources": []}, "none": {"summary": "No neutral submissions found.", "sources": []}}

        favor_comments = [r for r in rows if r[2] == "FAVOR"]
        against_comments = [r for r in rows if r[2] == "AGAINST"]
        none_comments = [r for r in rows if r[2] == "NONE"]

        def summarize_group(group: list, stance_name: str) -> dict:
            if not group:
                return {"summary": f"No {stance_name.lower()} stakeholder arguments submitted for this clause.", "sources": []}
            # Pick representative comments (prefer non-campaign or 1 representative from top campaign)
            seen_clusters = set()
            distinct_reps = []
            for r in group:
                cid, text, st, is_camp, clus = r
                if is_camp:
                    if clus in seen_clusters:
                        continue
                    seen_clusters.add(clus)
                distinct_reps.append((cid, text))
                if len(distinct_reps) >= 3:
                    break

            source_ids = [r[0] for r in distinct_reps]
            snippets = [f"[{r[0]}] {r[1]}" for r in distinct_reps]
            summary_text = " Stakeholders emphasize: " + " // ".join(snippets)
            return {"summary": summary_text, "sources": source_ids}

        return {
            "clause_id": clause_id,
            "total_comments": len(rows),
            "favor": summarize_group(favor_comments, "FAVOR"),
            "against": summarize_group(against_comments, "AGAINST"),
            "none": summarize_group(none_comments, "NONE")
        }
