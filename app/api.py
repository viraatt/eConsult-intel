"""
FastAPI REST API for eConsultation Intelligence Dashboard.
High-speed endpoints serving SQLite-backed consultation analytics for SIH PS 25035.
"""

import os
import sqlite3
from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from econsult.contract import ConsultationDataContract

app = FastAPI(
    title="eConsultation Intelligence API",
    description="Backend service for public consultation NLP intelligence, clause stance breakdown, campaign deduplication, and traceable summarization.",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
contract = ConsultationDataContract(BASE_DIR)
contract.init_sqlite_db()

def get_db():
    conn = sqlite3.connect(contract.db_path)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/api/health")
def health():
    return {"status": "healthy", "service": "econsult-api", "version": "1.0.0"}

@app.get("/api/stats")
def get_stats():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM comments")
    total_comments = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM comments WHERE is_campaign = 1")
    campaign_comments = cur.fetchone()[0]
    organic_comments = total_comments - campaign_comments

    cur.execute("SELECT COUNT(DISTINCT dup_cluster_id) FROM comments WHERE is_campaign = 1")
    unique_campaign_clusters = cur.fetchone()[0]

    cur.execute("SELECT predicted_stance, COUNT(*) FROM comments GROUP BY predicted_stance")
    stance_dist = {r[0]: r[1] for r in cur.fetchall()}

    cur.execute("SELECT COUNT(*) FROM clauses WHERE clause_id != 'UNMATCHED'")
    total_clauses = cur.fetchone()[0]

    conn.close()
    return {
        "total_comments": total_comments,
        "organic_comments": organic_comments,
        "campaign_comments": campaign_comments,
        "campaign_percentage": round((campaign_comments / total_comments * 100), 2) if total_comments else 0,
        "unique_campaign_clusters": unique_campaign_clusters,
        "stance_distribution": stance_dist,
        "total_clauses": total_clauses
    }

@app.get("/api/clauses")
def list_clauses():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT clause_id, title, text FROM clauses ORDER BY clause_id ASC")
    clauses = [dict(r) for r in cur.fetchall()]

    for c in clauses:
        cid = c["clause_id"]
        cur.execute("SELECT predicted_stance, COUNT(*) FROM comments WHERE matched_clause_id = ? GROUP BY predicted_stance", (cid,))
        s_counts = {r[0]: r[1] for r in cur.fetchall()}
        cur.execute("SELECT COUNT(*) FROM comments WHERE matched_clause_id = ?", (cid,))
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM comments WHERE matched_clause_id = ? AND is_campaign = 1", (cid,))
        camp_count = cur.fetchone()[0]

        c["total_comments"] = total
        c["campaign_comments"] = camp_count
        c["favor"] = s_counts.get("FAVOR", 0)
        c["against"] = s_counts.get("AGAINST", 0)
        c["none"] = s_counts.get("NONE", 0)
        c["favor_pct"] = round((c["favor"] / total * 100), 1) if total else 0
        c["against_pct"] = round((c["against"] / total * 100), 1) if total else 0
        c["none_pct"] = round((c["none"] / total * 100), 1) if total else 0

    conn.close()
    return clauses

@app.get("/api/comments")
def list_comments(
    clause_id: Optional[str] = Query(None),
    stance: Optional[str] = Query(None),
    is_campaign: Optional[bool] = Query(None),
    dup_cluster_id: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    conn = get_db()
    cur = conn.cursor()

    conditions = []
    params = []

    if clause_id:
        conditions.append("matched_clause_id = ?")
        params.append(clause_id)
    if stance:
        conditions.append("predicted_stance = ?")
        params.append(stance.upper())
    if is_campaign is not None:
        conditions.append("is_campaign = ?")
        params.append(1 if is_campaign else 0)
    if dup_cluster_id:
        conditions.append("dup_cluster_id = ?")
        params.append(dup_cluster_id)
    if search:
        conditions.append("text LIKE ?")
        params.append(f"%{search}%")

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

    cur.execute(f"SELECT COUNT(*) FROM comments{where_clause}", params)
    total_matching = cur.fetchone()[0]

    cur.execute(
        f"""
        SELECT comment_id, text, stakeholder_type, matched_clause_id, clause_sim,
               predicted_stance, stance_conf, dup_cluster_id, dup_cluster_size,
               is_campaign, llm_prob, paraphrase_level
        FROM comments{where_clause}
        ORDER BY comment_id ASC
        LIMIT ? OFFSET ?
        """,
        params + [limit, offset]
    )
    comments = [dict(r) for r in cur.fetchall()]
    conn.close()

    return {
        "total": total_matching,
        "limit": limit,
        "offset": offset,
        "items": comments
    }

@app.get("/api/campaigns")
def list_campaigns():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT dup_cluster_id, COUNT(*) as size,
               GROUP_CONCAT(DISTINCT matched_clause_id) as clauses,
               GROUP_CONCAT(DISTINCT paraphrase_level) as paraphrase_levels,
               MIN(text) as sample_text
        FROM comments
        WHERE is_campaign = 1
        GROUP BY dup_cluster_id
        ORDER BY size DESC
    """)
    rows = cur.fetchall()
    campaigns = []
    for r in rows:
        campaigns.append({
            "dup_cluster_id": r["dup_cluster_id"],
            "size": r["size"],
            "clauses": r["clauses"].split(",") if r["clauses"] else [],
            "paraphrase_levels": r["paraphrase_levels"].split(",") if r["paraphrase_levels"] else [],
            "sample_text": r["sample_text"]
        })
    conn.close()
    return campaigns

@app.get("/api/clauses/{clause_id}/summary")
def get_clause_summary(clause_id: str):
    res = contract.get_summary_traceable(clause_id)
    return res

# Static frontend serving
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        file_path = os.path.join(static_dir, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        index_file = os.path.join(static_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Not Found")
