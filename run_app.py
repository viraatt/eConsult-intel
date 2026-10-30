"""
Unified Single-Command Runner for eConsultation Intelligence Backend + Policy Dashboard.
Usage:
    python run_app.py
    or
    python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
"""

import os
import sys
import uvicorn

# Ensure proper path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

if __name__ == "__main__":
    print("=" * 70)
    print("  eConsult-Intel: MCA Policy Consultation Intelligence & FastAPI")
    print("  Serving at: http://127.0.0.1:8000")
    print("=" * 70)
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
