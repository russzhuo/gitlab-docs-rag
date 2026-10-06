"""Export the FastAPI OpenAPI schema to frontend/openapi.json.

The frontend generates TypeScript types from this file via `npm run gen:types`.
Run from the repo root with the venv active:

    python scripts/export_openapi.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import app

OUT = ROOT / "frontend" / "openapi.json"
OUT.write_text(json.dumps(app.openapi(), indent=2) + "\n", encoding="utf-8")
print(f"wrote {OUT}")
