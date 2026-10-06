# Runbook

## Prerequisites

- Docker (for Postgres + pgvector, image `pgvector/pgvector:pg16` — pgvector ≥ 0.5 for HNSW)
- Python 3.11+
- An OpenAI-compatible API key (OpenAI, Groq, DeepSeek, local vLLM, …)

## First run

```bash
cd gitlab-docs-rag
docker compose up -d
cp .env.example .env
# edit OPENAI_API_KEY and, if needed, OPENAI_BASE_URL / CHAT_MODEL

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/clean_handbook.py --clone --output data   # generate the corpus
python ingest.py
uvicorn app.main:app --reload --port 8000
```

UI: http://localhost:3000 (run `npm run dev` in `frontend/`)  
Health: http://localhost:8000/health  

Corpus path defaults to `data/`.

## Useful checks

```bash
# Postgres up?
docker compose ps

# How many chunks?
# GET /health → { "ok": true, "chunks": N }

python -m eval.run_eval --json
```

Compare `RETRIEVAL_MODE=vector` vs `RETRIEVAL_MODE=hybrid` (both against the same corpus)
to get the before/after recall@k / MRR numbers. The `expected` substrings in
`eval/questions.json` must match the actual corpus content — check the `rank=-` rows.

## Common failures

| Symptom | Likely cause |
|---------|----------------|
| `ingest.py` cannot connect | `docker compose up -d` not run, or port 5432 taken |
| `dimension mismatch` | `EMBED_DIM` does not match the embedding model |
| Every question is refused | `MIN_RELEVANCE` too high, or ingest never ran |
| Answers ignore department | UI still on `all`, or rows have empty `department` |
| Garbage citations | Source URL missing in the markdown front matter |

## Rebuild the index

`ingest.py` currently **truncates** `documents` and writes everything again.

```bash
python ingest.py
```

Do this after changing chunk size, embedding model, or the corpus files.

## Query the log

Every request (including refusals) is written to `queries`. Useful for debugging why
a question was refused or which chunks came back for a low-scoring query:

```sql
-- last 20 questions, most recent first
SELECT question, department, grounded, refused, best_score, created_at
FROM queries ORDER BY created_at DESC LIMIT 20;

-- refusals that may have missed a real answer
SELECT question, department, best_score, answer_preview
FROM queries WHERE refused = true ORDER BY created_at DESC LIMIT 20;
```

## Point at another Postgres

Set `DATABASE_URL` in `.env`. The instance must have the `vector` extension.

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Then run `python ingest.py` (it also calls `ensure_schema()`).
