# Roadmap / extension points

Keep the current loop stable (ingest → SQL retrieve → gate → cite) before adding product features.

## Phase 0 — current

- Postgres + pgvector
- Department filter
- Citation + refuse
- Tiny eval set
- Static chat page

## Phase 1 — retrieval quality (do this first)

Hybrid search (BM25 + vector, RRF-fused) is done. The rest is worth doing if eval misses real questions.

- ~~Hybrid search: `tsvector` / BM25 plus pgvector~~ — done (`app/rag.py`, `app/db.py`)
- A cross-encoder rerank on the top 20
- Better chunking for long policy pages (keep heading path in metadata)
- Persist `source_hash` so ingest can update one file instead of `TRUNCATE`

Touch: `app/db.py`, `ingest.py`, `app/rag.py`.  
Do not touch the UI until eval numbers move.

## Phase 2 — more than one knowledge base

- `collection` column (`handbook`, `runbook`, `eng-wiki`)
- Ingest flag `--collection handbook`
- `/chat` body: `{ "collection": "handbook", "department": "hr" }`

This is how you later add a second product without forking the repo.

## Phase 3 — product surface

- ~~Replace `web/index.html` with Next.js~~ — done (`frontend/`)
- ~~Stream tokens~~ — done (`POST /chat/stream`, SSE)
- Conversation table: `conversations`, `messages` (history now lives only in the browser)
- Upload endpoint: PDF / markdown → ingest job

Auth belongs here, not in phase 1.

## Phase 4 — Hong Kong / company deploy

- SSO (Microsoft Entra / Google Workspace) in front of the API
- Row-level filter: department **and** user group
- Store logs of question + retrieved ids (no need to store full answers at first)
- Postgres in Azure Hong Kong / AWS `ap-east-1` / on-prem
- Keep embeddings and source text in the same region as the app

Do not introduce a US-only vector SaaS unless IT already approved it.

## Phase 5 — optional tools (only after refuse works)

- Tool: “fetch this source URL again”
- Tool: structured lookup (e.g. expense category table)
- Never add a tool that invents policy

## Explicit non-goals

- Medical diagnosis, legal advice, or “approve this leave” decisions
- Training a private LLM on the handbook
- Crawling the entire GitLab Handbook

## Resume / interview talking points

When you add a feature, say which phase it belongs to:

1. Quality of retrieval  
2. Multi-collection data model  
3. App shell  
4. Deploy and access control  

That reads as product thinking, not “I installed another LangChain component”.
