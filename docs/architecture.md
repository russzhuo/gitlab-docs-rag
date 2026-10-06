# Architecture

## Goal

Answer staff questions from indexed handbook pages. If the index does not contain the answer, refuse. Do not invent policy.

This is **document lookup**, not an HR or legal decision system.

## Request path

```
Browser  →  POST /chat
         →  embed question
         →  hybrid retrieval: vector + full-text, RRF-fused (optional department filter)
         →  similarity gate (cosine, unchanged by mode)
         →  LLM with top chunks + citations
         →  JSON { answer, refused, citations, hits }  + one row logged to `queries`
```

LangChain is only used for splitting, embeddings, and the chat model.
Retrieval and persistence are Postgres.

## Modules

| Path | Responsibility | Safe to extend |
|------|----------------|----------------|
| `ingest.py` | Load markdown, split, embed, replace rows | New file types, incremental ingest |
| `app/db.py` | Connection + schema (single source) + query log | Migrations, pooling |
| `app/rag.py` | Hybrid retrieve + generate + refuse | Rerank, tools |
| `app/main.py` | HTTP API + static UI | Auth, new routes |
| `frontend/` | React (Next.js) chat UI + markdown | New UI, streaming |
| `eval/run_eval.py` | Eval driver against live DB | Larger gold set |
| `eval/metrics.py` | Pure metric functions | New metrics |

## Data

Table `documents`:

- identity: `path`, `chunk_index`
- filter: `department`
- cite: `title`, `source`
- retrieve: `content`, `embedding vector(1536)`, `fts tsvector`

Default embedding: `text-embedding-3-small` (1536 dims). Changing the model means changing `EMBED_DIM` and rebuilding the table.

## Relevance gate

Score is cosine similarity: `1 - (embedding <=> query)`.

If `best_score < MIN_RELEVANCE`, `/chat` returns `refused: true` and does not ask the LLM to guess.

The gate is always cosine similarity, independent of `RETRIEVAL_MODE`; hybrid/text only
change the ordering and which chunks reach the LLM.

Tune `MIN_RELEVANCE` with `python -m eval.run_eval --json`, not by feel.

## What this version does not do

- Login / SSO / per-user ACL
- Incremental ingest (current ingest truncates the table)
- Cross-encoder rerank (`retrieve()` has a `mode` extension point for it)
- Storing full answers (the `queries` table keeps a preview, not the full text)

Those belong in later phases; see `docs/roadmap.md`.
