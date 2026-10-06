# Handbook Assistant

Internal documentation RAG over a **subset** of the public GitLab Handbook.

Staff questions are answered from Postgres (`pgvector`). If the index does not contain the answer, the API **refuses** instead of inventing policy.

This is document lookup. It is not an HR system.

## Read in this order

| Doc | What it covers |
|-----|----------------|
| [docs/runbook.md](docs/runbook.md) | Install, run, rebuild index, common failures |
| [docs/architecture.md](docs/architecture.md) | Request path, modules, relevance gate |
| [docs/data.md](docs/data.md) | Corpus generation + license |
| [docs/roadmap.md](docs/roadmap.md) | Planned extension points, what not to add yet |

## Quick start

```bash
docker compose up -d
cp .env.example .env          # set EMBED_API_KEY / EMBED_BASE_URL / DEEPSEEK_API_KEY
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/clean_handbook.py --clone --output data   # generate the corpus (needs git + proxy)
python ingest.py

uvicorn app.main:app --reload --port 8000                # API — terminal 1
cd frontend && npm install && npm run dev                # frontend — terminal 2
```

http://localhost:3000 (frontend) · http://localhost:8000 (API)

Corpus: `data/` — generated from the GitLab Handbook (see `docs/data.md`).

## Current feature set

- Markdown ingest with source URL + top-level section
- Hybrid retrieval: full-text (BM25) + vector, RRF-fused, in Postgres
- Cosine similarity gate → refuse (never guesses policy)
- Citations back to handbook.gitlab.com
- Session auth + persistent multi-conversation history
- Query observability (question → hits → score → refusal decision)
- Retrieval eval: recall@k, MRR, refusal precision/recall
- Next.js (SSR) + TypeScript chat UI with markdown rendering

## Extension rule

New behaviour should plug into one place:

- better search → `app/services/rag_service.py` + `app/repositories/`
- new files → `scripts/clean_handbook.py` + `ingest.py`
- new product UI → edit `frontend/`, keep `/chat`
- company deploy → auth in front of `app/main.py`

Details and a phased list: [docs/roadmap.md](docs/roadmap.md).

## API

`POST /chat`

```json
{
  "question": "Can I expense a monitor?",
  "conversation_id": 3
}
```

`department` is optional and legacy — retrieval is the filter now.

`POST /chat/stream` — same body, server-sent events (`meta` / `token` / `done` / `refused`).

Retrieval mode is set by `RETRIEVAL_MODE` (`vector` | `text` | `hybrid`, default `hybrid`).

## Evaluation

The gold set lives in [eval/questions.json](eval/questions.json) — answerable questions
with `expected` terms, plus "should refuse" negatives. Run it after `ingest.py`:

```bash
python -m eval.run_eval            # per-question table + summary at MIN_RELEVANCE
python -m eval.run_eval --json     # machine-readable
python -m eval.run_eval --sweep    # scan thresholds to pick MIN_RELEVANCE
```

Metrics: `recall@k` (did the right chunk reach the top-k), `MRR` (how high it ranked),
and `refusal precision/recall` (refuse when it should, answer when it should). Use
`--sweep` to re-tune `MIN_RELEVANCE` after changing the embedding model or the corpus —
a threshold is only as good as the eval it was tuned on.

## Resume line

Internal documentation assistant over employee handbook policies. Chunks and embeddings live in Postgres with pgvector; retrieval is SQL with department filters, source citations, and a similarity gate that refuses when the answer is not in the corpus.

## License

Code in this repository is MIT licensed — see [LICENSE](LICENSE).

The handbook corpus is **not** committed here: it is generated locally from GitLab's
public [Handbook](https://handbook.gitlab.com), which is © GitLab and licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). See
[docs/data.md](docs/data.md#license) for attribution requirements.
