from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from langchain_deepseek import ChatDeepSeek
from sqlalchemy.orm import Session

from app.config import (
    CHAT_MODEL,
    DEEPSEEK_API_KEY,
    EMBED_API_KEY,
    EMBED_BASE_URL,
    EMBED_MODEL,
    MIN_RELEVANCE,
    RERANK,
    RERANK_CANDIDATES,
    RERANK_MIN_RELEVANCE,
    RERANK_MODEL,
    DASHSCOPE_API_KEY,
    DASHSCOPE_WORKSPACE_ID,
    RETRIEVAL_MODE,
    SEARCH_K,
    TOP_K,
)
from app.repositories import documents as doc_repo
from app.repositories import query_log as qlog_repo
from app.schemas import DoneEvent, MetaEvent, RefusedEvent, TokenEvent

logger = logging.getLogger(__name__)

SYSTEM = """You are an internal company handbook assistant.
Answer ONLY from the provided handbook excerpts.
Rules:
- If the excerpts are not enough, say you cannot find this in the handbook. Do not guess policy.
- Quote or closely paraphrase the relevant rule.
- Format your answer as Markdown (bullet lists, bold, tables) where it helps readability.
- You are not giving legal or medical advice. This is documentation lookup only.
- If the user mentions personal details (dates, role), use them only to pick the relevant policy, not to invent an exception.
"""

RRF_K = 60


def _is_english(text: str) -> bool:
    """Reject questions that contain non-ASCII letters (e.g. CJK, Cyrillic)."""
    return not any(ch.isalpha() and ord(ch) > 127 for ch in text)


def _embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=EMBED_API_KEY,
        base_url=EMBED_BASE_URL,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
        chunk_size=20,
    )


def get_llm() -> ChatDeepSeek:
    return ChatDeepSeek(model=CHAT_MODEL, api_key=DEEPSEEK_API_KEY, temperature=0)


_reranker_client = None
def _get_reranker_client():
    global _reranker_client
    if _reranker_client is None:
        _reranker_client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=f"https://{DASHSCOPE_WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/compatible-api/v1",
        )
    return _reranker_client


def _rerank(question: str, hits: list[dict]) -> list[dict]:
    if not hits:
        return hits

    client = _get_reranker_client()
    response = client.post(
        "/reranks",
        body={
            "model": RERANK_MODEL,
            "query": question,
            "documents": [h["content"] for h in hits],
            "top_n": TOP_K,
        },
        cast_to=object,
    )

    results = sorted(
        response["results"],
        key=lambda r: float(r["relevance_score"]),
        reverse=True,
    )

    ordered = []
    for r in results:
        h = hits[r["index"]]
        h["rerank"] = round(float(r["relevance_score"]), 4)
        ordered.append(h)
    return ordered


def _to_hits(rows: list) -> list[dict]:
    hits = []
    for doc, score in rows:
        hits.append({
            "id": doc.id,
            "title": doc.title,
            "section": doc.section,
            "source": doc.source,
            "path": doc.path,
            "preview": doc.preview,
            "content": doc.content,
            "score": score,
        })
    return hits


def rrf_fuse(vector_hits: list[dict], text_hits: list[dict], k: int, k_rank: int = RRF_K) -> list[dict]:
    """Reciprocal-rank-fuse two ranked lists, merged by document id."""
    fused: dict = {}
    rrf: dict = {}
    for rank, h in enumerate(vector_hits, start=1):
        key = h["id"]
        fused[key] = h
        rrf[key] = rrf.get(key, 0.0) + 1.0 / (k_rank + rank)
    for rank, h in enumerate(text_hits, start=1):
        key = h["id"]
        if key not in fused:
            fused[key] = h
        rrf[key] = rrf.get(key, 0.0) + 1.0 / (k_rank + rank)
    ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)
    hits = []
    for key, rrf_score in ordered[:k]:
        h = dict(fused[key])
        h["rrf"] = round(rrf_score, 4)
        hits.append(h)
    return hits


def _dedup_by_path(hits: list[dict], k: int) -> list[dict]:
    """Keep at most one chunk per document path, then cap to k."""
    seen: set = set()
    out: list[dict] = []
    for h in hits:
        p = h.get("path")
        if p in seen:
            continue
        seen.add(p)
        out.append(h)
        if len(out) >= k:
            break
    return out


def retrieve(db: Session, question: str, k: int = TOP_K, mode: str | None = None,
             rerank: bool | None = None):
    """Return (hits, best_signal). best_signal is the grounding score: the top
    cosine similarity among candidates, or the top rerank score when reranking.

    `rerank` overrides the RERANK env var (None = use config), so the eval can
    A/B compare rerank on/off over the same questions.
    """
    mode = mode or RETRIEVAL_MODE
    qvec = _embedder().embed_query(question)
    vector_hits = _to_hits(doc_repo.vector_search(db, qvec, SEARCH_K))

    if mode == "vector":
        candidates = vector_hits
    elif mode == "text":
        candidates = _to_hits(doc_repo.text_search(db, qvec, question, SEARCH_K))
    elif mode == "hybrid":
        text_hits = _to_hits(doc_repo.text_search(db, qvec, question, SEARCH_K))
        candidates = rrf_fuse(vector_hits, text_hits, SEARCH_K * 2)
    else:
        raise ValueError(f"Unknown RETRIEVAL_MODE: {mode}")

    # Non-rerank grounding signal: top cosine similarity among candidates.
    # `score` is cosine in every mode, so this never mixes scales.
    best_signal = max((h["score"] for h in candidates), default=0.0)

    use_rerank = RERANK if rerank is None else rerank
    pool = _dedup_by_path(candidates, RERANK_CANDIDATES if use_rerank else k)
    if use_rerank:
        logger.info("reranking %d candidates for question %r", len(pool), question)
        pool = _rerank(question, pool)[:k]
        best_signal = max((h["rerank"] for h in pool), default=0.0)

    return pool, best_signal


def build_messages(question: str, context: list[str]) -> list:
    return [
        SystemMessage(content=SYSTEM),
        HumanMessage(
            content="Handbook excerpts:\n\n" + "\n\n".join(context) + f"\n\nQuestion: {question}"
        ),
    ]


def _citations(hits: list[dict]) -> list[dict]:
    return [
        {"n": i, "title": h["title"], "section": h["section"],
         "source": h["source"], "score": round(h["score"], 3)}
        for i, h in enumerate(hits, start=1)
    ]


def _without_content(hits: list[dict]) -> list[dict]:
    return [{k: v for k, v in h.items() if k != "content"} for h in hits]


def _log_query(db: Session, question: str, hits: list[dict],
               best_score: float, grounded: bool, refused: bool, answer_text: str) -> None:
    try:
        top_hits = [
            {"id": h["id"], "title": h["title"], "section": h["section"],
             "score": round(h["score"], 3)}
            for h in hits
        ]
        qlog_repo.add_query(db, question, top_hits, best_score,
                            grounded, refused, (answer_text or "")[:200])
        db.commit()
    except Exception:
        db.rollback()  # observability must never break the answer path


def _grounding_threshold() -> float:
    """Pick the threshold matching the signal `retrieve()` returns."""
    return RERANK_MIN_RELEVANCE if RERANK else MIN_RELEVANCE


def answer(db: Session, question: str) -> dict:
    if not _is_english(question):
        _log_query(db, question, [], 0.0, False, True, "")
        return {
            "answer": "Only English is supported. Please rephrase your question in English.",
            "refused": True,
            "citations": [],
            "hits": [],
        }

    hits, best_signal = retrieve(db, question)
    grounded = bool(hits) and best_signal >= _grounding_threshold()

    if not grounded:
        result = {
            "answer": "I cannot find this in the indexed handbook. I will not guess a company policy. Try a more specific question.",
            "refused": True,
            "citations": [],
            "hits": _without_content(hits),
        }
        _log_query(db, question, hits, best_signal, False, True, "")
        return result

    context = [
        f"[{i}] title={h['title']} section={h['section']} url={h['source']}\n{h['content']}"
        for i, h in enumerate(hits, start=1)
    ]
    messages = build_messages(question, context)
    resp = get_llm().invoke(messages)
    answer_text = resp.content
    _log_query(db, question, hits, best_signal, True, False, answer_text)
    return {
        "answer": answer_text,
        "refused": False,
        "citations": _citations(hits),
        "hits": _without_content(hits),
    }


def stream_answer(db: Session, question: str):
    """Yield SSE-ready stream events: meta / token / done / refused."""
    if not _is_english(question):
        _log_query(db, question, [], 0.0, False, True, "")
        yield RefusedEvent(answer="Only English is supported. Please rephrase your question in English.")
        return

    hits, best_signal = retrieve(db, question)
    grounded = bool(hits) and best_signal >= _grounding_threshold()

    if not grounded:
        _log_query(db, question, hits, best_signal, False, True, "")
        yield RefusedEvent(answer="I cannot find this in the indexed handbook. I will not guess a company policy.")
        return

    context = [
        f"[{i}] title={h['title']} section={h['section']} url={h['source']}\n{h['content']}"
        for i, h in enumerate(hits, start=1)
    ]
    messages = build_messages(question, context)
    yield MetaEvent(citations=_citations(hits))
    parts: list[str] = []
    for chunk in get_llm().stream(messages):
        text = chunk.content or ""
        if text:
            parts.append(text)
            yield TokenEvent(text=text)
    yield DoneEvent()
    _log_query(db, question, hits, best_signal, True, False, "".join(parts))
