from langchain_core.messages import HumanMessage, SystemMessage

from app.services.rag_service import _dedup_by_path, _is_english, build_messages, rrf_fuse


def test_build_messages_no_history():
    msgs = build_messages("q", ["ctx"])
    assert len(msgs) == 2
    assert type(msgs[0]) is SystemMessage
    assert type(msgs[1]) is HumanMessage


def test_is_english_accepts_ascii():
    assert _is_english("What is the PTO policy?")


def test_is_english_rejects_cjk():
    assert not _is_english("请假policy")
    assert not _is_english("那财务呢")


def test_rrf_fuse_ranks_shared_doc_first():
    vec = [{"id": 1, "title": "a", "score": 0.9}, {"id": 2, "title": "b", "score": 0.8}]
    txt = [{"id": 2, "title": "b", "score": 0.8}, {"id": 3, "title": "c", "score": 0.7}]
    fused = rrf_fuse(vec, txt, k=3)
    assert [h["id"] for h in fused] == [2, 1, 3]


def test_rrf_fuse_caps_k():
    vec = [{"id": i, "title": str(i), "score": 1.0} for i in range(10)]
    fused = rrf_fuse(vec, [], k=4)
    assert len(fused) == 4


def test_dedup_by_path_keeps_first_per_path():
    hits = [
        {"path": "a.md", "title": "a1", "score": 0.9},
        {"path": "b.md", "title": "b", "score": 0.8},
        {"path": "a.md", "title": "a2", "score": 0.7},
        {"path": "c.md", "title": "c", "score": 0.6},
    ]
    out = _dedup_by_path(hits, 4)
    assert [h["path"] for h in out] == ["a.md", "b.md", "c.md"]


def test_rerank_sorts_by_relevance(monkeypatch):
    from app.services import rag_service

    class FakeClient:
        def post(self, path, body, cast_to=None):
            docs = body["documents"]
            results = [
                {"index": i, "relevance_score": 1.0 if "monitor" in doc else 0.0}
                for i, doc in enumerate(docs)
            ]
            return {"results": results}

    monkeypatch.setattr(rag_service, "_get_reranker_client", lambda: FakeClient())
    hits = [
        {"content": "unrelated content", "path": "a.md", "score": 0.9},
        {"content": "monitor expense policy", "path": "b.md", "score": 0.7},
    ]
    out = rag_service._rerank("monitor expense", hits)
    assert out[0]["path"] == "b.md"
    assert "rerank" in out[0]
