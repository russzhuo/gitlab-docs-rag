from eval.metrics import evaluate, sweep


def _retriever(results):
    def fn(question):
        return results[question]
    return fn


def test_evaluate_clean():
    results = {
        "pto": ([{"title": "Time Off", "path": "hr/pto.md", "content": "flexible pto", "score": 0.8}], 0.8),
        "salary": ([{"title": "Comp", "content": "nope", "score": 0.1}], 0.1),
    }
    items = [
        {"id": 1, "question": "pto", "expect_in_docs": True, "expected_any": ["pto"], "expected_absent": []},
        {"id": 2, "question": "salary", "expect_in_docs": False},
    ]
    r = evaluate(items, _retriever(results), min_relevance=0.35)
    assert r["recall@k"] == 1.0
    assert r["mrr"] == 1.0
    assert r["confusion"] == {"tp": 1, "fp": 0, "tn": 1, "fn": 0}
    assert r["refusal_precision"] == 1.0
    assert r["refusal_recall"] == 1.0


def test_evaluate_refusal_leak_is_fp():
    results = {"salary": ([{"title": "Comp", "content": "x", "score": 0.6}], 0.6)}
    items = [{"id": 1, "question": "salary", "expect_in_docs": False}]
    r = evaluate(items, _retriever(results), min_relevance=0.35)
    assert r["confusion"]["fp"] == 1
    assert r["refusal_recall"] == 0.0


def test_evaluate_miss_is_fn():
    results = {"pto": ([], 0.0)}
    items = [{"id": 1, "question": "pto", "expect_in_docs": True, "expected_any": ["pto"], "expected_absent": []}]
    r = evaluate(items, _retriever(results), min_relevance=0.35)
    assert r["confusion"]["fn"] == 1
    assert r["recall@k"] == 0.0


def test_expected_absent_excludes_false_positive():
    # a hit contains the "any" term but also an "absent" term -> not a match
    results = {
        "pto": ([{"title": "X", "content": "pto is approved publicly", "score": 0.9}], 0.9),
    }
    items = [{"id": 1, "question": "pto", "expect_in_docs": True,
              "expected_any": ["pto"], "expected_absent": ["publicly"]}]
    r = evaluate(items, _retriever(results), min_relevance=0.35)
    assert r["recall@k"] == 0.0  # matched "pto" but excluded by "publicly"


def test_expected_absent_flags_leak_on_should_refuse():
    # should-refuse question, but top-k hits contain answer-signature terms -> leak
    results = {
        "stock": ([{"title": "Comp", "content": "stock price", "score": 0.2}], 0.2),
    }
    items = [{"id": 1, "question": "stock", "expect_in_docs": False,
              "expected_absent": ["stock price"]}]
    r = evaluate(items, _retriever(results), min_relevance=0.35)
    assert r["leaked"] == 1


def test_sweep_retrieves_once_and_scores():
    calls = {"n": 0}
    results = {
        "pto": ([{"title": "Time Off", "content": "pto", "score": 0.8}], 0.8),
        "salary": ([], 0.0),
    }

    def retriever(question):
        calls["n"] += 1
        return results[question]

    items = [
        {"id": 1, "question": "pto", "expect_in_docs": True, "expected_any": ["pto"], "expected_absent": []},
        {"id": 2, "question": "salary", "expect_in_docs": False},
    ]
    r = sweep(items, retriever, [0.3, 0.5, 0.9])
    assert calls["n"] == 2  # retrieved once per item, not once per threshold
    assert set(r.keys()) == {0.3, 0.5, 0.9}
    assert r[0.5]["confusion"] == {"tp": 1, "fp": 0, "tn": 1, "fn": 0}
    assert r[0.9]["confusion"]["fn"] == 1
