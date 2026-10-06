"""Retrieval + refusal metrics. Pure functions, no DB or API calls."""

from __future__ import annotations

from tqdm import tqdm


def _haystack(hit: dict) -> str:
    return " ".join(str(hit.get(k, "")) for k in ("title", "content")).lower()


def _matched(hit: dict, expected_any: list[str], expected_absent: list[str]) -> bool:
    haystack = _haystack(hit)
    hit_any = any(e.lower() in haystack for e in expected_any)
    hit_absent = any(e.lower() in haystack for e in expected_absent)
    return hit_any and not hit_absent


def _first_rank(hits: list[dict], expected_any: list[str], expected_absent: list[str]) -> int | None:
    for rank, h in enumerate(hits, start=1):
        if _matched(h, expected_any, expected_absent):
            return rank
    return None


def _score(results, min_relevance: float) -> dict:
    """Score pre-retrieved results. `results`: iterable of (item, hits, best_score)."""
    per_item: list[dict] = []
    tp = fp = tn = fn = 0
    recall_hits = 0
    recall_n = 0
    mrr_sum = 0.0
    mrr_n = 0
    leaked = 0

    for q, hits, best in results:
        grounded = bool(hits) and best >= min_relevance
        expect = q.get("expect_in_docs", True)
        expected_any = q.get("expected_any") or []
        expected_absent = q.get("expected_absent") or []

        if expect and grounded:
            tp += 1
        elif expect and not grounded:
            fn += 1
        elif not expect and grounded:
            fp += 1
        else:
            tn += 1

        rank = None
        if expect and expected_any:
            recall_n += 1
            rank = _first_rank(hits, expected_any, expected_absent)
            if rank is not None:
                recall_hits += 1
                mrr_n += 1
                mrr_sum += 1.0 / rank

        # For should-refuse questions, `expected_absent` lists answer-signature
        # terms. If retrieval surfaces them anyway, it's a retrieval "leak".
        item_leaked = False
        if not expect and expected_absent:
            item_leaked = any(
                e.lower() in _haystack(h) for h in hits for e in expected_absent
            )
            if item_leaked:
                leaked += 1

        per_item.append({
            "id": q.get("id"),
            "grounded": grounded,
            "expect": expect,
            "rank": rank,
            "leaked": item_leaked,
            "top": [h.get("title", "") for h in hits[:3]],
            "top_scores": [round(float(h.get("score", 0.0)), 3) for h in hits[:3]],
        })

    n = len(results)
    return {
        "n": n,
        "recall@k": (recall_hits / recall_n) if recall_n else 1.0,
        "mrr": (mrr_sum / mrr_n) if mrr_n else 0.0,
        # "refuse" is the positive class for these two: of the refusals the system
        # made, how many were correct; of should-refuse questions, how many caught.
        "refusal_precision": tn / (tn + fn) if (tn + fn) else 0.0,
        "refusal_recall": tn / (tn + fp) if (tn + fp) else 1.0,
        "accuracy": (tp + tn) / n if n else 0.0,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "leaked": leaked,
        "per_item": per_item,
    }


def _retrieve_all(items: list[dict], retrieve_fn):
    results = []
    for q in tqdm(items, desc="eval", unit="question"):
        results.append((q, *retrieve_fn(q["question"])))
    return results


def evaluate(items: list[dict], retrieve_fn, min_relevance: float) -> dict:
    return _score(_retrieve_all(items, retrieve_fn), min_relevance)


def sweep_results(results, thresholds) -> dict:
    """Score already-retrieved `results` at each threshold (no re-embedding)."""
    return {
        t: {k: v for k, v in _score(results, t).items() if k != "per_item"}
        for t in thresholds
    }


def sweep(items: list[dict], retrieve_fn, thresholds) -> dict:
    """Retrieve once per item, then score at each threshold (no re-embedding)."""
    return sweep_results(_retrieve_all(items, retrieve_fn), thresholds)


def best_at_fp(sweep: dict, target_fp: int) -> tuple:
    """Return (threshold, metrics) from a {threshold: metrics} sweep whose
    confusion fp is closest to target_fp; ties broken by higher accuracy, then
    higher threshold (more conservative)."""
    def key(t):
        m = sweep[t]
        return (abs(m["confusion"]["fp"] - target_fp), -m["accuracy"], -t)

    t = min(sweep, key=key)
    return t, sweep[t]
