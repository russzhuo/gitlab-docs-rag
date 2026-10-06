"""Drive evaluate() against the live DB + embeddings.

Usage:
    python -m eval.run_eval                 # table + summary at current MIN_RELEVANCE
    python -m eval.run_eval --json          # machine-readable
    python -m eval.run_eval --sweep         # scan thresholds to pick MIN_RELEVANCE
    python -m eval.run_eval --compare       # A/B rerank on vs off (recall/MRR delta)
    python -m eval.run_eval --compare-grounding  # cosine vs rerank-score grounding, matched-fp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from app.config import MIN_RELEVANCE, RERANK_MIN_RELEVANCE, RETRIEVAL_MODE, TOP_K
from app.database import SessionLocal
from app.services.rag_service import retrieve
from eval.metrics import best_at_fp, evaluate, sweep, sweep_results

QUESTIONS = Path(__file__).with_name("questions.json")


def _retrieve(question: str):
    with SessionLocal() as db:
        return retrieve(db, question)


def _retrieve_with(rerank: bool):
    def fn(question: str):
        with SessionLocal() as db:
            return retrieve(db, question, rerank=rerank)
    return fn


def _retrieve_rerank_signal(question: str):
    """Retrieve with rerank on; grounding signal is the top-1 rerank score."""
    with SessionLocal() as db:
        hits, _ = retrieve(db, question, rerank=True)
        signal = hits[0]["rerank"] if hits else 0.0
        return hits, signal


def _collect(items: list[dict], retrieve_fn):
    return [(q, *retrieve_fn(q["question"])) for q in tqdm(items, desc="retrieve", unit="question")]


def _grid(signals: list[float], n: int = 120) -> list[float]:
    """Threshold grid spanning the observed signal range (+5% padding), so the
    sweep is scale-agnostic whether the signal is cosine or a rerank score."""
    lo, hi = min(signals), max(signals)
    pad = (hi - lo) * 0.05 if hi > lo else 0.1
    lo, hi = lo - pad, hi + pad
    return [round(lo + (hi - lo) * i / (n - 1), 6) for i in range(n)]


def _print_summary(result: dict) -> None:
    c = result["confusion"]
    print(f"recall@{TOP_K}: {result['recall@k']:.3f}   MRR: {result['mrr']:.3f}")
    print(f"refusal precision: {result['refusal_precision']:.3f}   "
          f"refusal recall: {result['refusal_recall']:.3f}   accuracy: {result['accuracy']:.3f}")
    print(f"answered/should-answer={c['tp']}  refused/should-answer={c['fn']}  "
          f"answered/should-refuse={c['fp']}  refused/should-refuse={c['tn']}")
    print(f"retrieval leaks: {result['leaked']}")


def _print_compare(no_rerank: dict, rerank: dict) -> None:
    print(f"{'metric':>18} {'no-rerank':>10} {'rerank':>10} {'delta':>9}")
    for m in ("recall@k", "mrr", "refusal_precision", "refusal_recall", "accuracy"):
        a, b = no_rerank[m], rerank[m]
        print(f"{m:>18} {a:>10.3f} {b:>10.3f} {b - a:>+9.3f}")
    ca, cb = no_rerank["confusion"], rerank["confusion"]
    ca_s = f"{ca['tp']}/{ca['fp']}/{ca['tn']}/{ca['fn']}"
    cb_s = f"{cb['tp']}/{cb['fp']}/{cb['tn']}/{cb['fn']}"
    print(f"{'confusion tp/fp/tn/fn':>18} {ca_s:>10} {cb_s:>10}")
    la, lb = no_rerank["leaked"], rerank["leaked"]
    print(f"{'leaks':>18} {la:>10} {lb:>10} {lb - la:>+9}")


def _print_compare_grounding(cos_sweep: dict, rr_sweep: dict,
                             cos_signals: list[float], rr_signals: list[float]) -> None:
    def rng(s):
        return f"{min(s):.3f}..{max(s):.3f}"

    def row(name, t, m):
        c = m["confusion"]
        return (f"{name:>7} thr={t:>6.3f} fp={c['fp']:<2} fn={c['fn']:<2} "
                f"acc={m['accuracy']:.3f} recall@k={m['recall@k']:.3f} mrr={m['mrr']:.3f}")

    print(f"grounding signal range — cosine: {rng(cos_signals)}   rerank: {rng(rr_signals)}")

    print("\nbest-accuracy operating point:")
    for name, sweep in (("cosine", cos_sweep), ("rerank", rr_sweep)):
        t = max(sweep, key=lambda t: sweep[t]["accuracy"])
        print("  " + row(name, t, sweep[t]))

    print("\nzero-leak (fp=0) operating point:")
    for name, sweep in (("cosine", cos_sweep), ("rerank", rr_sweep)):
        t, m = best_at_fp(sweep, 0)
        print("  " + row(name, t, m))

    max_fp = max(
        max(m["confusion"]["fp"] for m in cos_sweep.values()),
        max(m["confusion"]["fp"] for m in rr_sweep.values()),
    )
    print(f"\nmatched-fp table (fp=leaks; fn=valid wrongly refused):")
    print(f"{'fp':>3} | {'cosine: thr  fn  acc  rec@k  mrr':>34} | {'rerank: thr  fn  acc  rec@k  mrr':>34}")
    for fp in range(max_fp + 1):
        tc, mc = best_at_fp(cos_sweep, fp)
        tr, mr = best_at_fp(rr_sweep, fp)
        cc, cr = mc["confusion"], mr["confusion"]
        left = f"{tc:>6.3f} {cc['fn']:>3} {mc['accuracy']:.3f} {mc['recall@k']:.3f} {mc['mrr']:.3f}"
        right = f"{tr:>6.3f} {cr['fn']:>3} {mr['accuracy']:.3f} {mr['recall@k']:.3f} {mr['mrr']:.3f}"
        print(f"{fp:>3} | {left:>34} | {right:>34}")

    print("\nAt the same fp, prefer lower fn and higher recall@k/mrr.")


def _print_sweep(results: dict) -> None:
    print(f"{'min_rel':>7} {'recall@k':>9} {'mrr':>6} {'refP':>6} {'refR':>6} "
          f"{'acc':>6}  {'tp':>2} {'fp':>2} {'tn':>2} {'fn':>2}")
    best_t, best_acc = None, -1.0
    zero_fp_t = None
    for t in sorted(results):
        r = results[t]
        c = r["confusion"]
        print(f"{t:>7.2f} {r['recall@k']:>9.3f} {r['mrr']:>6.3f} "
              f"{r['refusal_precision']:>6.3f} {r['refusal_recall']:>6.3f} "
              f"{r['accuracy']:>6.3f}  {c['tp']:>2} {c['fp']:>2} {c['tn']:>2} {c['fn']:>2}")
        if r["accuracy"] > best_acc:
            best_acc, best_t = r["accuracy"], t
        if c["fp"] == 0 and zero_fp_t is None:
            zero_fp_t = t
    print(f"\nbest accuracy: {best_acc:.3f} at MIN_RELEVANCE={best_t:.2f}")
    if zero_fp_t is not None:
        print(f"lowest threshold with zero dangerous leaks (fp=0): {zero_fp_t:.2f}")
    print("fp = answered when it should refuse (the dangerous leak).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--sweep", action="store_true", help="scan MIN_RELEVANCE values")
    parser.add_argument("--compare", action="store_true", help="A/B compare rerank on vs off")
    parser.add_argument("--compare-grounding", action="store_true",
                        help="A/B compare cosine vs rerank-score grounding at matched fp")
    parser.add_argument("--min", type=float, default=0.30)
    parser.add_argument("--max", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    items = json.loads(QUESTIONS.read_text())

    if args.compare:
        no_rerank = evaluate(items, _retrieve_with(False), MIN_RELEVANCE)
        rerank = evaluate(items, _retrieve_with(True), RERANK_MIN_RELEVANCE)
        if args.json:
            print(json.dumps({"no_rerank": no_rerank, "rerank": rerank}, indent=2))
        else:
            _print_compare(no_rerank, rerank)
        return

    if args.compare_grounding:
        cos_results = _collect(items, _retrieve_with(False))
        rr_results = _collect(items, _retrieve_rerank_signal)
        cos_signals = [best for _, _, best in cos_results]
        rr_signals = [best for _, _, best in rr_results]
        cos_sweep = sweep_results(cos_results, _grid(cos_signals))
        rr_sweep = sweep_results(rr_results, _grid(rr_signals))
        if args.json:
            print(json.dumps({"cosine": cos_sweep, "rerank": rr_sweep}, indent=2))
        else:
            _print_compare_grounding(cos_sweep, rr_sweep, cos_signals, rr_signals)
        return

    if args.sweep:
        thresholds = []
        t = args.min
        while t <= args.max + 1e-9:
            thresholds.append(round(t, 3))
            t += args.step
        results = sweep(items, _retrieve, thresholds)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            _print_sweep(results)
        return

    result = evaluate(items, _retrieve, MIN_RELEVANCE)
    if args.json:
        print(json.dumps(result, indent=2))
        return

    for it in result["per_item"]:
        verdict = "PASS" if it["grounded"] == it["expect"] else "FAIL"
        rank = it["rank"] if it["rank"] is not None else "-"
        tops = ", ".join(it["top"]) or "-"
        print(f"{verdict}  q{it['id']}  grounded={it['grounded']}  rank={rank}  {tops}")

    print(f"\n--- summary (mode={RETRIEVAL_MODE}, k={TOP_K}, min_relevance={MIN_RELEVANCE}) ---")
    _print_summary(result)


if __name__ == "__main__":
    main()
