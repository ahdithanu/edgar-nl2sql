"""Measure real latency + token/cost per query over the pipeline.

Runs the FULL pipeline (retrieval -> generate -> guard -> execute -> synthesize,
retries and all) against the live services, once per question, and reports:

    latency p50 / p95 / mean / max   (wall clock, end to end)
    cost    mean / total             (measured tokens x configured rates)
    retry-rate                       (% of queries that needed > 1 attempt)
    failure-rate                     (% with success=False)
    mode split                       (sql vs narrative)

Questions come from the eval sets (eval/golden_set.yaml + holdout_set.yaml) so
the benchmark exercises the same distribution the accuracy gate does. This spends
real API money — one embed + up to 3 Claude calls + synthesis per question.

Usage:
    python scripts/bench.py                 # every eval question, once
    python scripts/bench.py --limit 10      # first 10 questions
    python scripts/bench.py --repeat 4      # 4x the set (bigger N for p95)
    python scripts/bench.py --out bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import metrics  # noqa: E402
from app.pipeline import run_pipeline  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"


def load_questions() -> list[str]:
    questions: list[str] = []
    for name in ("golden_set.yaml", "holdout_set.yaml"):
        path = EVAL_DIR / name
        if not path.exists():
            continue
        for item in yaml.safe_load(path.read_text()) or []:
            q = item.get("question")
            if q:
                questions.append(q)
    # Dedup while preserving order.
    seen: set[str] = set()
    return [q for q in questions if not (q in seen or seen.add(q))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark pipeline latency + cost.")
    parser.add_argument("--limit", type=int, help="Use only the first N unique questions")
    parser.add_argument("--repeat", type=int, default=1, help="Run the whole set this many times")
    parser.add_argument("--out", help="Write per-query records as JSON to this path")
    args = parser.parse_args()

    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]
    questions = questions * args.repeat
    if not questions:
        print("No questions found in eval/.", file=sys.stderr)
        return 1

    print(f"Benchmarking {len(questions)} queries (this spends API money)...\n")
    records: list[dict] = []
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        resp = run_pipeline(q, f"bench-{i}")
        latency_s = time.perf_counter() - t0
        m = metrics.current_metrics()
        rec = {
            "question": q,
            "latency_s": round(latency_s, 3),
            "cost_usd": round(m.est_cost_usd(), 6) if m else None,
            "input_tokens": m.input_tokens if m else None,
            "output_tokens": m.output_tokens if m else None,
            "llm_calls": len(m.llm_calls) if m else None,
            "attempts": len(resp.attempts),
            "success": resp.success,
            "mode": resp.mode,
        }
        records.append(rec)
        print(f"  [{i}/{len(questions)}] {latency_s:5.2f}s  "
              f"${rec['cost_usd'] or 0:.4f}  {resp.mode:9s} "
              f"{'ok ' if resp.success else 'FAIL'} attempts={rec['attempts']}  {q[:48]}")

    latencies = [r["latency_s"] for r in records]
    costs = [r["cost_usd"] or 0.0 for r in records]
    n = len(records)
    retries = sum(1 for r in records if r["attempts"] > 1)
    failures = sum(1 for r in records if not r["success"])
    narrative = sum(1 for r in records if r["mode"] == "narrative")

    print("\n" + "=" * 56)
    print(f"  queries            {n}")
    print(f"  latency p50        {metrics.percentile(latencies, 50):.2f}s")
    print(f"  latency p95        {metrics.percentile(latencies, 95):.2f}s")
    print(f"  latency mean       {sum(latencies)/n:.2f}s")
    print(f"  latency max        {max(latencies):.2f}s")
    print(f"  cost / query mean  ${sum(costs)/n:.4f}")
    print(f"  cost total         ${sum(costs):.4f}")
    print(f"  retry rate         {retries/n*100:.0f}%  ({retries}/{n} needed >1 attempt)")
    print(f"  failure rate       {failures/n*100:.0f}%  ({failures}/{n})")
    print(f"  mode split         {n-narrative} sql / {narrative} narrative")
    print("=" * 56)

    if args.out:
        Path(args.out).write_text(json.dumps(records, indent=2))
        print(f"\nWrote per-query records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
