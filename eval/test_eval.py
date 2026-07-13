"""End-to-end eval harness: golden questions vs. the live pipeline.

WHAT THIS MEASURES
------------------
Not "does the code run" (the unit suite covers that) but "does the system
give CORRECT answers": for every item in golden_set.yaml we

1. execute the human-written ``reference_sql`` against the live database to
   get ground truth (never hardcoded — the data loader may be rerun with
   fresher EDGAR data at any time);
2. run the FULL pipeline (retrieval, generation, guard, execution, the
   3-attempt self-correction loop) on the natural-language ``question``;
3. compare the two result sets, order-insensitively, with numeric coercion —
   ``check: relative`` allows 1% per numeric cell, ``check: exact`` requires
   identical sorted rows.

Graceful-failure items (``expect_failure: true``) invert the assertion: the
pipeline must return success=False with a readable explanation instead of
hallucinating an answer.

These tests need a live database and real API keys, so they are opt-in:
they auto-skip unless RUN_EVAL=1 (and CI only sets that when the secrets
exist). The summary test prints an accuracy table and gates on
EVAL_MIN_ACCURACY (default 0.75).
"""

from __future__ import annotations

import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml

# Make `app` importable when pytest is invoked from anywhere (eval/ has no
# __init__.py, so pytest puts eval/ — not the repo root — on sys.path).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(
        os.environ.get("RUN_EVAL") != "1",
        reason="eval requires a live DB + API keys; opt in with RUN_EVAL=1",
    ),
]

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.yaml"
RELATIVE_TOLERANCE = 0.01  # 1% per numeric cell for check: relative

with GOLDEN_SET_PATH.open() as f:
    GOLDEN_ITEMS: list[dict] = yaml.safe_load(f)

# Shared scoreboard: each per-item test records its verdict here, and the
# summary test (which runs last — pytest preserves file order) reads it.
RESULTS: dict[str, bool] = {}


# ---------------------------------------------------------------------------
# Row-set comparison
# ---------------------------------------------------------------------------


def _coerce(value: Any) -> Any:
    """Normalize a cell for comparison: numbers to float, rest to str."""
    if isinstance(value, bool):  # bool is an int subclass — keep it symbolic
        return str(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return str(value)


def _normalize_rows(rows: list[dict]) -> list[tuple]:
    """Rows -> sorted tuples of coerced cell values.

    Column NAMES are ignored on purpose: the pipeline is free to alias
    columns differently than the reference SQL ('value' vs 'revenue') —
    what must match is the data. Cell order within a row follows the
    query's column order, which for like-shaped queries lines up; sorting
    the rows removes ORDER BY differences.
    """
    normalized = [tuple(_coerce(v) for v in row.values()) for row in rows]
    return sorted(normalized, key=repr)


def _cells_match(expected: Any, actual: Any, check: str) -> bool:
    if isinstance(expected, float) and isinstance(actual, float):
        if check == "relative":
            return math.isclose(expected, actual, rel_tol=RELATIVE_TOLERANCE)
        return expected == actual
    return expected == actual


def rowsets_match(expected: list[dict], actual: list[dict], check: str) -> bool:
    """Order-insensitive result-set comparison with per-cell tolerance."""
    exp_rows = _normalize_rows(expected)
    act_rows = _normalize_rows(actual)
    if len(exp_rows) != len(act_rows):
        return False
    return all(
        len(e) == len(a) and all(_cells_match(ec, ac, check) for ec, ac in zip(e, a))
        for e, a in zip(exp_rows, act_rows)
    )


# ---------------------------------------------------------------------------
# Per-item tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("item", GOLDEN_ITEMS, ids=[i["id"] for i in GOLDEN_ITEMS])
def test_golden_item(item: dict) -> None:
    # Imports deferred so collecting (and skipping) this module never touches
    # settings, the DB, or provider SDKs on machines without credentials.
    from app.db import execute_readonly
    from app.pipeline import run_pipeline

    request_id = f"eval-{uuid.uuid4().hex[:12]}"
    response = run_pipeline(item["question"], request_id)

    if item.get("expect_failure"):
        # Graceful failure IS the correct behavior for a nonsense question:
        # no hallucinated rows, success=False, and a real explanation.
        passed = (
            response.success is False
            and response.rows == []
            and len(response.answer) > 20
        )
        RESULTS[item["id"]] = passed
        assert passed, (
            f"expected graceful failure, got success={response.success} "
            f"answer={response.answer!r}"
        )
        return

    expected_rows = execute_readonly(item["reference_sql"])
    assert expected_rows, (
        f"reference_sql for {item['id']} returned no rows — the golden set "
        f"or the loaded data is broken; fix that before blaming the pipeline"
    )

    passed = response.success and rowsets_match(
        expected_rows, response.rows, item.get("check", "relative")
    )
    RESULTS[item["id"]] = passed

    assert passed, (
        f"[{item['id']}] pipeline answer did not match ground truth.\n"
        f"question: {item['question']}\n"
        f"pipeline sql: {response.sql}\n"
        f"pipeline rows ({len(response.rows)}): {response.rows[:5]}\n"
        f"expected rows ({len(expected_rows)}): {expected_rows[:5]}\n"
        f"attempt outcomes: {[a.outcome for a in response.attempts]}"
    )


# ---------------------------------------------------------------------------
# Summary + accuracy gate
# ---------------------------------------------------------------------------


def test_zz_accuracy_summary() -> None:
    """Print the accuracy table and enforce the CI gate.

    Named zz_* so it runs after every per-item test in this file. Items
    whose test errored before recording a verdict count as failures — an
    eval that crashes is not an eval that passed.
    """
    min_accuracy = float(os.environ.get("EVAL_MIN_ACCURACY", "0.75"))

    total = len(GOLDEN_ITEMS)
    passed = sum(1 for item in GOLDEN_ITEMS if RESULTS.get(item["id"], False))
    accuracy = passed / total if total else 0.0

    width = max(len(item["id"]) for item in GOLDEN_ITEMS) + 2
    print("\n\n=== EVAL ACCURACY SUMMARY " + "=" * 40)
    for item in GOLDEN_ITEMS:
        verdict = "PASS" if RESULTS.get(item["id"], False) else "FAIL"
        print(f"  {item['id']:<{width}} {verdict}")
    print("-" * 66)
    print(f"  accuracy: {passed}/{total} = {accuracy:.1%} (gate: {min_accuracy:.0%})")
    print("=" * 66)

    assert accuracy >= min_accuracy, (
        f"accuracy {accuracy:.1%} is below the EVAL_MIN_ACCURACY gate "
        f"of {min_accuracy:.0%}"
    )
