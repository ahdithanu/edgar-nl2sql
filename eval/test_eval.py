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
exist).

THE BUILD GATE IS AGGREGATE, NOT PER-ITEM. Individual misses print full
diagnostics and record a verdict, but only ``test_zz_accuracy_summary``
raises — it prints the accuracy table and asserts against EVAL_MIN_ACCURACY
(default 0.85, against a measured 100% baseline). Rationale: this pipeline
is LLM-backed and therefore nondeterministic. Failing the build on any
single miss would make the threshold decorative and would block deploys on
sampling noise; gating on the aggregate catches real quality regressions
while tolerating roughly two unlucky rolls out of seventeen.
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
HOLDOUT_SET_PATH = Path(__file__).parent / "holdout_set.yaml"
RELATIVE_TOLERANCE = 0.01  # 1% per numeric cell for check: relative


def _load(path: Path, split: str) -> list[dict]:
    with path.open() as f:
        items = yaml.safe_load(f)
    for item in items:
        item["_split"] = split
    return items


# Two splits, kept separate on purpose. `dev` questions were authored alongside
# the prompt and corpus, so their accuracy reflects fitting; `holdout` questions
# were written afterward and never used to tune anything, so their accuracy is
# the honest generalization number. The summary reports each split separately;
# blending them would hide the tuning.
GOLDEN_ITEMS: list[dict] = _load(GOLDEN_SET_PATH, "dev")
HOLDOUT_ITEMS: list[dict] = _load(HOLDOUT_SET_PATH, "holdout")
ALL_ITEMS: list[dict] = GOLDEN_ITEMS + HOLDOUT_ITEMS

# Shared scoreboard: each per-item test records its verdict here, and the
# summary test (which runs last — pytest preserves file order) reads it.
RESULTS: dict[str, bool] = {}
SPLIT_OF: dict[str, str] = {item["id"]: item["_split"] for item in ALL_ITEMS}

# Diagnostics for misses, keyed by item id. These are printed by the SUMMARY
# test rather than by the per-item test, because pytest captures stdout for
# passing tests — and per-item tests now pass by design (the gate is aggregate).
# Printing from the failing summary test is what makes a CI failure debuggable
# without re-running anything.
DIAGNOSTICS: dict[str, str] = {}


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


def _row_contains(expected_row: tuple, actual_row: tuple, check: str) -> bool:
    """True if every expected cell matches a DISTINCT cell of the actual row.

    Containment, not equality: the pipeline legitimately returns richer rows
    than the minimal reference SQL (e.g. reference selects just `ticker`,
    the model also includes `name` and the metric value). Requiring identical
    column counts would fail correct answers for being more informative.
    """
    remaining = list(actual_row)
    for exp in expected_row:
        for i, act in enumerate(remaining):
            if _cells_match(exp, act, check):
                del remaining[i]
                break
        else:
            return False
    return True


def rowsets_match(expected: list[dict], actual: list[dict], check: str) -> bool:
    """Order-insensitive result-set comparison with per-cell tolerance.

    Row counts must match exactly (a correct answer has no missing or extra
    rows); within a row, the pipeline may return a superset of the reference
    columns (see _row_contains). Rows pair greedily, each used at most once.
    """
    exp_rows = _normalize_rows(expected)
    act_rows = _normalize_rows(actual)
    if len(exp_rows) != len(act_rows):
        return False
    unused = list(act_rows)
    for e in exp_rows:
        for i, a in enumerate(unused):
            if _row_contains(e, a, check):
                del unused[i]
                break
        else:
            return False
    return True


# ---------------------------------------------------------------------------
# Per-item tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("item", ALL_ITEMS, ids=[i["id"] for i in ALL_ITEMS])
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
        if not passed:
            # Recorded, not raised — see the rationale on the main path below.
            DIAGNOSTICS[item["id"]] = (
                f"expected graceful failure, got success={response.success}\n"
                f"      answer: {response.answer!r}\n"
                f"      attempt outcomes: {[a.outcome for a in response.attempts]}"
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

    # A per-item miss is REPORTED, not raised. WHY: the build gate is aggregate
    # accuracy (test_zz_accuracy_summary), exactly as specified. Hard-failing
    # here would make that threshold decorative — any single miss would fail CI
    # no matter how high accuracy was — and this pipeline is LLM-backed, so one
    # unlucky sample must not block a deploy. A genuine regression moves the
    # aggregate and trips the gate; noise does not. Full diagnostics still print
    # so a miss is never silent.
    if not passed:
        last_error = next(
            (a.error_message for a in reversed(response.attempts) if a.error_message),
            None,
        )
        DIAGNOSTICS[item["id"]] = (
            f"question: {item['question']}\n"
            f"      pipeline success: {response.success}\n"
            f"      pipeline sql: {response.sql}\n"
            f"      pipeline rows ({len(response.rows)}): {response.rows[:3]}\n"
            f"      expected rows ({len(expected_rows)}): {expected_rows[:3]}\n"
            f"      attempt outcomes: {[a.outcome for a in response.attempts]}\n"
            f"      last error: {last_error}"
        )


# ---------------------------------------------------------------------------
# Summary + accuracy gate
# ---------------------------------------------------------------------------


def _split_accuracy(split: str) -> tuple[int, int]:
    items = [i for i in ALL_ITEMS if i["_split"] == split]
    passed = sum(1 for i in items if RESULTS.get(i["id"], False))
    return passed, len(items)


def test_zz_accuracy_summary() -> None:
    """Print the per-split accuracy table and enforce the CI gate.

    Named zz_* so it runs after every per-item test in this file. Items
    whose test errored before recording a verdict count as failures — an
    eval that crashes is not an eval that passed.

    The gate is on the HELD-OUT split, because that is the honest measure of
    whether the system generalizes past the questions it was tuned on. Dev
    accuracy is reported for context but not gated (gating on the tuned set
    would reward overfitting). EVAL_MIN_ACCURACY sets the held-out floor.
    """
    min_accuracy = float(os.environ.get("EVAL_MIN_ACCURACY", "0.85"))

    width = max(len(i["id"]) for i in ALL_ITEMS) + 2
    print("\n\n=== EVAL ACCURACY SUMMARY " + "=" * 40)
    for split in ("dev", "holdout"):
        print(f"\n  [{split}]")
        for item in ALL_ITEMS:
            if item["_split"] != split:
                continue
            verdict = "PASS" if RESULTS.get(item["id"], False) else "FAIL"
            print(f"    {item['id']:<{width}} {verdict}")

    dev_p, dev_n = _split_accuracy("dev")
    hold_p, hold_n = _split_accuracy("holdout")
    dev_acc = dev_p / dev_n if dev_n else 0.0
    hold_acc = hold_p / hold_n if hold_n else 0.0
    print("-" * 66)
    print(f"  dev      : {dev_p}/{dev_n} = {dev_acc:.1%}  (reported, not gated)")
    print(f"  held-out : {hold_p}/{hold_n} = {hold_acc:.1%}  (gate: >= {min_accuracy:.0%})")
    print("=" * 66)

    # Print every miss's detail HERE, from the test that actually fails, so the
    # reason is visible in CI logs without a re-run. (pytest swallows stdout
    # from the per-item tests because they pass by design.)
    if DIAGNOSTICS:
        print("\n--- MISS DETAILS " + "-" * 49)
        for item_id, detail in DIAGNOSTICS.items():
            split = SPLIT_OF.get(item_id, "?")
            print(f"\n  [{split}] [{item_id}] {detail}")
        print("-" * 66)

    assert hold_acc >= min_accuracy, (
        f"held-out accuracy {hold_acc:.1%} ({hold_p}/{hold_n}) is below the "
        f"EVAL_MIN_ACCURACY gate of {min_accuracy:.0%}"
    )
