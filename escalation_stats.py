"""
escalation_stats.py — ChainIQ START Hack 2026
==============================================
Compares our engine's escalation decisions against historical_awards.csv
and produces a full confusion-matrix analysis.

Confusion matrix (our decision vs. history):

                     History: escalated   History: not escalated
  We: escalated      True Positive (TP)   False Positive (FP)
  We: not escalated  False Negative (FN)  True Negative (TN)

Derived metrics: Precision, Recall, F1, Accuracy, Specificity.
Breakdowns: FPs by rule fired, FNs by historical escalation target.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from csv import DictReader
from pathlib import Path

from supplier_engine import SupplierEngine, _load_json, DATA_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_awards(data_dir: Path) -> dict[str, list[dict]]:
    with open(data_dir / "historical_awards.csv", encoding="utf-8") as f:
        rows = list(DictReader(f))
    index: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        index[row["request_id"]].append(row)
    return dict(index)


def awarded_row(rows: list[dict]) -> dict | None:
    winners = [r for r in rows if r.get("awarded", "").strip().lower() == "true"]
    return winners[0] if winners else None


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100*n/d:.1f}%)" if d else "N/A"


def _safe_div(n: int, d: int) -> float:
    return n / d if d else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_dir: Path = DATA_DIR) -> None:
    print("Running engine on all requests...")
    engine   = SupplierEngine(data_dir)
    requests = _load_json(data_dir / "requests.json")
    awards   = load_awards(data_dir)

    req_lookup = {r["request_id"]: r for r in requests}
    comparable = sorted(set(req_lookup) & set(awards))

    # ── Classify each comparable request ─────────────────────────────────────
    # Each cell also carries metadata for the breakdown sections below.
    tp_rows: list[dict] = []  # we escalated, history escalated
    fp_rows: list[dict] = []  # we escalated, history did NOT escalate
    fn_rows: list[dict] = []  # we did NOT escalate, history DID escalate
    tn_rows: list[dict] = []  # neither escalated

    for rid in comparable:
        result  = engine.process(req_lookup[rid])
        winner  = awarded_row(awards[rid])
        if winner is None:
            continue  # no awarded row → skip

        we_escalated   = len(result.get("escalations", [])) > 0
        hist_escalated = winner["escalation_required"].strip().lower() == "true"
        hist_target    = winner.get("escalated_to", "").strip()

        our_rules    = [e["rule"]        for e in result.get("escalations", [])]
        our_targets  = [e["escalate_to"] for e in result.get("escalations", [])]
        our_blocking = [e for e in result.get("escalations", []) if e.get("blocking")]

        req = req_lookup[rid]
        meta = {
            "request_id":    rid,
            "category":      f"{req['category_l2']} / {req['country']}",
            "our_status":    result["recommendation"]["status"],
            "our_rules":     our_rules,
            "our_targets":   our_targets,
            "our_blocking":  [e["rule"] for e in our_blocking],
            "hist_target":   hist_target,
        }

        if we_escalated and hist_escalated:
            tp_rows.append(meta)
        elif we_escalated and not hist_escalated:
            fp_rows.append(meta)
        elif not we_escalated and hist_escalated:
            fn_rows.append(meta)
        else:
            tn_rows.append(meta)

    tp, fp, fn, tn = len(tp_rows), len(fp_rows), len(fn_rows), len(tn_rows)
    total = tp + fp + fn + tn

    precision   = _safe_div(tp, tp + fp)
    recall      = _safe_div(tp, tp + fn)
    f1          = _safe_div(2 * precision * recall, precision + recall)
    accuracy    = _safe_div(tp + tn, total)
    specificity = _safe_div(tn, tn + fp)

    # ── Confusion matrix ─────────────────────────────────────────────────────
    w = 28  # column width
    print(f"\n{'='*62}")
    print("ESCALATION CONFUSION MATRIX")
    print(f"{'='*62}")
    print(f"  {'':30} {'Hist: escalated':>16}  {'Hist: not escalated':>20}")
    print(f"  {'We: escalated':<30} {'TP = ' + str(tp):>16}  {'FP = ' + str(fp):>20}")
    print(f"  {'We: not escalated':<30} {'FN = ' + str(fn):>16}  {'TN = ' + str(tn):>20}")
    print(f"\n  Total comparable requests: {total}")

    # ── Derived metrics ───────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("DERIVED METRICS")
    print(f"{'='*62}")
    print(f"  Accuracy    (TP+TN) / total         : {_pct(tp+tn, total)}")
    print(f"  Precision   TP / (TP+FP)            : {_pct(tp, tp+fp)}  ← of our escalations, % justified")
    print(f"  Recall      TP / (TP+FN)            : {_pct(tp, tp+fn)}  ← of hist. escalations, % we caught")
    print(f"  Specificity TN / (TN+FP)            : {_pct(tn, tn+fp)}  ← of clean awards, % we agreed")
    print(f"  F1 score    2·P·R / (P+R)           : {f1:.3f}")

    # ── FP breakdown ─────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"FALSE POSITIVES ({fp})  — we escalated, history awarded cleanly")
    print(f"{'='*62}")
    fp_rule_counts: Counter = Counter()
    fp_blocking_counts: Counter = Counter()
    for row in fp_rows:
        for r in row["our_rules"]:
            fp_rule_counts[r] += 1
        for r in row["our_blocking"]:
            fp_blocking_counts[r] += 1

    print("  Rules fired in FP cases (all escalations):")
    for rule, count in fp_rule_counts.most_common():
        print(f"    {rule:<12} {count:>4}×")
    if fp_blocking_counts:
        print("  Of which blocking (cause cannot_proceed / needs_clarification):")
        for rule, count in fp_blocking_counts.most_common():
            print(f"    {rule:<12} {count:>4}×  ← most impactful to fix")

    # Sample FP cases
    print(f"\n  Sample FP cases:")
    for row in fp_rows[:10]:
        blocking_flag = f"  [blocking: {', '.join(row['our_blocking'])}]" if row["our_blocking"] else ""
        print(f"    {row['request_id']}  [{row['category']}]  status={row['our_status']}")
        print(f"      rules={row['our_rules']}{blocking_flag}")

    # ── FN breakdown ─────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"FALSE NEGATIVES ({fn})  — history escalated, we did not")
    print(f"{'='*62}")
    fn_target_counts: Counter = Counter(
        row["hist_target"] for row in fn_rows if row["hist_target"]
    )
    print("  Historical escalation targets we missed:")
    for target, count in fn_target_counts.most_common():
        print(f"    {target:<40} {count:>4}×")

    print(f"\n  Sample FN cases:")
    for row in fn_rows[:10]:
        print(f"    {row['request_id']}  [{row['category']}]  status={row['our_status']}")
        print(f"      hist escalated to: '{row['hist_target']}'")

    # ── TP summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"TRUE POSITIVES ({tp})  — both escalated (agreement)")
    print(f"{'='*62}")
    tp_target_match = sum(
        1 for row in tp_rows
        if row["hist_target"] and row["hist_target"] in row["our_targets"]
    )
    print(f"  Of these, escalation target also matches: {_pct(tp_target_match, tp)}")
    tp_rule_counts: Counter = Counter()
    for row in tp_rows:
        for r in row["our_rules"]:
            tp_rule_counts[r] += 1
    print("  Rules most commonly correct:")
    for rule, count in tp_rule_counts.most_common(5):
        print(f"    {rule:<12} {count:>4}×")

    # ── TN summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"TRUE NEGATIVES ({tn})  — neither escalated (clean agreement)")
    print(f"{'='*62}")
    tn_statuses: Counter = Counter(row["our_status"] for row in tn_rows)
    print("  Our recommendation statuses in TN cases:")
    for status, count in tn_statuses.most_common():
        print(f"    {status:<30} {count:>4}×")

    # ── Write JSON ────────────────────────────────────────────────────────────
    report = {
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "total": total},
        "metrics": {
            "accuracy":    round(accuracy,    3),
            "precision":   round(precision,   3),
            "recall":      round(recall,       3),
            "specificity": round(specificity, 3),
            "f1":          round(f1,           3),
        },
        "fp_rules":    dict(fp_rule_counts.most_common()),
        "fp_blocking": dict(fp_blocking_counts.most_common()),
        "fn_targets":  dict(fn_target_counts.most_common()),
        "fp_cases": fp_rows,
        "fn_cases": fn_rows,
        "tp_cases": tp_rows,
        "tn_cases": tn_rows,
    }
    out = Path(__file__).parent / "escalation_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report → {out}")


if __name__ == "__main__":
    run()
