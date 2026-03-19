"""
fit_scoring_weights.py — ChainIQ START Hack 2026
=================================================
Learns the scoring weights for supplier_engine.py by fitting a pairwise
logistic regression against historical_awards.csv.

Approach — pairwise ranking (Bradley-Terry style):
  For every request where the awarded supplier is in our candidate set,
  create one training sample per (awarded, competitor) pair:

      diff = features(awarded) - features(competitor)

  P(awarded beats competitor) = sigmoid(w · diff)

  Fitting logistic regression on these diff vectors gives weights w where:
    - positive coeff → awarded tends to have HIGHER value of that feature
      (e.g. quality, is_preferred) → subtract this feature in the score function
    - negative coeff → awarded tends to have LOWER value
      (e.g. price, risk_score) → add this feature in the score function

Features used (all per-supplier, computed against the actual request):
  price_ratio     total_price / budget  (relative price pressure)
  risk_score      0–100
  quality_score   0–100
  esg_score       0–100
  is_preferred    0/1
  is_incumbent    0/1
  is_mentioned    0/1

Outputs:
  - Console: coefficients, ranking accuracy before/after
  - scoring_weights.json: fitted weights ready to drop into supplier_engine.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from supplier_engine import SupplierEngine, _load_json, DATA_DIR
from escalation_stats import load_awards, awarded_row

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

FEATURES = ["price_ratio", "risk_score", "quality_score", "esg_score",
            "is_preferred", "is_incumbent", "is_mentioned"]


def extract_features(sup: dict, req: dict) -> dict[str, float]:
    p         = sup.get("pricing") or {}
    qty       = req.get("quantity") or 1
    budget    = req.get("budget_amount") or 1
    unit_price = float(p["unit_price"]) if p.get("unit_price") else None
    total      = unit_price * qty if unit_price is not None else None

    return {
        "price_ratio":   (total / budget) if total is not None else 1.0,
        "risk_score":    float(sup.get("risk_score")    or 50),
        "quality_score": float(sup.get("quality_score") or 50),
        "esg_score":     float(sup.get("esg_score")     or 50),
        "is_preferred":  float(sup.get("preferred", False)),
        "is_incumbent":  float(sup.get("incumbent", False)),
        "is_mentioned":  float(
            bool(req.get("preferred_supplier_mentioned") and
                 req["preferred_supplier_mentioned"].lower() in sup["supplier_name"].lower())
        ),
    }


def to_vec(f: dict[str, float]) -> np.ndarray:
    return np.array([f[k] for k in FEATURES])


# ---------------------------------------------------------------------------
# Build pairwise dataset
# ---------------------------------------------------------------------------

def build_dataset(
    engine: SupplierEngine,
    requests: list[dict],
    awards: dict[str, list[dict]],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Returns X (pairwise diff vectors), y (all ones — awarded beats competitor),
    and metadata for each pair.
    """
    req_lookup = {r["request_id"]: r for r in requests}
    X_rows: list[np.ndarray] = []
    meta:   list[dict]       = []
    skipped = 0

    for rid in sorted(set(req_lookup) & set(awards)):
        req     = req_lookup[rid]
        result  = engine.process(req)
        winner  = awarded_row(awards[rid])
        if not winner:
            continue

        shortlist = result.get("supplier_shortlist", [])
        awarded_id = winner["supplier_id"]

        awarded_sup = next((s for s in shortlist if s["supplier_id"] == awarded_id), None)
        if awarded_sup is None:
            skipped += 1
            continue

        f_awarded = to_vec(extract_features(awarded_sup, req))

        competitors = [s for s in shortlist if s["supplier_id"] != awarded_id]
        if not competitors:
            continue

        for comp in competitors:
            f_comp = to_vec(extract_features(comp, req))
            diff   = f_awarded - f_comp
            X_rows.append(diff)
            meta.append({
                "request_id":      rid,
                "awarded_id":      awarded_id,
                "awarded_name":    awarded_sup["supplier_name"],
                "competitor_id":   comp["supplier_id"],
                "competitor_name": comp["supplier_name"],
            })

    X = np.array(X_rows)
    y = np.ones(len(X_rows), dtype=int)   # awarded always wins (label = 1)
    print(f"  Pairwise samples built : {len(X_rows)}  ({skipped} requests skipped — awarded supplier not in our shortlist)")
    return X, y, meta


# ---------------------------------------------------------------------------
# Ranking accuracy helpers
# ---------------------------------------------------------------------------

def ranking_accuracy(
    engine: SupplierEngine,
    requests: list[dict],
    awards: dict[str, list[dict]],
    weights: dict[str, float] | None = None,
) -> tuple[int, int]:
    """
    Returns (# requests where rank-1 == awarded, # comparable requests).
    If weights is None, uses the engine's current hardcoded scoring.
    If weights is provided, re-scores the shortlist with those weights.
    """
    req_lookup = {r["request_id"]: r for r in requests}
    correct, total = 0, 0

    for rid in sorted(set(req_lookup) & set(awards)):
        req    = req_lookup[rid]
        result = engine.process(req)
        winner = awarded_row(awards[rid])
        if not winner:
            continue

        shortlist  = result.get("supplier_shortlist", [])
        awarded_id = winner["supplier_id"]
        if not any(s["supplier_id"] == awarded_id for s in shortlist):
            continue

        total += 1

        if weights is None:
            rank1 = shortlist[0]["supplier_id"] if shortlist else None
        else:
            # Re-rank with fitted weights
            def rescore(sup: dict) -> float:
                f = extract_features(sup, req)
                # Lower is better — flip signs so: price/risk add, quality/esg/flags subtract
                return (
                    f["price_ratio"]  * weights["price_ratio"]
                    + f["risk_score"]    * weights["risk_score"]
                    - f["quality_score"] * weights["quality_score"]
                    - f["esg_score"]     * weights["esg_score"]
                    - f["is_preferred"]  * weights["is_preferred"]
                    - f["is_incumbent"]  * weights["is_incumbent"]
                    - f["is_mentioned"]  * weights["is_mentioned"]
                )
            rank1 = min(shortlist, key=rescore)["supplier_id"]

        if rank1 == awarded_id:
            correct += 1

    return correct, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_dir: Path = DATA_DIR) -> None:
    print("Loading data and running engine...")
    engine   = SupplierEngine(data_dir)
    requests = _load_json(data_dir / "requests.json")
    awards   = load_awards(data_dir)

    # ── Baseline ranking accuracy (current hardcoded weights) ─────────────
    print("\nComputing baseline ranking accuracy (current hardcoded weights)...")
    base_correct, base_total = ranking_accuracy(engine, requests, awards)
    print(f"  Baseline: {base_correct}/{base_total} ({100*base_correct/base_total:.1f}%)")

    # ── Build pairwise dataset ────────────────────────────────────────────
    print("\nBuilding pairwise training dataset...")
    X, y, meta = build_dataset(engine, requests, awards)

    # ── Fit logistic regression on raw (unscaled) diffs ──────────────────
    # We fit on both the diff and its negation (competitor beats awarded = 0)
    # so the model sees balanced classes. This is equivalent to fitting with
    # the convention that all labels are 1 but using the symmetric loss.
    X_aug = np.vstack([X, -X])
    y_aug = np.concatenate([np.ones(len(X)), np.zeros(len(X))]).astype(int)

    # Scale features so coefficients are comparable across features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_aug)

    print("\nFitting logistic regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        C=1.0,
    )
    model.fit(X_scaled, y_aug)

    # Cross-validation accuracy
    cv_scores = cross_val_score(model, X_scaled, y_aug, cv=5, scoring="accuracy")
    print(f"  5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Report coefficients ───────────────────────────────────────────────
    coef        = model.coef_[0]           # in standardised space
    feature_std = scaler.scale_[:len(FEATURES)]  # std of each feature (from X_aug)
    # Denormalise: coef_raw[i] = coef_scaled[i] / std[i]
    coef_raw = coef / feature_std

    print("\n" + "=" * 62)
    print("FITTED COEFFICIENTS  (positive = awarded has higher value)")
    print("=" * 62)
    print(f"  {'Feature':<20} {'Scaled coef':>12}  {'Raw coef':>12}  Interpretation")
    print(f"  {'-'*20}  {'-'*11}  {'-'*11}  {'-'*25}")
    interp = {
        "price_ratio":   "lower price → awarded",
        "risk_score":    "lower risk  → awarded",
        "quality_score": "higher qual → awarded",
        "esg_score":     "higher esg  → awarded",
        "is_preferred":  "preferred   → awarded",
        "is_incumbent":  "incumbent   → awarded",
        "is_mentioned":  "mentioned   → awarded",
    }
    for feat, sc, rc in zip(FEATURES, coef, coef_raw):
        sign = "✓" if (
            (feat in ("price_ratio", "risk_score") and rc < 0) or
            (feat not in ("price_ratio", "risk_score") and rc > 0)
        ) else "✗ unexpected"
        print(f"  {feat:<20} {sc:>+12.4f}  {rc:>+12.4f}  {interp[feat]}  {sign}")

    # ── Convert to score-function weights ─────────────────────────────────
    # score = price_ratio * w_p + risk * w_r - quality * w_q - esg * w_e
    #         - is_preferred * w_pref - is_incumbent * w_inc - is_mentioned * w_ment
    # Lower score = better.  We flip sign for features where coef is negative
    # (i.e. awarded has lower value → should add to score to penalise higher values).
    # We normalise so the largest absolute weight = 1.0, then scale to readable values.
    raw_weights = {
        "price_ratio":   abs(coef_raw[FEATURES.index("price_ratio")]),
        "risk_score":    abs(coef_raw[FEATURES.index("risk_score")]),
        "quality_score": abs(coef_raw[FEATURES.index("quality_score")]),
        "esg_score":     abs(coef_raw[FEATURES.index("esg_score")]),
        "is_preferred":  abs(coef_raw[FEATURES.index("is_preferred")]),
        "is_incumbent":  abs(coef_raw[FEATURES.index("is_incumbent")]),
        "is_mentioned":  abs(coef_raw[FEATURES.index("is_mentioned")]),
    }
    max_w = max(raw_weights.values())
    norm_weights = {k: round(v / max_w, 4) for k, v in raw_weights.items()}

    print("\n" + "=" * 62)
    print("NORMALISED WEIGHTS  (largest = 1.0)")
    print("=" * 62)
    for feat, w in sorted(norm_weights.items(), key=lambda x: -x[1]):
        print(f"  {feat:<20}  {w:.4f}")

    # ── Ranking accuracy with fitted weights ──────────────────────────────
    print("\nComputing ranking accuracy with fitted weights...")
    fit_correct, fit_total = ranking_accuracy(engine, requests, awards, weights=norm_weights)
    print(f"  Fitted weights: {fit_correct}/{fit_total} ({100*fit_correct/fit_total:.1f}%)")
    print(f"  vs baseline:    {base_correct}/{base_total} ({100*base_correct/base_total:.1f}%)")
    delta = fit_correct - base_correct
    print(f"  Δ correct:      {delta:+d} requests")

    # ── Translate to supplier_engine.py equivalents ───────────────────────
    # Current engine uses absolute price (total_price_eur), not price_ratio.
    # We express price_ratio weight in terms of a "per-EUR" multiplier by
    # noting price_ratio = total / budget, so w_price_ratio / budget ≈ w_total.
    # We report both so the user can decide.
    avg_budget = np.mean([
        float(r["budget_amount"])
        for r in requests
        if r.get("budget_amount")
    ])
    print(f"\n  Average budget across all requests: EUR {avg_budget:,.0f}")
    print(f"  To translate price_ratio weight to absolute price weight:")
    print(f"  w_total ≈ w_price_ratio / avg_budget = {norm_weights['price_ratio']:.4f} / {avg_budget:,.0f}")
    print(f"         ≈ {norm_weights['price_ratio'] / avg_budget:.2e}  per EUR")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "method": "pairwise logistic regression (Bradley-Terry)",
        "training_pairs": len(X),
        "cv_accuracy": {"mean": round(float(cv_scores.mean()), 4),
                        "std":  round(float(cv_scores.std()),  4)},
        "ranking_accuracy": {
            "baseline":      {"correct": base_correct, "total": base_total,
                              "pct": round(100 * base_correct / base_total, 1)},
            "fitted_weights":{"correct": fit_correct,  "total": fit_total,
                              "pct": round(100 * fit_correct  / fit_total,  1)},
            "delta_correct":  delta,
        },
        "coefficients": {
            "scaled":  dict(zip(FEATURES, [round(float(c), 6) for c in coef])),
            "raw":     dict(zip(FEATURES, [round(float(c), 6) for c in coef_raw])),
        },
        "normalised_weights": norm_weights,
        "supplier_engine_mapping": {
            "_note": (
                "In supplier_engine._rank(), score = total + risk*w_risk - quality*w_quality "
                "- esg*w_esg - total*(pref*w_pref + inc*w_inc + ment*w_ment). "
                "Weights below are normalised (max=1). "
                "Scale them to your preferred absolute range (e.g. multiply by 500)."
            ),
            "w_risk":     norm_weights["risk_score"],
            "w_quality":  norm_weights["quality_score"],
            "w_esg":      norm_weights["esg_score"],
            "w_preferred":norm_weights["is_preferred"],
            "w_incumbent":norm_weights["is_incumbent"],
            "w_mentioned":norm_weights["is_mentioned"],
            "price_note": (
                f"price_ratio weight={norm_weights['price_ratio']:.4f}. "
                f"Equivalent per-EUR weight ≈ {norm_weights['price_ratio']/avg_budget:.2e} "
                f"(based on avg budget EUR {avg_budget:,.0f})."
            ),
        },
    }

    out_path = Path(__file__).parent / "scoring_weights.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    run()
