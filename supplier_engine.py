"""
SupplierEngine — ChainIQ START Hack 2026
=========================================
Processes a parsed procurement request and returns a ranked supplier shortlist
with policy evaluation, pricing, escalation triggers, and audit trail.

Usage:
    from supplier_engine import SupplierEngine
    engine = SupplierEngine()
    result = engine.process(request_dict)

    # Or process the full requests.json in one call:
    python supplier_engine.py   →  writes outputs.json
"""
from __future__ import annotations

import json
import re
from csv import DictReader
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data/data"

# ISO-2 country code → pricing region used in pricing.csv
# Distinct region values in pricing.csv: EU, CH, Americas, APAC, MEA
_COUNTRY_TO_REGION: dict[str, str] = {
    "CH": "CH",
    "US": "Americas", "CA": "Americas", "BR": "Americas", "MX": "Americas",
    "AU": "APAC", "SG": "APAC", "JP": "APAC", "IN": "APAC",
    "UAE": "MEA", "ZA": "MEA",
}


def country_to_region(country: str) -> str:
    return _COUNTRY_TO_REGION.get(country, "EU")


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(DictReader(f))


def _matches_category_scope(policy_row: dict[str, Any], cat_l1: str, cat_l2: str) -> bool:
    row_l1 = policy_row.get("category_l1")
    row_l2 = policy_row.get("category_l2")

    if row_l1 is not None and row_l1 != cat_l1:
        return False
    if row_l2 is not None and row_l2 != cat_l2:
        return False
    return True


def _matches_policy_applies_to(policy_row: dict[str, Any], cat_l1: str, cat_l2: str) -> bool:
    applies_to = policy_row.get("applies_to")
    if not applies_to:
        return True
    return cat_l2 in applies_to or cat_l1 in applies_to


_BELOW_THRESHOLD_RE = re.compile(
    r"below\s+([A-Z]{3})(?:/([A-Z]{3}))?\s*([0-9][0-9,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
_EXCEPTION_APPROVER_RE = re.compile(
    r"requires?\s+(.+?)\s+exception approval",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SupplierEngine:
    # Fallback weights used when scoring_weights.json is not present
    _DEFAULT_WEIGHTS: dict[str, float] = {
        "risk_score":    0.5732,
        "quality_score": 0.5967,
        "esg_score":     0.0269,
        "is_preferred":  0.3451,
        "is_incumbent":  0.0049,
        "is_mentioned":  1.0000,
    }
    # Multiplier so score deltas are human-readable integers
    _WEIGHT_SCALE = 100.0

    def __init__(self, data_dir: Path = DATA_DIR):
        self.suppliers = _load_csv(data_dir / "suppliers.csv")
        self.pricing = _load_csv(data_dir / "pricing.csv")
        self.policies = _load_json(Path(__file__).parent / "cleaned_policies.json")
        self.awards = _load_csv(data_dir / "historical_awards.csv")
        self._today = date.today()

        # Pre-built policy lookup sets
        # {(supplier_id, category_l1, category_l2)}
        self._preferred_set = self._build_preferred_set()
        # {(supplier_id, category_l1, category_l2): [policy rows]}
        self._restricted_map = self._build_restricted_map()

        # Load fitted scoring weights if available, else fall back to defaults
        weights_path = Path(__file__).parent / "scoring_weights.json"
        if weights_path.exists():
            self._weights = _load_json(weights_path)["normalised_weights"]
        else:
            self._weights = self._DEFAULT_WEIGHTS

        self._benchmark_score = self._build_benchmark(data_dir)

    # ------------------------------------------------------------------
    # Policy index builders
    # ------------------------------------------------------------------

    def _build_preferred_set(self) -> set[tuple[str, str, str]]:
        return {
            (ps["supplier_id"], ps["category_l1"], ps["category_l2"])
            for ps in self.policies["preferred_suppliers"]
        }

    def _build_restricted_map(self) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
        result: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for rs in self.policies["restricted_suppliers"]:
            key = (rs["supplier_id"], rs["category_l1"], rs["category_l2"])
            result.setdefault(key, []).append(rs)
        return result

    def _build_benchmark(self, data_dir: Path) -> float | None:
        requests_list = _load_json(data_dir / "requests.json")
        req_lookup = {r["request_id"]: r for r in requests_list}

        # Build supplier lookup: first row per supplier_id
        sup_lookup: dict[str, dict] = {}
        for row in self.suppliers:
            sid = row["supplier_id"]
            if sid not in sup_lookup:
                sup_lookup[sid] = row

        w = self._weights
        s = self._WEIGHT_SCALE
        scores: list[float] = []

        for award in self.awards:
            if str(award.get("award_rank")) != "1":
                continue
            req = req_lookup.get(award["request_id"])
            if req is None:
                continue
            sup = sup_lookup.get(award["supplier_id"])
            if sup is None:
                continue

            cat_l2 = award.get("category_l2", "")
            sup_id = award["supplier_id"]
            sup_name = award.get("supplier_name", "")
            incumbent = req.get("incumbent_supplier")
            preferred_mentioned = req.get("preferred_supplier_mentioned")
            esg_req = bool(req.get("esg_requirement"))

            cat_l1 = award.get("category_l1", req.get("category_l1", ""))
            is_preferred = (sup_id, cat_l1, cat_l2) in self._preferred_set
            is_incumbent = bool(incumbent and incumbent.lower() in sup_name.lower())
            is_mentioned = bool(preferred_mentioned and preferred_mentioned.lower() in sup_name.lower())

            quality = int(sup.get("quality_score", 50))
            risk = int(sup.get("risk_score", 50))
            esg = int(sup.get("esg_score", 50))
            esg_w = w["esg_score"] * (2.0 if esg_req else 1.0)

            score = (
                + risk * w["risk_score"] * s
                - quality * w["quality_score"] * s
                - esg * esg_w * s
                - float(is_preferred) * w["is_preferred"] * s
                - float(is_incumbent) * w["is_incumbent"] * s
                - float(is_mentioned) * w["is_mentioned"] * s
            )
            scores.append(score)

        if not scores:
            return None
        return min(scores)

    def _restriction_scope_matches(
        self,
        policy_rows: list[dict[str, Any]],
        primary_country: str,
        delivery_countries: list[str],
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for row in policy_rows:
            scopes = row.get("restriction_scope", [])
            if (
                "all" in scopes
                or primary_country in scopes
                or any(country in scopes for country in delivery_countries)
            ):
                matches.append(row)
        return matches

    def _restriction_threshold(self, reason: str) -> tuple[set[str], float] | None:
        match = _BELOW_THRESHOLD_RE.search(reason)
        if not match:
            return None
        currencies = {match.group(1).upper()}
        if match.group(2):
            currencies.add(match.group(2).upper())
        amount = float(match.group(3).replace(",", ""))
        return currencies, amount

    def _restriction_exception_approver(self, reason: str) -> str | None:
        match = _EXCEPTION_APPROVER_RE.search(reason)
        if match:
            return match.group(1).strip()
        if "exception approval" in reason.lower():
            return "Policy Exception Review"
        return None

    def _evaluate_restriction(
        self,
        policy_rows: list[dict[str, Any]],
        primary_country: str,
        delivery_countries: list[str],
        total_value: float | None = None,
        currency: str | None = None,
    ) -> dict[str, Any] | None:
        for row in self._restriction_scope_matches(policy_rows, primary_country, delivery_countries):
            reason = row.get("restriction_reason", "Policy restricted.")
            threshold = self._restriction_threshold(reason)
            approver = self._restriction_exception_approver(reason)
            if threshold is None:
                if approver:
                    return {
                        "action": "approval_required",
                        "reason": reason,
                        "approver": approver,
                    }
                return {
                    "action": "exclude",
                    "reason": reason,
                    "approver": None,
                }
            if total_value is None or currency is None:
                continue
            allowed_currencies, max_amount = threshold
            if currency.upper() not in allowed_currencies or total_value < max_amount:
                continue
            detailed_reason = (
                f"{reason} Evaluated total {currency.upper()} {total_value:,.2f} "
                f"exceeds the allowed threshold."
            )
            if approver:
                return {
                    "action": "approval_required",
                    "reason": detailed_reason,
                    "approver": approver,
                }
            return {
                "action": "exclude",
                "reason": detailed_reason,
                "approver": None,
            }
        return None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process(self, request: dict) -> dict:
        today = self._today

        # Unpack the most-used fields once
        req_id = request["request_id"]
        cat_l1 = request["category_l1"]
        cat_l2 = request["category_l2"]
        currency = request["currency"]
        budget = request.get("budget_amount")
        quantity = request.get("quantity")
        required_by = request.get("required_by_date")
        delivery_countries = request.get("delivery_countries") or [
            request["country"]]
        primary_country = delivery_countries[0] if delivery_countries else request["country"]
        data_residency = request.get("data_residency_constraint", False)
        esg_req = request.get("esg_requirement", False)
        preferred_mentioned = request.get("preferred_supplier_mentioned")
        incumbent = request.get("incumbent_supplier")
        region = country_to_region(primary_country)

        # ── 1. Validate ────────────────────────────────────────────────
        validation_issues, escalations = self._validate(request, today)

        # ── 2. Filter eligible suppliers ───────────────────────────────
        eligible, excluded = self._filter_suppliers(
            cat_l1, cat_l2, delivery_countries, primary_country, data_residency
        )

        # ── 3. Attach pricing ──────────────────────────────────────────
        priced: list[dict] = []
        for sup in eligible:
            pricing_row = self._get_pricing(
                sup["supplier_id"], cat_l1, cat_l2, region, currency, quantity
            )
            if pricing_row is None:
                excluded.append({
                    "supplier_id": sup["supplier_id"],
                    "supplier_name": sup["supplier_name"],
                    "reason": "No valid pricing row for category / region / currency.",
                })
                continue
            total_value = float(pricing_row["unit_price"]) * (quantity or 1)
            restriction_eval = self._evaluate_restriction(
                self._restricted_map.get((sup["supplier_id"], cat_l1, cat_l2), []),
                primary_country,
                delivery_countries,
                total_value=total_value,
                currency=currency,
            )
            if restriction_eval and restriction_eval["action"] == "exclude":
                excluded.append({
                    "supplier_id": sup["supplier_id"],
                    "supplier_name": sup["supplier_name"],
                    "reason": restriction_eval["reason"],
                })
                continue
            if restriction_eval and restriction_eval["action"] == "approval_required":
                priced.append({
                    **sup,
                    "pricing": pricing_row,
                    "policy_exception_required": True,
                    "policy_exception_reason": restriction_eval["reason"],
                    "policy_exception_approver": restriction_eval["approver"],
                })
                continue
            priced.append({**sup, "pricing": pricing_row})

        if not priced:
            if not any(e["rule"] == "ER-004" for e in escalations):
                escalations.append({
                    "escalation_id": f"ESC-{len(escalations)+1:03d}",
                    "rule": "ER-004",
                    "trigger": "No fully compliant supplier found. A best-effort shortlist has been generated for human review.",
                    "escalate_to": "Head of Category",
                    "blocking": True,
                })
            # Build best-bad shortlist from excluded suppliers so a human can still decide
            priced = self._best_bad_suppliers(
                cat_l1, cat_l2, delivery_countries, primary_country,
                region, currency, quantity
            )

        # ── 4. Policy evaluation ───────────────────────────────────────
        policy_eval = self._evaluate_policy(
            request, priced, quantity, budget, currency,
            primary_country, cat_l1, cat_l2, escalations
        )

        # ── 5. Budget feasibility check ────────────────────────────────
        # Historical data shows budgets are often indicative rather than hard caps.
        # Overage ≤ 20%  → validation warning only (recommendation can still proceed).
        # Overage  > 20% → blocking escalation requiring requester clarification.
        _BUDGET_TOLERANCE = 0.20
        priced_with_price = [s for s in priced if s.get(
            "pricing") and s["pricing"].get("unit_price")]
        if budget is not None and priced_with_price:
            min_total = min(
                float(s["pricing"]["unit_price"]) * (quantity or 1)
                for s in priced_with_price
            )
            if min_total > budget:
                overage_pct = (min_total - budget) / budget
                severity = "critical" if overage_pct > _BUDGET_TOLERANCE else "high"
                validation_issues.append({
                    "issue_id": f"V-{len(validation_issues)+1:03d}",
                    "severity": severity,
                    "type": "budget_insufficient",
                    "description": (
                        f"Budget {currency} {budget:,.2f} cannot cover {quantity or '?'} units "
                        f"at any compliant supplier. Lowest available total: {currency} {min_total:,.2f} "
                        f"({overage_pct*100:.1f}% over budget)."
                    ),
                    "action_required": (
                        f"Increase budget to at least {currency} {min_total:,.2f} or reduce quantity."
                    ),
                })
                if overage_pct > _BUDGET_TOLERANCE:
                    if not any(e["rule"] == "ER-001" and "budget" in e["trigger"].lower() for e in escalations):
                        escalations.append({
                            "escalation_id": f"ESC-{len(escalations)+1:03d}",
                            "rule": "ER-001",
                            "trigger": (
                                f"Budget {currency} {budget:,.2f} is insufficient by more than 20% — "
                                f"lowest compliant total is {currency} {min_total:,.2f} "
                                f"({overage_pct*100:.1f}% over budget)."
                            ),
                            "escalate_to": "Requester Clarification",
                            "blocking": True,
                        })

        # ── 6. Lead-time feasibility check ─────────────────────────────
        priced_with_lead = [
            s for s in priced
            if s.get("pricing") and s["pricing"].get("expedited_lead_time_days")
        ]
        if required_by and priced_with_lead:
            req_date = date.fromisoformat(required_by)
            days_left = (req_date - today).days
            all_infeasible = all(
                int(s["pricing"]["expedited_lead_time_days"]) > days_left
                for s in priced_with_lead
            )
            if all_infeasible and not any(e["rule"] == "ER-004" and "lead" in e["trigger"].lower() for e in escalations):
                exp_range = (
                    f"{min(int(s['pricing']['expedited_lead_time_days']) for s in priced_with_lead)}–"
                    f"{max(int(s['pricing']['expedited_lead_time_days']) for s in priced_with_lead)} days"
                )
                # Non-blocking: history shows awards proceed even with infeasible lead times.
                # The deadline is typically a desired date, not a hard veto.
                escalations.append({
                    "escalation_id": f"ESC-{len(escalations)+1:03d}",
                    "rule": "ER-004",
                    "trigger": (
                        f"Lead time advisory: required delivery {required_by} "
                        f"({days_left}d). All suppliers' expedited lead times are {exp_range}. "
                        f"Confirm with requester whether deadline is a hard constraint."
                    ),
                    "escalate_to": "Head of Category",
                    "blocking": False,
                })

        # ── 7. Rank ────────────────────────────────────────────────────
        shortlist = self._rank(
            priced, quantity, budget, incumbent,
            preferred_mentioned, esg_req, required_by, today
        )
        self._add_top_supplier_exception_escalation(shortlist, escalations)
        policy_trace = self._build_policy_trace(
            request=request,
            priced=priced,
            policy_eval=policy_eval,
            escalations=escalations,
            validation_issues=validation_issues,
        )

        # ── 8. Assemble output ─────────────────────────────────────────
        days_until = None
        if required_by:
            days_until = (date.fromisoformat(required_by) - today).days

        return {
            "request_id": req_id,
            "processed_at": datetime.now(tz=timezone.utc).isoformat(),
            "request_interpretation": {
                "category_l1": cat_l1,
                "category_l2": cat_l2,
                "quantity": quantity,
                "unit_of_measure": request.get("unit_of_measure"),
                "budget_amount": budget,
                "currency": currency,
                "delivery_country": primary_country,
                "required_by_date": required_by,
                "days_until_required": days_until,
                "data_residency_required": data_residency,
                "esg_requirement": esg_req,
                "preferred_supplier_stated": preferred_mentioned,
                "incumbent_supplier": incumbent,
                "requester_instruction": self._extract_instruction(request.get("request_text", "")),
            },
            "validation": {
                "completeness": (
                    "fail"
                    if any(v["severity"] == "critical" for v in validation_issues)
                    else "pass"
                ),
                "issues_detected": validation_issues,
            },
            "policy_evaluation": policy_eval,
            "policy_trace": policy_trace,
            "supplier_shortlist": shortlist,
            "suppliers_excluded": excluded,
            "escalations": escalations,
            "recommendation": self._build_recommendation(shortlist, escalations),
            "audit_trail": self._build_audit_trail(request, priced, excluded, policy_eval),
        }

    # ------------------------------------------------------------------
    # Step 1 – Validate
    # ------------------------------------------------------------------

    def _validate(self, req: dict, today: date) -> tuple[list[dict], list[dict]]:
        issues: list[dict] = []
        escalations: list[dict] = []

        def add_issue(severity, type_, desc, action):
            issues.append({
                "issue_id": f"V-{len(issues)+1:03d}",
                "severity": severity,
                "type": type_,
                "description": desc,
                "action_required": action,
            })

        if req.get("quantity") is None:
            add_issue(
                "high", "missing_quantity",
                "Quantity not specified.",
                "Requester must confirm quantity before pricing can be calculated.",
            )
            escalations.append({
                "escalation_id": "ESC-001",
                "rule": "ER-001",
                "trigger": "Quantity missing — cannot compute total contract value or select pricing tier.",
                "escalate_to": "Requester Clarification",
                "blocking": True,
            })

        if req.get("budget_amount") is None:
            add_issue(
                "high", "missing_budget",
                "Budget amount not specified.",
                "Requester must confirm budget before sourcing can proceed.",
            )

        required_by = req.get("required_by_date")
        if required_by:
            req_date = date.fromisoformat(required_by)
            days_left = (req_date - today).days
            if days_left < 0:
                add_issue(
                    "critical", "deadline_passed",
                    f"Required-by date {required_by} is in the past ({abs(days_left)} days ago).",
                    "Requester must update the delivery date.",
                )
            elif days_left <= 5:
                add_issue(
                    "high", "tight_lead_time",
                f"Only {days_left} day{'' if days_left==1 else 's'} until required delivery — most suppliers need >5 days.",
                    "Confirm whether deadline is a hard constraint; expedited options may not be available.",
                )

        return issues, escalations

    # ------------------------------------------------------------------
    # Step 2 – Filter suppliers
    # ------------------------------------------------------------------

    def _filter_suppliers(
        self,
        cat_l1: str,
        cat_l2: str,
        delivery_countries: list[str],
        primary_country: str,
        data_residency: bool,
    ) -> tuple[list[dict], list[dict]]:
        eligible: list[dict] = []
        excluded: list[dict] = []
        seen: set[str] = set()

        for row in self.suppliers:
            sup_id = row["supplier_id"]

            # Category match (one row per supplier × category_l2)
            if not _matches_category_scope(row, cat_l1, cat_l2):
                continue

            # Dedup — keep first matching row per supplier
            if sup_id in seen:
                continue

            # Active contract
            if row.get("contract_status", "").lower() != "active":
                excluded.append({
                    "supplier_id": sup_id,
                    "supplier_name": row["supplier_name"],
                    "reason": f"Contract status '{row.get('contract_status')}' is not active.",
                })
                seen.add(sup_id)
                continue

            # Delivery country coverage (service_regions is semicolon-delimited)
            service_regions = {r.strip() for r in row.get(
                "service_regions", "").split(";") if r.strip()}
            if not any(c in service_regions for c in delivery_countries):
                excluded.append({
                    "supplier_id": sup_id,
                    "supplier_name": row["supplier_name"],
                    "reason": f"Does not cover delivery country/ies {delivery_countries}. Covers: {sorted(service_regions)}.",
                })
                seen.add(sup_id)
                continue

            # Data residency
            if data_residency and row.get("data_residency_supported", "").lower() != "true":
                excluded.append({
                    "supplier_id": sup_id,
                    "supplier_name": row["supplier_name"],
                    "reason": "Data residency not supported (required by this request).",
                })
                seen.add(sup_id)
                continue

            # Policy restriction
            restriction_eval = self._evaluate_restriction(
                self._restricted_map.get((sup_id, cat_l1, cat_l2), []),
                primary_country,
                delivery_countries,
            )
            if restriction_eval and restriction_eval["action"] == "exclude":
                excluded.append({
                    "supplier_id": sup_id,
                    "supplier_name": row["supplier_name"],
                    "reason": restriction_eval["reason"],
                })
                seen.add(sup_id)
                continue

            eligible.append(row)
            seen.add(sup_id)

        return eligible, excluded

    # ------------------------------------------------------------------
    # Best-bad fallback: suppliers that failed hard filters but are the
    # closest matches available — returned for human review when no
    # fully compliant supplier exists.
    # ------------------------------------------------------------------

    def _best_bad_suppliers(
        self,
        cat_l1: str,
        cat_l2: str,
        delivery_countries: list[str],
        primary_country: str,
        region: str,
        currency: str,
        quantity: int | None,
    ) -> list[dict]:
        """
        Returns suppliers matching the category that failed one or more filters,
        with pricing attached where available.  Each entry is tagged with
        `policy_compliant=False` and a `violation_reasons` list so the user
        knows exactly what rule is being waived.
        """
        candidates: list[dict] = []
        seen: set[str] = set()

        for row in self.suppliers:
            sup_id = row["supplier_id"]
            if not _matches_category_scope(row, cat_l1, cat_l2):
                continue
            if sup_id in seen:
                continue
            seen.add(sup_id)

            violations: list[str] = []

            if row.get("contract_status", "").lower() != "active":
                violations.append(
                    f"contract_status={row.get('contract_status')}")

            service_regions = {r.strip() for r in row.get(
                "service_regions", "").split(";") if r.strip()}
            if not any(c in service_regions for c in delivery_countries):
                violations.append(f"does not cover {delivery_countries}")

            if not violations:
                continue  # passed all filters — not a best-bad candidate

            pricing_row = self._get_pricing(
                sup_id,
                cat_l1,
                cat_l2,
                region,
                currency,
                quantity,
                allow_region_fallback=True,
            )
            if pricing_row is None:
                violations.append("no pricing row for this region/currency")
            total_value = float(pricing_row["unit_price"]) * (quantity or 1) if pricing_row and pricing_row.get("unit_price") else None
            restriction_eval = self._evaluate_restriction(
                self._restricted_map.get((sup_id, cat_l1, cat_l2), []),
                primary_country,
                delivery_countries,
                total_value=total_value,
                currency=currency if total_value is not None else None,
            )
            if restriction_eval is not None:
                violations.append(restriction_eval["reason"])
                if restriction_eval["action"] == "approval_required":
                    row = {
                        **row,
                        "policy_exception_required": True,
                        "policy_exception_reason": restriction_eval["reason"],
                        "policy_exception_approver": restriction_eval["approver"],
                    }

            candidates.append({
                **row,
                "pricing": pricing_row or {},
                "policy_compliant": False,
                "violation_reasons": violations,
            })

        # Sort: fewest violations first, then by risk score ascending
        candidates.sort(key=lambda s: (
            len(s["violation_reasons"]), int(s.get("risk_score") or 99)))
        return candidates

    # ------------------------------------------------------------------
    # Step 3 – Pricing lookup
    # ------------------------------------------------------------------

    def _get_pricing(
        self,
        supplier_id: str,
        cat_l1: str,
        cat_l2: str,
        region: str,
        currency: str,
        quantity: int | None,
        allow_region_fallback: bool = False,
    ) -> dict | None:
        today_str = str(self._today)

        def _match(p: dict, rgn: str) -> bool:
            return (
                p["supplier_id"] == supplier_id
                and _matches_category_scope(p, cat_l1, cat_l2)
                and p["region"] == rgn
                and p["currency"] == currency
                and p.get("valid_from", "0000-00-00") <= today_str
                and today_str <= p.get("valid_to", "9999-99-99")
            )

        candidates = [p for p in self.pricing if _match(p, region)]

        # Fallback 1: CH requests — most EU suppliers only have EU/EUR pricing rows.
        # Try EU region with same currency first, then EU region with EUR (CHF→EUR cross).
        if not candidates and region == "CH":
            candidates = [p for p in self.pricing if _match(p, "EU")]
        if not candidates and region == "CH" and currency == "CHF":
            candidates = [
                p for p in self.pricing
                if (
                    p["supplier_id"] == supplier_id
                    and _matches_category_scope(p, cat_l1, cat_l2)
                    and p["region"] == "EU"
                    and p["currency"] == "EUR"
                    and p.get("valid_from", "0000-00-00") <= today_str
                    and today_str <= p.get("valid_to", "9999-99-99")
                )
            ]
            for row in candidates:
                row["_currency_note"] = "EUR pricing used as CHF proxy (no CHF-specific row available)"

        # Best-effort shortlist generation can use indicative pricing from a
        # different region when there is no exact regional match but the
        # currency and quantity tier still line up.
        if not candidates and allow_region_fallback:
            candidates = [
                p for p in self.pricing
                if (
                    p["supplier_id"] == supplier_id
                    and _matches_category_scope(p, cat_l1, cat_l2)
                    and p["currency"] == currency
                    and p.get("valid_from", "0000-00-00") <= today_str
                    and today_str <= p.get("valid_to", "9999-99-99")
                )
            ]
            for row in candidates:
                row["_pricing_note"] = (
                    f"Indicative {row.get('region', 'unknown')} pricing used; "
                    f"no {region} price row available."
                )

        if not candidates:
            return None

        candidates.sort(key=lambda p: int(p["min_quantity"]))

        if quantity is None:
            return candidates[0]  # lowest tier

        for row in candidates:
            if int(row["min_quantity"]) <= quantity <= int(row["max_quantity"]):
                return row

        # Quantity exceeds all tier maxima → return highest tier
        return candidates[-1]

    # ------------------------------------------------------------------
    # Step 4 – Policy evaluation
    # ------------------------------------------------------------------

    def _evaluate_policy(
        self,
        req: dict,
        priced: list[dict],
        quantity: int | None,
        budget: float | None,
        currency: str,
        primary_country: str,
        cat_l1: str,
        cat_l2: str,
        escalations: list[dict],
    ) -> dict:
        qty = quantity or 1
        totals = [
            float(s["pricing"]["unit_price"]) * qty
            for s in priced
            if s.get("pricing") and s["pricing"].get("unit_price")
        ]
        ref_value = min(totals) if totals else (budget or 0.0)

        at = self._find_approval_threshold(currency, ref_value)

        # Preferred-supplier policy check
        preferred_mentioned = req.get("preferred_supplier_mentioned")
        preferred_eval = None
        if preferred_mentioned:
            matched = next(
                (s for s in priced if preferred_mentioned.lower()
                 in s["supplier_name"].lower()),
                None,
            )
            if matched:
                is_pref = (matched["supplier_id"], cat_l1,
                           cat_l2) in self._preferred_set
                preferred_eval = {
                    "supplier": matched["supplier_name"],
                    "status": "eligible",
                    "is_preferred": is_pref,
                    "covers_delivery_country": True,
                    "is_restricted": False,
                    "policy_note": (
                        "Preferred supplier — include in comparison; does not mandate sole-source award."
                        if is_pref else
                        "Mentioned by requester but not on the preferred-supplier list for this category."
                    ),
                }
            else:
                preferred_eval = {
                    "supplier": preferred_mentioned,
                    "status": "ineligible_or_restricted",
                    "is_preferred": False,
                    "covers_delivery_country": None,
                    "is_restricted": True,
                    "policy_note": "Not found in eligible set — restriction or coverage issue.",
                }
                if not any(e["rule"] == "ER-002" for e in escalations):
                    escalations.append({
                        "escalation_id": f"ESC-{len(escalations)+1:03d}",
                        "rule": "ER-002",
                        "trigger": f"Preferred supplier '{preferred_mentioned}' is restricted or ineligible.",
                        "escalate_to": "Procurement Manager",
                        "blocking": False,
                    })

        # AT-driven quote-count escalation
        if at:
            quotes_required = at.get("min_supplier_quotes") or at.get(
                "quotes_required") or 1
            if len(priced) < quotes_required:
                escalations.append({
                    "escalation_id": f"ESC-{len(escalations)+1:03d}",
                    "rule": at["threshold_id"],
                    "trigger": (
                f"Policy {at['threshold_id']} requires {quotes_required} supplier quote{'' if quotes_required == 1 else 's'} "
                f"but only {len(priced)} eligible supplier{'' if len(priced) == 1 else 's'} found."
                    ),
                    "escalate_to": (
                        (at.get("deviation_approval_required_from")
                         or ["Procurement Manager"])[0]
                    ),
                    "blocking": False,
                })

        # Category rules
        cat_rules = [
            cr for cr in self.policies.get("category_rules", [])
            if _matches_category_scope(cr, cat_l1, cat_l2)
        ]
        for cr in cat_rules:
            self._apply_category_rule(
                cr, req, quantity, ref_value, escalations)

        # Geography rules
        geo_rules = [
            gr for gr in self.policies.get("geography_rules", [])
            if (
                gr.get("country") == primary_country
                or primary_country in gr.get("countries", [])
            )
            and _matches_policy_applies_to(gr, cat_l1, cat_l2)
        ]

        return {
            "approval_threshold": {
                "rule_applied": at["threshold_id"] if at else "N/A",
                "basis": f"Estimated contract value {currency} {ref_value:,.2f}",
                "quotes_required": (
                    (at.get("min_supplier_quotes") or at.get(
                        "quotes_required") or 1) if at else 1
                ),
                "approvers": at.get("managed_by") or at.get("approvers") or [] if at else [],
                "deviation_approval": (
                    (at.get("deviation_approval_required_from")
                     or [None])[0] if at else None
                ),
            },
            "preferred_supplier": preferred_eval,
            "eligible_supplier_count": len(priced),
            "category_rules_applied": [cr["rule_id"] for cr in cat_rules],
            "geography_rules_applied": [gr["rule_id"] for gr in geo_rules],
        }

    def _build_policy_trace(
        self,
        request: dict,
        priced: list[dict],
        policy_eval: dict,
        escalations: list[dict],
        validation_issues: list[dict],
    ) -> list[dict]:
        trace: list[dict] = []
        covered_rules: set[str] = set()
        escalation_by_rule = {e["rule"]: e for e in escalations}
        currency = request.get("currency", "EUR")

        approval = policy_eval.get("approval_threshold", {})
        threshold_rule = approval.get("rule_applied")
        if threshold_rule and threshold_rule != "N/A":
            covered_rules.add(threshold_rule)
            quotes_required = approval.get("quotes_required") or 1
            eligible_count = policy_eval.get("eligible_supplier_count", len(priced))
            threshold_escalation = escalation_by_rule.get(threshold_rule)
            status = "passed" if eligible_count >= quotes_required else "needs_approval"
            detail = (
                f"{approval.get('basis', 'Approval threshold applied')}. "
                f"Required quotes: {quotes_required}. Eligible suppliers found: {eligible_count}."
            )
            if threshold_escalation:
                detail = (
                    f"{detail} Approval required because {threshold_escalation['trigger']} "
                    f"Escalates to {threshold_escalation['escalate_to']}."
                )
            trace.append({
                "id": threshold_rule,
                "category": "approval_threshold",
                "status": status,
                "title": f"Approval threshold {threshold_rule}",
                "summary": (
                    f"Quote requirement met: {eligible_count}/{quotes_required}"
                    if status == "passed"
                    else f"Quote requirement not met: {eligible_count}/{quotes_required}"
                ),
                "detail": detail,
                "rule": threshold_rule,
                "approver": (
                    threshold_escalation["escalate_to"]
                    if threshold_escalation
                    else approval.get("deviation_approval")
                ),
                "blocking": bool(threshold_escalation and threshold_escalation.get("blocking")),
            })

        preferred = policy_eval.get("preferred_supplier")
        if preferred:
            preferred_rule = "preferred_supplier"
            trace.append({
                "id": preferred_rule,
                "category": "preferred_supplier",
                "status": (
                    "passed"
                    if preferred.get("status") == "eligible"
                    else "failed"
                ),
                "title": "Preferred supplier check",
                "summary": preferred.get("supplier") or "No preferred supplier named",
                "detail": preferred.get("policy_note") or "No policy note available.",
                "rule": preferred_rule,
                "approver": None,
                "blocking": False,
            })

        category_lookup = {
            rule["rule_id"]: rule
            for rule in self.policies.get("category_rules", [])
            if rule["rule_id"] in policy_eval.get("category_rules_applied", [])
        }
        for rule_id, rule in category_lookup.items():
            covered_rules.add(rule_id)
            escalation = escalation_by_rule.get(rule_id)
            trace.append({
                "id": rule_id,
                "category": "category_rule",
                "status": "needs_approval" if escalation else "passed",
                "title": f"Category rule {rule_id}",
                "summary": rule.get("rule_type", "category rule").replace("_", " "),
                "detail": (
                    f"{rule.get('rule_text', 'Rule applied.')} "
                    + (
                        f"Approval required from {escalation['escalate_to']}."
                        if escalation else
                        "Rule evaluated without escalation."
                    )
                ).strip(),
                "rule": rule_id,
                "approver": escalation["escalate_to"] if escalation else None,
                "blocking": bool(escalation and escalation.get("blocking")),
            })

        geo_lookup = {
            rule["rule_id"]: rule
            for rule in self.policies.get("geography_rules", [])
            if rule["rule_id"] in policy_eval.get("geography_rules_applied", [])
        }
        for rule_id, rule in geo_lookup.items():
            covered_rules.add(rule_id)
            escalation = escalation_by_rule.get(rule_id)
            scope = rule.get("country") or ", ".join(rule.get("countries", []))
            trace.append({
                "id": rule_id,
                "category": "geography_rule",
                "status": "needs_approval" if escalation else "passed",
                "title": f"Geography rule {rule_id}",
                "summary": scope or "Geography policy applied",
                "detail": (
                    f"{rule.get('rule_text', 'Geography rule applied.')} "
                    + (
                        f"Approval required from {escalation['escalate_to']}."
                        if escalation else
                        "Rule evaluated without escalation."
                    )
                ).strip(),
                "rule": rule_id,
                "approver": escalation["escalate_to"] if escalation else None,
                "blocking": bool(escalation and escalation.get("blocking")),
            })

        for issue in validation_issues:
            trace.append({
                "id": issue["issue_id"],
                "category": "validation",
                "status": "failed" if issue.get("severity") == "critical" else "warning",
                "title": f"Validation {issue['issue_id']}",
                "summary": issue.get("type", "validation issue").replace("_", " "),
                "detail": (
                    f"{issue.get('description', '')} "
                    f"Action required: {issue.get('action_required', 'Review required.')}."
                ).strip(),
                "rule": issue["issue_id"],
                "approver": None,
                "blocking": issue.get("severity") == "critical",
            })

        for escalation in escalations:
            rule_id = escalation["rule"]
            if rule_id in covered_rules:
                continue
            trace.append({
                "id": escalation["escalation_id"],
                "category": "escalation",
                "status": "failed" if escalation.get("blocking") else "needs_approval",
                "title": f"Escalation {rule_id}",
                "summary": escalation.get("trigger", "Escalation raised"),
                "detail": (
                    f"Escalated to {escalation.get('escalate_to', 'manual review')}. "
                    f"Blocking: {'yes' if escalation.get('blocking') else 'no'}."
                ),
                "rule": rule_id,
                "approver": escalation.get("escalate_to"),
                "blocking": bool(escalation.get("blocking")),
            })

        supplier_policy_ok = all(s.get("policy_compliant", True) for s in priced)
        trace.append({
            "id": "supplier_policy_compliance",
            "category": "supplier_screening",
            "status": "passed" if supplier_policy_ok else "failed",
            "title": "Supplier screening",
            "summary": (
                f"All {len(priced)} ranked suppliers passed policy screening"
                if supplier_policy_ok else
                "One or more ranked suppliers carry policy violations"
            ),
            "detail": (
            f"Evaluated {len(priced)} priced supplier{'' if len(priced) == 1 else 's'} in {currency}. "
                "Best-effort suppliers are marked with explicit violation reasons."
            ),
            "rule": "supplier_policy_compliance",
            "approver": None,
            "blocking": not supplier_policy_ok,
        })

        return trace

    def _find_approval_threshold(self, currency: str, value: float) -> dict | None:
        for at in self.policies["approval_thresholds"]:
            if at.get("currency") != currency:
                continue
            lo = float(at.get("min_amount") or at.get("min_value") or 0)
            hi_raw = at.get("max_amount") or at.get("max_value")
            hi = float(hi_raw) if hi_raw is not None else None
            if hi is None:
                if value >= lo:
                    return at
            elif lo <= value <= hi:
                return at
        return None

    def _apply_category_rule(
        self,
        cr: dict,
        req: dict,
        quantity: int | None,
        ref_value: float,
        escalations: list[dict],
    ) -> None:
        rule_type = cr.get("rule_type", "")
        rule_text = cr.get("rule_text", "")

        triggers: dict[str, tuple[bool, str]] = {
            "mandatory_comparison":   (ref_value > 100_000,             "Procurement Manager"),
            "engineering_spec_review": (bool(quantity and quantity > 50), "Engineering / CAD Lead"),
            "security_review":        (ref_value > 250_000,             "Security Architecture Team"),
            "cv_review":              (bool(quantity and quantity > 60), "Category Manager"),
            "brand_safety":           (True,                             "Marketing Governance Lead"),
            "residency_check":        (bool(req.get("data_residency_constraint")), "Security and Compliance Review"),
            "design_signoff":         (True,                             "Business Design Lead"),
            "certification_check":    (True,                             "Category Manager"),
            "performance_baseline":   (True,                             "Category Manager"),
            # informational only
            "fast_track":             (False,                            ""),
        }

        should_trigger, escalate_to = triggers.get(rule_type, (False, ""))
        if should_trigger:
            escalations.append({
                "escalation_id": f"ESC-{len(escalations)+1:03d}",
                "rule": cr["rule_id"],
                "trigger": rule_text,
                "escalate_to": escalate_to,
                "blocking": False,
            })

    # ------------------------------------------------------------------
    # Step 5 – Rank
    # ------------------------------------------------------------------

    def _rank(
        self,
        priced: list[dict],
        quantity: int | None,
        budget: float | None,
        incumbent: str | None,
        preferred_mentioned: str | None,
        esg_req: bool,
        required_by: str | None,
        today: date,
    ) -> list[dict]:
        req_date = date.fromisoformat(required_by) if required_by else None
        days_left = (req_date - today).days if req_date else None
        qty = quantity or 1

        scored: list[tuple[float, dict]] = []

        for sup in priced:
            p = sup.get("pricing") or {}
            sup_id = sup["supplier_id"]
            name = sup["supplier_name"]
            # Best-bad suppliers may have no pricing — use sentinel values
            unit_price = float(p["unit_price"]) if p.get(
                "unit_price") else None
            total = round(unit_price * qty,
                          2) if unit_price is not None else None
            std_lead = int(p["standard_lead_time_days"]) if p.get(
                "standard_lead_time_days") else None
            exp_lead = int(p["expedited_lead_time_days"]) if p.get(
                "expedited_lead_time_days") else None
            exp_unit = float(p["expedited_unit_price"]) if p.get(
                "expedited_unit_price") else None
            exp_total = round(
                exp_unit * qty, 2) if exp_unit is not None else None
            quality = int(sup.get("quality_score", 50))
            risk = int(sup.get("risk_score",    50))
            esg = int(sup.get("esg_score",     50))

            is_preferred = (sup_id, sup["category_l1"], sup["category_l2"]) in self._preferred_set
            is_incumbent = bool(
                incumbent and incumbent.lower() in name.lower())
            is_mentioned = bool(
                preferred_mentioned and preferred_mentioned.lower() in name.lower())
            over_budget = budget is not None and total is not None and total > budget
            policy_exception_required = bool(sup.get("policy_exception_required"))
            policy_exception_reason = sup.get("policy_exception_reason")
            policy_exception_approver = sup.get("policy_exception_approver")

            # Scoring weights fitted by pairwise logistic regression against
            # historical_awards.csv (see fit_scoring_weights.py).
            # Lower score = better rank.
            w = self._weights
            s = self._WEIGHT_SCALE
            esg_w = w["esg_score"] * (2.0 if esg_req else 1.0)

            violation_reasons = sup.get("violation_reasons", [])
            score = (
                + risk * w["risk_score"] * s   # higher risk  → worse
                - quality * w["quality_score"] * s   # higher qual  → better
                - esg * esg_w * s   # higher esg   → better
                - float(is_preferred) * w["is_preferred"] * s
                - float(is_incumbent) * w["is_incumbent"] * s
                - float(is_mentioned) * w["is_mentioned"] * s
                + float(policy_exception_required) * 100_000
                # best-bad always last
                + len(violation_reasons) * 1_000_000
            )

            if self._benchmark_score is not None and self._benchmark_score < 0 and not violation_reasons:
                raw_conf = (score / self._benchmark_score) * 100
                confidence_pct = round(min(max(raw_conf, 0.0), 100.0), 1)
            else:
                confidence_pct = 0.0 if violation_reasons else None

            # Build human-readable note
            notes: list[str] = []
            if violation_reasons:
                notes.append(
                    f"[BEST-EFFORT — policy violations: {'; '.join(violation_reasons)}]")
            if is_preferred:
                notes.append("Preferred supplier.")
            if is_incumbent:
                notes.append("Incumbent supplier.")
            if is_mentioned:
                notes.append("Requester's stated preference.")
            if policy_exception_required:
                notes.append(
                    f"Exception approval required from {policy_exception_approver}: {policy_exception_reason}"
                )
            if over_budget:
                notes.append(
                    f"Total {total:,.2f} exceeds budget {budget:,.2f}.")
            if days_left is not None and std_lead is not None and exp_lead is not None:
                if std_lead <= days_left:
                    notes.append(
                        f"Standard lead time {std_lead}d meets deadline.")
                elif exp_lead <= days_left:
                    notes.append(
                        f"Expedited option ({exp_lead}d) meets deadline; standard ({std_lead}d) does not.")
                else:
                    notes.append(
                        f"Both standard ({std_lead}d) and expedited ({exp_lead}d) lead times "
                        f"exceed the {days_left}d window."
                    )

            scored.append((score, {
                "supplier_id": sup_id,
                "supplier_name": name,
                "country_hq": sup.get("country_hq"),
                "preferred": is_preferred,
                "incumbent": is_incumbent,
                "confidence_pct": confidence_pct,
                "pricing_tier_applied": f"{p.get('min_quantity', '?')}–{p.get('max_quantity', '?')} units" if p else "N/A",
                "unit_price_eur": unit_price,
                "total_price_eur": total,
                "pricing_note": p.get("_pricing_note") or p.get("_currency_note"),
                "standard_lead_time_days": std_lead,
                "expedited_lead_time_days": exp_lead,
                "expedited_unit_price_eur": exp_unit,
                "expedited_total_eur": exp_total,
                "quality_score": quality,
                "risk_score": risk,
                "esg_score": esg,
                "policy_compliant": not bool(violation_reasons) and not policy_exception_required,
                "violation_reasons": violation_reasons or None,
                "covers_delivery_country": not any("cover" in v for v in violation_reasons),
                "policy_exception_required": policy_exception_required,
                "policy_exception_reason": policy_exception_reason,
                "policy_exception_approver": policy_exception_approver,
                "recommendation_note": " ".join(notes) if notes else "No issues.",
            }))

        scored.sort(key=lambda x: x[0])
        return [dict(rank=i + 1, **entry) for i, (_, entry) in enumerate(scored)]

    # ------------------------------------------------------------------
    # Step 6 – Recommendation
    # ------------------------------------------------------------------

    def _build_recommendation(
        self, shortlist: list[dict], escalations: list[dict]
    ) -> dict:
        # Impossible blockers — request cannot proceed without external changes
        # (missing info, budget gap, no suppliers, infeasible deadline).
        hard_blockers = [e for e in escalations if e.get("blocking")]

        # Approval gates — request is executable but needs sign-off before award.
        # These are non-blocking escalations with a named approver.
        approval_gates = [
            e for e in escalations
            if not e.get("blocking") and e.get("escalate_to")
        ]

        if hard_blockers:
            top = shortlist[0] if shortlist else None
            min_total = min(
                (s["total_price_eur"]
                 for s in shortlist if s["total_price_eur"] is not None),
                default=None,
            )

            # ER-001: missing/insufficient requester input — award is possible once provided.
            # ER-004: no compliant suppliers or infeasible lead time — truly impossible.
            # If any ER-004 is present, the request cannot proceed regardless of clarification.
            truly_impossible = [
                e for e in hard_blockers if e.get("rule") == "ER-004"]
            clarification_needed = [
                e for e in hard_blockers if e.get("rule") == "ER-001"]

            if truly_impossible:
                return {
                    "status": "cannot_proceed",
                    "reason": (
                f"{len(truly_impossible)} infeasibility issue{'' if len(truly_impossible) == 1 else 's'} cannot be resolved by requester input: "
                        + "; ".join(e["trigger"][:80]
                                    for e in truly_impossible)
                    ),
                    "preferred_supplier_if_resolved": top["supplier_name"] if top else None,
                    "preferred_supplier_rationale": top["recommendation_note"] if top else None,
                    "minimum_budget_required": min_total,
                }

            # Only ER-001 blockers — award is unblocked once requester provides the missing input.
            clarifications_needed = [
                {"field": e["trigger"], "rule": e["rule"],
                    "escalate_to": e["escalate_to"]}
                for e in clarification_needed
            ]
            return {
                "status": "needs_clarification",
                "reason": (
                    f"{len(clarification_needed)} piece"
                    f"{'' if len(clarification_needed) == 1 else 's'} of requester input required before sourcing can proceed."
                ),
                "clarifications_needed": clarifications_needed,
                "preferred_supplier_if_clarified": top["supplier_name"] if top else None,
                "preferred_supplier_rationale": top["recommendation_note"] if top else None,
                "minimum_budget_required": min_total,
            }

        if not shortlist:
            return {"status": "no_eligible_suppliers", "reason": "No compliant suppliers found."}

        top = shortlist[0]

        if approval_gates:
            # Deduplicate approvers and preserve order
            seen: set[str] = set()
            approvals_required: list[dict] = []
            for e in approval_gates:
                approver = e["escalate_to"]
                if approver not in seen:
                    seen.add(approver)
                    approvals_required.append({
                        "approver": approver,
                        "reason": e["trigger"],
                        "rule": e["rule"],
                    })
            return {
                "status": "pending_approval",
                "recommended_supplier": top["supplier_name"],
                "recommended_supplier_id": top["supplier_id"],
                "total_price": top["total_price_eur"],
                "rationale": top["recommendation_note"],
                "approvals_required": approvals_required,
            }

        return {
            "status": "ready_to_award",
            "recommended_supplier": top["supplier_name"],
            "recommended_supplier_id": top["supplier_id"],
            "total_price": top["total_price_eur"],
            "rationale": top["recommendation_note"],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_instruction(self, text: str) -> str | None:
        lowered = text.lower()
        for phrase in ("no exception", "single supplier only", "must use", "do not", "please use"):
            idx = lowered.find(phrase)
            if idx != -1:
                return text[max(0, idx):idx + 60].strip()
        return None

    def _add_top_supplier_exception_escalation(
        self,
        shortlist: list[dict[str, Any]],
        escalations: list[dict[str, Any]],
    ) -> None:
        if not shortlist:
            return
        top = shortlist[0]
        if not top.get("policy_exception_required"):
            return
        approver = top.get("policy_exception_approver") or "Policy Exception Review"
        trigger = top.get("policy_exception_reason") or f"Supplier {top['supplier_name']} requires exception approval."
        if any(
            e.get("rule") == "RS-001"
            and e.get("escalate_to") == approver
            and top["supplier_name"] in e.get("trigger", "")
            for e in escalations
        ):
            return
        escalations.append({
            "escalation_id": f"ESC-{len(escalations)+1:03d}",
            "rule": "RS-001",
            "trigger": f"Recommended supplier {top['supplier_name']} requires exception approval. {trigger}",
            "escalate_to": approver,
            "blocking": False,
        })

    def _build_audit_trail(
        self,
        req: dict,
        priced: list[dict],
        excluded: list[dict],
        policy_eval: dict,
    ) -> dict:
        cat_l2 = req["category_l2"]
        country = req.get("country", "")
        hist = [
            a for a in self.awards
            if a.get("category_l2") == cat_l2 and a.get("country") == country
        ]
        policies_checked = list({
            policy_eval["approval_threshold"]["rule_applied"],
            *policy_eval.get("category_rules_applied", []),
            *policy_eval.get("geography_rules_applied", []),
        } - {"N/A"})

        return {
            "policies_checked": policies_checked,
            "supplier_ids_evaluated": (
                [s["supplier_id"] for s in priced]
                + [e["supplier_id"] for e in excluded if "supplier_id" in e]
            ),
            "pricing_tiers_applied": (
                f"{country_to_region(country)} region, {req['currency']} currency"
            ),
            "data_sources_used": [
                "requests.json", "suppliers.csv", "pricing.csv", "cleaned_policies.json"
            ],
            "historical_awards_consulted": len(hist) > 0,
            "historical_award_note": (
            f"{len(hist)} prior award{'' if len(hist) == 1 else 's'} found for {cat_l2} in {country}: "
                + ", ".join(
                    f"{a['award_id']} → {a['supplier_name']} ({a['currency']} {a['total_value']})"
                    for a in hist[:5]
                )
            ) if hist else f"No prior awards found for {cat_l2} in {country}.",
        }


# ---------------------------------------------------------------------------
# Convenience: batch-process the full requests.json
# ---------------------------------------------------------------------------

def process_all(data_dir: Path = DATA_DIR) -> list[dict]:
    engine = SupplierEngine(data_dir)
    requests = _load_json(data_dir / "requests.json")
    return [engine.process(req) for req in requests]


if __name__ == "__main__":
    results = process_all()
    out_path = Path(__file__).parent / "outputs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Processed {len(results)} request(s) → {out_path}")
