from __future__ import annotations

import json
import os
import re
import uuid
from csv import DictReader
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from supplier_engine import SupplierEngine

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data" / "data"
REQUEST_JSON_PATH = ROOT_DIR / "request.json"
CRITERIA_PATH = ROOT_DIR / "criteria.json"

COUNTRY_NAMES = {
    "AE": "United Arab Emirates",
    "AT": "Austria",
    "AU": "Australia",
    "BE": "Belgium",
    "BR": "Brazil",
    "CA": "Canada",
    "CH": "Switzerland",
    "DE": "Germany",
    "ES": "Spain",
    "FR": "France",
    "GB": "United Kingdom",
    "IE": "Ireland",
    "IN": "India",
    "IT": "Italy",
    "JP": "Japan",
    "KR": "South Korea",
    "MX": "Mexico",
    "NL": "Netherlands",
    "PL": "Poland",
    "PT": "Portugal",
    "SG": "Singapore",
    "US": "United States",
    "ZA": "South Africa",
}

CITY_TO_COUNTRY = {
    "amsterdam": "NL",
    "antwerp": "BE",
    "barcelona": "ES",
    "berlin": "DE",
    "bern": "CH",
    "brussels": "BE",
    "dublin": "IE",
    "frankfurt": "DE",
    "geneva": "CH",
    "london": "GB",
    "madrid": "ES",
    "milan": "IT",
    "munich": "DE",
    "paris": "FR",
    "rome": "IT",
    "singapore": "SG",
    "tokyo": "JP",
    "vienna": "AT",
    "warsaw": "PL",
    "zurich": "CH",
}

COUNTRY_COORDINATES = {
    "AE": {"lat": 24.4539, "lng": 54.3773},
    "AT": {"lat": 48.2082, "lng": 16.3738},
    "AU": {"lat": -35.2809, "lng": 149.13},
    "BE": {"lat": 50.8503, "lng": 4.3517},
    "BR": {"lat": -15.7975, "lng": -47.8919},
    "CA": {"lat": 45.4215, "lng": -75.6972},
    "CH": {"lat": 47.3769, "lng": 8.5417},
    "DE": {"lat": 52.52, "lng": 13.405},
    "ES": {"lat": 40.4168, "lng": -3.7038},
    "FR": {"lat": 48.8566, "lng": 2.3522},
    "GB": {"lat": 51.5074, "lng": -0.1278},
    "IE": {"lat": 53.3498, "lng": -6.2603},
    "IN": {"lat": 28.6139, "lng": 77.209},
    "IT": {"lat": 41.9028, "lng": 12.4964},
    "JP": {"lat": 35.6762, "lng": 139.6503},
    "KR": {"lat": 37.5665, "lng": 126.978},
    "MX": {"lat": 19.4326, "lng": -99.1332},
    "NL": {"lat": 52.3676, "lng": 4.9041},
    "PL": {"lat": 52.2297, "lng": 21.0122},
    "PT": {"lat": 38.7223, "lng": -9.1393},
    "SG": {"lat": 1.3521, "lng": 103.8198},
    "US": {"lat": 38.9072, "lng": -77.0369},
    "ZA": {"lat": -25.7479, "lng": 28.2293},
}

COUNTRY_ALIASES = {
    **{code.lower(): code for code in COUNTRY_NAMES},
    **{name.lower(): code for code, name in COUNTRY_NAMES.items()},
    "uk": "GB",
    "uae": "UAE",
    "swiss": "CH",
}

CURRENCY_BY_COUNTRY = {
    "CH": "CHF",
    "GB": "GBP",
    "US": "USD",
}


def _load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        return list(DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class ParseResult:
    request_json: dict[str, Any]
    source: str


class RequestWorkflowService:
    def __init__(self, engine: SupplierEngine):
        self.engine = engine
        self.categories = _load_csv(DATA_DIR / "categories.csv")
        self.supplier_rows = _load_csv(DATA_DIR / "suppliers.csv")
        self.criteria = _load_json(CRITERIA_PATH)
        self.critical_criteria = self.criteria["fields"]["critical"]
        self.pending_sessions: dict[str, dict[str, Any]] = {}
        self._supplier_names = sorted(
            {row["supplier_name"] for row in self.supplier_rows},
            key=len,
            reverse=True,
        )
        self._supplier_by_id = {row["supplier_id"]: row for row in self.supplier_rows}

    def run(self, message: str, session_id: str | None = None) -> dict[str, Any]:
        session_id = session_id or f"session-{uuid.uuid4().hex[:12]}"
        session_state = self.pending_sessions.get(session_id)
        parse_result = self._parse_request(message, session_state)
        missing_fields = self._find_missing_critical_fields(parse_result.request_json)

        REQUEST_JSON_PATH.write_text(
            json.dumps(parse_result.request_json, indent=2),
            encoding="utf-8",
        )

        if missing_fields:
            question = self._build_follow_up_question(missing_fields)
            self.pending_sessions[session_id] = {
                "request_json": parse_result.request_json,
                "messages": [
                    *(session_state.get("messages", []) if session_state else []),
                    {"role": "user", "content": message},
                ],
                "missing_fields": missing_fields,
            }
            return {
                "status": "needs_clarification",
                "session_id": session_id,
                "request_json_path": str(REQUEST_JSON_PATH),
                "request": parse_result.request_json,
                "parser_source": parse_result.source,
                "missing_critical_fields": missing_fields,
                "follow_up_question": question,
                "engine_output": None,
                "ui": {
                    "summary": question,
                    "suppliers": [],
                    "notifications": [],
                },
            }

        engine_output = self.engine.process(parse_result.request_json)
        ui_suppliers = self._build_ui_suppliers(engine_output)
        self.pending_sessions.pop(session_id, None)
        return {
            "status": "completed",
            "session_id": session_id,
            "request_json_path": str(REQUEST_JSON_PATH),
            "request": parse_result.request_json,
            "parser_source": parse_result.source,
            "missing_critical_fields": [],
            "follow_up_question": None,
            "engine_output": engine_output,
            "ui": {
                "summary": self._build_summary(engine_output),
                "suppliers": ui_suppliers,
                "notifications": self._build_notifications(engine_output),
            },
        }

    def _parse_request(self, message: str, session_state: dict[str, Any] | None) -> ParseResult:
        if session_state:
            updated = self._update_with_moonshot(session_state["request_json"], message)
            if updated is not None:
                hinted = self._parse_follow_up_hint(message, session_state.get("missing_fields", []))
                merged_update = self._merge_request_data(
                    self._merge_request_data(session_state["request_json"], updated),
                    hinted,
                )
                return ParseResult(
                    request_json=self._normalise_request(merged_update, session_state["request_json"].get("request_text", ""), message),
                    source="moonshot-update",
                )

            hinted = self._parse_follow_up_hint(message, session_state.get("missing_fields", []))
            merged = self._merge_request_data(
                self._merge_request_data(session_state["request_json"], self._heuristic_parse(message)),
                hinted,
            )
            return ParseResult(
                request_json=self._normalise_request(merged, session_state["request_json"].get("request_text", ""), message),
                source="heuristic-update",
            )

        moonshot_result = self._parse_with_moonshot(message)
        if moonshot_result is not None:
            return ParseResult(
                request_json=self._normalise_request(moonshot_result, message),
                source="moonshot",
            )

        return ParseResult(
            request_json=self._normalise_request(self._heuristic_parse(message), message),
            source="heuristic",
        )

    def _parse_with_moonshot(self, message: str) -> dict[str, Any] | None:
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            return None

        base_url = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1").rstrip("/")
        model = os.getenv("MOONSHOT_MODEL", "kimik2.5")
        categories = [
            {
                "category_l1": row["category_l1"],
                "category_l2": row["category_l2"],
                "typical_unit": row["typical_unit"],
            }
            for row in self.categories
        ]
        system_prompt = (
            "Convert the user's procurement chat message into a JSON object. "
            "Return JSON only. Use one of the allowed categories provided. "
            "Preserve unknown fields as null instead of inventing values. "
            "Expected keys: category_l1, category_l2, title, quantity, unit_of_measure, "
            "budget_amount, currency, required_by_date, country, site, delivery_countries, "
            "preferred_supplier_mentioned, incumbent_supplier, contract_type_requested, "
            "data_residency_constraint, esg_requirement, business_unit, requester_role, "
            "request_language, scenario_tags. "
            f"Allowed categories: {json.dumps(categories)}"
        )
        return self._call_moonshot(base_url, api_key, model, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ])

    def _update_with_moonshot(self, current_request: dict[str, Any], message: str) -> dict[str, Any] | None:
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            return None

        base_url = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1").rstrip("/")
        model = os.getenv("MOONSHOT_MODEL", "kimik2.5")
        critical_fields = list(self.critical_criteria.keys())
        system_prompt = (
            "You update an existing procurement request JSON with a user's clarification. "
            "Return JSON only. Keep all existing fields unless the user clarification changes them. "
            "Only fill values supported by the clarification or already present in the request. "
            f"Critical fields that must be preserved or completed when possible: {critical_fields}."
        )
        user_prompt = json.dumps({
            "current_request": current_request,
            "user_clarification": message,
        })
        return self._call_moonshot(base_url, api_key, model, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

    def _call_moonshot(self, base_url: str, api_key: str, model: str, messages: list[dict[str, str]]) -> dict[str, Any] | None:
        payload = {
            "model": model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        req = request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=30) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
            return None

        content = raw.get("choices", [{}])[0].get("message", {}).get("content")
        if not isinstance(content, str):
            return None
        try:
            return json.loads(self._strip_json_wrapping(content))
        except json.JSONDecodeError:
            return None

    def _strip_json_wrapping(self, content: str) -> str:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        return cleaned

    def _heuristic_parse(self, message: str) -> dict[str, Any]:
        category = self._detect_category(message)
        country = self._detect_country(message)
        quantity = self._extract_quantity(message)
        budget_amount, currency = self._extract_budget(message)
        required_by_date = self._extract_required_date(message)
        preferred = self._find_supplier_name(message)
        return {
            "category_l1": category["category_l1"] if category else None,
            "category_l2": category["category_l2"] if category else None,
            "title": f"{category['category_l2']} procurement request" if category else "Procurement request",
            "quantity": quantity,
            "unit_of_measure": category["typical_unit"] if category else None,
            "budget_amount": budget_amount,
            "currency": currency,
            "required_by_date": required_by_date,
            "country": country,
            "site": self._extract_site(message),
            "delivery_countries": [country] if country else [],
            "preferred_supplier_mentioned": preferred,
            "incumbent_supplier": None,
            "contract_type_requested": "purchase",
            "data_residency_constraint": "data residency" in message.lower(),
            "esg_requirement": "esg" in message.lower() or "sustainable" in message.lower(),
            "business_unit": "Frontend Intake",
            "requester_role": "Requester",
            "request_language": "en",
            "scenario_tags": [],
        }

    def _normalise_request(
        self,
        parsed: dict[str, Any],
        original_message: str,
        follow_up_message: str | None = None,
    ) -> dict[str, Any]:
        combined_message = original_message if not follow_up_message else f"{original_message}\nFollow-up: {follow_up_message}"
        category = self._coerce_category(parsed.get("category_l1"), parsed.get("category_l2"), combined_message)
        country = self._coerce_country(parsed.get("country"), parsed.get("site"), parsed.get("delivery_countries"), combined_message)
        currency = self._normalise_currency_value(parsed.get("currency")) or CURRENCY_BY_COUNTRY.get(country, "EUR")
        request_id = parsed.get("request_id") or f"REQ-{uuid.uuid4().hex[:8].upper()}"
        title = parsed.get("title") or (f"{category['category_l2']} request" if category else "Procurement request")

        quantity = self._coerce_int(parsed.get("quantity"))
        if quantity is None:
            quantity = self._extract_quantity(follow_up_message or combined_message)
        budget_amount = self._coerce_float(parsed.get("budget_amount"))
        if budget_amount is None:
            budget_source = parsed.get("budget_amount") if isinstance(parsed.get("budget_amount"), str) else None
            budget_amount, detected_currency = self._extract_budget(budget_source or follow_up_message or combined_message)
            currency = currency or detected_currency or "EUR"
        preferred = self._find_supplier_name(parsed.get("preferred_supplier_mentioned") or combined_message)
        incumbent = self._find_supplier_name(parsed.get("incumbent_supplier") or "")
        delivery_countries = parsed.get("delivery_countries") or ([country] if country else [])

        return {
            "request_id": request_id,
            "created_at": parsed.get("created_at") or datetime.now(tz=timezone.utc).isoformat(),
            "request_channel": parsed.get("request_channel") or "frontend_chat",
            "request_language": parsed.get("request_language") or "en",
            "business_unit": parsed.get("business_unit") or "Frontend Intake",
            "country": country,
            "site": parsed.get("site") or self._extract_site(combined_message) or country,
            "requester_id": parsed.get("requester_id") or "frontend-user",
            "requester_role": parsed.get("requester_role") or "Requester",
            "submitted_for_id": parsed.get("submitted_for_id") or "frontend-user",
            "category_l1": category["category_l1"] if category else None,
            "category_l2": category["category_l2"] if category else None,
            "title": title,
            "request_text": combined_message.strip(),
            "currency": currency or "EUR",
            "budget_amount": budget_amount,
            "quantity": quantity,
            "unit_of_measure": parsed.get("unit_of_measure") or (category["typical_unit"] if category else None),
            "required_by_date": parsed.get("required_by_date") or self._extract_required_date(combined_message),
            "preferred_supplier_mentioned": preferred,
            "incumbent_supplier": incumbent,
            "contract_type_requested": parsed.get("contract_type_requested") or "purchase",
            "delivery_countries": delivery_countries,
            "data_residency_constraint": bool(parsed.get("data_residency_constraint")),
            "esg_requirement": bool(parsed.get("esg_requirement")),
            "status": parsed.get("status") or "pending_review",
            "scenario_tags": parsed.get("scenario_tags") or [],
        }

    def _find_missing_critical_fields(self, request_json: dict[str, Any]) -> list[dict[str, Any]]:
        missing: list[dict[str, Any]] = []
        for field_name, criteria in self.critical_criteria.items():
            if field_name.startswith("_"):
                continue
            value = request_json.get(field_name)
            if value in (None, "", []):
                missing.append({
                    "field": field_name,
                    "reason": "missing",
                    "criteria": criteria,
                    "attempted_value": self._extract_requested_product_phrase(request_json.get("request_text", "")) if field_name == "category_l2" else None,
                })
                continue
            if field_name == "category_l2" and value not in criteria.get("values", []):
                missing.append({"field": field_name, "reason": "invalid", "criteria": criteria, "attempted_value": value})
            elif field_name == "country" and value not in criteria.get("values", []):
                missing.append({"field": field_name, "reason": "invalid", "criteria": criteria})
            elif field_name == "currency" and value not in criteria.get("values", []):
                missing.append({"field": field_name, "reason": "invalid", "criteria": criteria})
            elif field_name == "quantity" and not self._is_positive_number(value):
                missing.append({"field": field_name, "reason": "invalid", "criteria": criteria})
            elif field_name == "budget_amount" and not self._is_positive_number(value):
                missing.append({"field": field_name, "reason": "invalid", "criteria": criteria})
        return missing

    def _build_follow_up_question(self, missing_fields: list[dict[str, Any]]) -> str:
        prompts: list[str] = []
        for item in missing_fields:
            field = item["field"]
            criteria = item["criteria"]
            if field == "category_l2":
                attempted_value = item.get("attempted_value")
                if attempted_value:
                    return f"{attempted_value} isn't a supplied product, try asking for a valid one."
                examples = ", ".join(criteria["values"][:5])
                prompts.append(f"what are you buying exactly? Use a category such as {examples}")
            elif field == "country":
                prompts.append("which delivery country should I use? Provide the ISO-2 country code such as DE, CH, or US")
            elif field == "quantity":
                prompts.append("what quantity do you need?")
            elif field == "budget_amount":
                prompts.append("what is the budget amount?")
            elif field == "currency":
                prompts.append("which currency should I use? Choose EUR, CHF, or USD")
        if not prompts:
            field_names = ", ".join(item["field"] for item in missing_fields)
            return f"I still need critical request details before I can run supplier matching: please provide valid values for {field_names}."
        joined = " Also tell me ".join(prompts)
        return f"I still need critical request details before I can run supplier matching: {joined}."

    def _coerce_category(self, category_l1: str | None, category_l2: str | None, message: str) -> dict[str, str] | None:
        if category_l1 and category_l2:
            for row in self.categories:
                if row["category_l1"].lower() == str(category_l1).lower() and row["category_l2"].lower() == str(category_l2).lower():
                    return row
        if category_l2:
            for row in self.categories:
                if row["category_l2"].lower() == str(category_l2).lower():
                    return row
        return self._detect_category(message)

    def _detect_category(self, message: str) -> dict[str, str] | None:
        lowered = message.lower()
        keyword_map = {
            "dock": "Docking Stations",
            "laptop": "Laptops",
            "notebook": "Laptops",
            "monitor": "Monitors",
            "screen": "Monitors",
            "phone": "Smartphones",
            "smartphone": "Smartphones",
            "tablet": "Tablets",
            "chair": "Office Chairs",
            "desk": "Workstations and Desks",
            "cloud compute": "Cloud Compute",
            "storage": "Cloud Storage",
            "security": "Cloud Security Services",
            "software development": "Software Development Services",
            "cybersecurity": "Cybersecurity Advisory",
        }
        for keyword, category_l2 in keyword_map.items():
            if keyword in lowered:
                for row in self.categories:
                    if row["category_l2"] == category_l2:
                        return row
        return None

    def _coerce_country(self, country: str | None, site: str | None, delivery_countries: list[str] | None, message: str) -> str | None:
        for candidate in [country, *(delivery_countries or []), site, self._detect_country(message)]:
            if not candidate:
                continue
            resolved = COUNTRY_ALIASES.get(str(candidate).strip().lower())
            if resolved:
                return self._request_country_code(resolved)
        return None

    def _detect_country(self, message: str) -> str | None:
        lowered = message.lower()
        for alias, code in COUNTRY_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                return code
        for city, code in CITY_TO_COUNTRY.items():
            if re.search(rf"\b{re.escape(city)}\b", lowered):
                return code
        return None

    def _extract_quantity(self, message: str | None) -> int | None:
        if not message:
            return None
        match = re.search(r"\b(\d{1,6})\s+(?:x\s+)?(?:units?|devices?|laptops?|docks?|stations?|chairs?|desks?)\b", message, re.IGNORECASE)
        if match:
            return int(match.group(1))
        explicit = re.search(r"\b(?:qty|quantity)\s*[:=]?\s*(\d{1,6})\b", message, re.IGNORECASE)
        if explicit:
            return int(explicit.group(1))
        fallback = re.search(r"\bneed\s+(\d{1,6})\b", message, re.IGNORECASE)
        if fallback:
            return int(fallback.group(1))
        return None

    def _extract_budget(self, message: str | None) -> tuple[float | None, str | None]:
        if not message:
            return None, None
        compact = re.search(r"\b(\d+(?:\.\d+)?)\s*([kKmM])\s*(EUR|CHF|USD|GBP|euro|euros|dollars?)?\b", message, re.IGNORECASE)
        if compact:
            base_amount = float(compact.group(1))
            multiplier = 1000 if compact.group(2).lower() == "k" else 1000000
            currency_token = compact.group(3)
            currency = self._normalise_currency(currency_token)
            return base_amount * multiplier, currency
        patterns = [
            (r"(?:budget|capped at|cap of|under|max(?:imum)? of)\s*(?:is\s*)?(?:(EUR|CHF|USD|GBP)\s*)?(\d[\d,\s]*(?:\.\d{1,2})?)\s*(EUR|CHF|USD|GBP)?", 2),
            (r"(?:(EUR|CHF|USD|GBP)\s*)(\d[\d,\s]*(?:\.\d{1,2})?)", 2),
            (r"(\d[\d,\s]*(?:\.\d{1,2})?)\s*(EUR|CHF|USD|GBP|euro|euros|dollars?)\b", 1),
        ]
        for pattern, amount_group in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if not match:
                continue
            amount = float(match.group(amount_group).replace(",", "").replace(" ", ""))
            currency_groups = [self._normalise_currency(group) for group in match.groups() if self._normalise_currency(group)]
            currency = currency_groups[0] if currency_groups else None
            if amount >= 100:
                return amount, currency
        return None, None

    def _extract_required_date(self, message: str | None) -> str | None:
        if not message:
            return None
        iso_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", message)
        if iso_match:
            return iso_match.group(1)
        return None

    def _extract_site(self, message: str | None) -> str | None:
        if not message:
            return None
        lowered = message.lower()
        for city in CITY_TO_COUNTRY:
            if re.search(rf"\b{re.escape(city)}\b", lowered):
                return city.title()
        return None

    def _extract_requested_product_phrase(self, message: str | None) -> str | None:
        if not message:
            return None
        cleaned = re.sub(r"follow-up\s*:\s*", " ", message.lower())
        cleaned = re.sub(r"\b\d+(?:\.\d+)?[kKmM]?\b", " ", cleaned)
        cleaned = re.sub(r"\b(to|for|in|at|by|under|budget|with|need|needs|qty|quantity|eur|usd|chf|gbp|euro|euros)\b", " ", cleaned)
        cleaned = re.sub(r"[^a-zA-Z\s/-]", " ", cleaned)
        tokens = [token for token in cleaned.split() if len(token) > 2 and token not in CITY_TO_COUNTRY]
        if not tokens:
            return None
        phrase = " ".join(tokens[:3]).strip()
        return phrase.capitalize() if phrase else None

    def _find_supplier_name(self, text: str | None) -> str | None:
        if not text:
            return None
        lowered = text.lower()
        for supplier_name in self._supplier_names:
            if supplier_name.lower() in lowered:
                return supplier_name
        return None

    def _build_ui_suppliers(self, engine_output: dict[str, Any]) -> list[dict[str, Any]]:
        shortlist = engine_output.get("supplier_shortlist", [])
        excluded = engine_output.get("suppliers_excluded", [])
        ui_suppliers: list[dict[str, Any]] = []
        for index, supplier in enumerate(shortlist, start=1):
            row = self._supplier_by_id.get(supplier["supplier_id"], {})
            ui_country_code = self._ui_country_code(row.get("country_hq", "CH"))
            coords = COUNTRY_COORDINATES.get(ui_country_code, COUNTRY_COORDINATES["CH"])
            ui_suppliers.append({
                "id": supplier["supplier_id"],
                "name": supplier["supplier_name"],
                "country": COUNTRY_NAMES.get(ui_country_code, row.get("country_hq", "CH")),
                "countryCode": ui_country_code,
                "lat": coords["lat"],
                "lng": coords["lng"],
                "rank": index,
                "accessibility": "open",
                "esgScore": self._to_number(supplier.get("esg_score") or row.get("esg_score")),
                "qualityScore": self._to_number(supplier.get("quality_score") or row.get("quality_score")),
                "riskScore": self._to_number(supplier.get("risk_score") or row.get("risk_score")),
                "unitPrice": supplier.get("unit_price_eur"),
                "totalPrice": supplier.get("total_price_eur"),
                "preferred": bool(supplier.get("preferred")),
                "incumbent": bool(supplier.get("incumbent")),
                "policyCompliant": bool(supplier.get("policy_compliant", True)),
                "standardLeadTimeDays": supplier.get("standard_lead_time_days"),
                "expeditedLeadTimeDays": supplier.get("expedited_lead_time_days"),
                "recommendationNote": supplier.get("recommendation_note"),
            })
        for supplier in excluded:
            row = self._supplier_by_id.get(supplier["supplier_id"], {})
            ui_country_code = self._ui_country_code(row.get("country_hq", "CH"))
            coords = COUNTRY_COORDINATES.get(ui_country_code, COUNTRY_COORDINATES["CH"])
            ui_suppliers.append({
                "id": supplier["supplier_id"],
                "name": supplier["supplier_name"],
                "country": COUNTRY_NAMES.get(ui_country_code, row.get("country_hq", "CH")),
                "countryCode": ui_country_code,
                "lat": coords["lat"],
                "lng": coords["lng"],
                "rank": len(ui_suppliers) + 1,
                "accessibility": "restricted",
                "esgScore": self._to_number(row.get("esg_score")),
                "qualityScore": self._to_number(row.get("quality_score")),
                "riskScore": self._to_number(row.get("risk_score")),
                "unitPrice": None,
                "totalPrice": None,
                "preferred": row.get("preferred_supplier") == "True",
                "incumbent": False,
                "policyCompliant": False,
                "standardLeadTimeDays": None,
                "expeditedLeadTimeDays": None,
                "recommendationNote": supplier.get("reason"),
            })
        return ui_suppliers

    def _build_notifications(self, engine_output: dict[str, Any]) -> list[dict[str, Any]]:
        notifications: list[dict[str, Any]] = []
        for index, escalation in enumerate(engine_output.get("escalations", []), start=1):
            notifications.append({
                "id": index,
                "type": "pending" if escalation.get("blocking") else "approved",
                "message": escalation.get("trigger", "Escalation raised"),
                "time": escalation.get("rule", "policy"),
            })
        base = len(notifications)
        for offset, issue in enumerate(engine_output.get("validation", {}).get("issues_detected", []), start=1):
            notifications.append({
                "id": base + offset,
                "type": "rejected" if issue.get("severity") == "critical" else "pending",
                "message": issue.get("description", "Validation issue detected"),
                "time": issue.get("severity", "issue"),
            })
        return notifications

    def _build_summary(self, engine_output: dict[str, Any]) -> str:
        recommendation = engine_output.get("recommendation", {})
        status = recommendation.get("status", "pending_review").replace("_", " ")
        reason = recommendation.get("reason", "No recommendation summary available.")
        shortlist = engine_output.get("supplier_shortlist") or []
        if shortlist and status != "cannot proceed":
            return f"Status: {status}. Leading supplier: {shortlist[0]['supplier_name']}. {reason}"
        return f"Status: {status}. {reason}"

    def _is_positive_number(self, value: Any) -> bool:
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return False

    def _merge_request_data(self, base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in updates.items():
            if value in (None, "", []):
                continue
            merged[key] = value
        return merged

    def _parse_follow_up_hint(self, message: str, missing_fields: list[dict[str, Any]]) -> dict[str, Any]:
        hinted: dict[str, Any] = {}
        field_names = {item.get("field") for item in missing_fields}
        stripped = message.strip()

        if "quantity" in field_names:
            plain_quantity = re.fullmatch(r"\d{1,6}", stripped.replace(",", ""))
            if plain_quantity:
                hinted["quantity"] = int(stripped.replace(",", ""))

        if "budget_amount" in field_names and "budget_amount" not in hinted:
            budget_amount, currency = self._extract_budget(stripped)
            if budget_amount is not None:
                hinted["budget_amount"] = budget_amount
                if currency:
                    hinted["currency"] = currency

        if "currency" in field_names and "currency" not in hinted:
            currency = self._normalise_currency(stripped)
            if currency:
                hinted["currency"] = currency

        if "country" in field_names:
            country = self._coerce_country(stripped, None, None, stripped)
            if country:
                hinted["country"] = country
                hinted["delivery_countries"] = [country]

        return hinted

    def _normalise_currency(self, token: str | None) -> str | None:
        if not token:
            return None
        lowered = token.lower()
        if lowered in {"eur", "euro", "euros"}:
            return "EUR"
        if lowered in {"usd", "dollar", "dollars"}:
            return "USD"
        if lowered == "chf":
            return "CHF"
        if lowered == "gbp":
            return "GBP"
        return None

    def _normalise_currency_value(self, value: Any) -> str | None:
        if isinstance(value, str):
            return self._normalise_currency(value.strip())
        return None

    def _coerce_int(self, value: Any) -> int | None:
        if value in (None, "", []):
            return None
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            match = re.search(r"\d+", value.replace(",", ""))
            if match:
                return int(match.group(0))
        return None

    def _coerce_float(self, value: Any) -> float | None:
        if value in (None, "", []):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            compact_amount, _ = self._extract_budget(value)
            if compact_amount is not None:
                return compact_amount
            try:
                return float(value.replace(",", "").strip())
            except ValueError:
                return None
        return None

    def _to_number(self, value: str | int | float | None) -> int | None:
        if value in (None, ""):
            return None
        return int(float(value))

    def _ui_country_code(self, country_code: str) -> str:
        return "AE" if country_code == "UAE" else country_code

    def _request_country_code(self, country_code: str) -> str:
        return "UAE" if country_code == "AE" else country_code
