from __future__ import annotations

import json
import os
import re
import uuid
from csv import DictReader
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path
from typing import Any
from urllib import error, request

from supplier_engine import SupplierEngine

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data" / "data"
REQUEST_JSON_PATH = Path("/tmp/request.json") if os.environ.get("VERCEL") else ROOT_DIR / "request.json"
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
    "united arab emirates": "UAE",
    "usa": "US",
    "united states of america": "US",
    "us": "US",
    "u.s.": "US",
    "u.s.a.": "US",
    "great britain": "GB",
    "england": "GB",
    "south korea": "KR",
    "korea, south": "KR",
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


class MoonshotParserError(RuntimeError):
    pass


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
        self._category_names = ", ".join(row["category_l2"] for row in self.categories)

    _NUMERIC_FIELDS = {"quantity", "budget_amount"}

    def run(self, message: str, session_id: str | None = None, answering_field: str | None = None) -> dict[str, Any]:
        run_started = perf_counter()
        session_id = session_id or f"session-{uuid.uuid4().hex[:12]}"
        session_state = self.pending_sessions.get(session_id)

        # Disambiguation: bare number with multiple numeric fields missing and no field clicked
        if session_state and not answering_field:
            missing = session_state.get("missing_fields", [])
            missing_names = {item["field"] for item in missing}
            ambiguous_numeric = missing_names & self._NUMERIC_FIELDS
            text = message.strip()
            is_bare_number = bool(re.match(r"^[\d,]+(?:\.\d+)?\s*[kKmM]?$", text))
            if is_bare_number and len(ambiguous_numeric) > 1:
                questions = self._build_follow_up_questions(missing, session_state["request_json"])
                numeric_questions = [q for q in questions if q["field"] in ambiguous_numeric]
                labels = " or ".join(f'"{q["question"]}"' for q in numeric_questions)
                disambiguation_msg = f"I wasn't sure which question \"{text}\" answers. Please click the question you'd like to answer."
                return {
                    "status": "needs_clarification",
                    "session_id": session_id,
                    "request_json_path": str(REQUEST_JSON_PATH),
                    "request": session_state["request_json"],
                    "parser_source": "disambiguation",
                    "missing_critical_fields": missing,
                    "follow_up_question": "\n\n".join(q["question"] for q in questions),
                    "follow_up_questions": questions,
                    "disambiguation_message": disambiguation_msg,
                    "engine_output": None,
                    "ui": {
                        "summary": disambiguation_msg,
                        "suppliers": [],
                        "notifications": [],
                    },
                }

        parse_started = perf_counter()
        parse_result = self._parse_request(message, session_state, answering_field)
        parse_ms = (perf_counter() - parse_started) * 1000
        missing_fields = self._find_missing_critical_fields(parse_result.request_json)

        REQUEST_JSON_PATH.write_text(
            json.dumps(parse_result.request_json, indent=2),
            encoding="utf-8",
        )

        if missing_fields:
            total_ms = (perf_counter() - run_started) * 1000
            questions = self._build_follow_up_questions(missing_fields, parse_result.request_json)
            question_text = "\n\n".join(q["question"] for q in questions)
            print(
                f"[workflow.timing] session_id={session_id} stage=clarification "
                f"parser={parse_result.source} parse_ms={parse_ms:.1f} total_ms={total_ms:.1f}"
            )
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
                "follow_up_question": question_text,
                "follow_up_questions": questions,
                "engine_output": None,
                "ui": {
                    "summary": question_text,
                    "suppliers": [],
                    "notifications": [],
                },
            }

        engine_started = perf_counter()
        engine_output = self.engine.process(parse_result.request_json)
        engine_ms = (perf_counter() - engine_started) * 1000
        ui_suppliers = self._build_ui_suppliers(engine_output)
        self.pending_sessions[session_id] = {
            "request_json": parse_result.request_json,
            "messages": [
                *(session_state.get("messages", []) if session_state else []),
                {"role": "user", "content": message},
            ],
            "missing_fields": [],
        }
        total_ms = (perf_counter() - run_started) * 1000
        print(
            f"[workflow.timing] session_id={session_id} stage=completed "
            f"parser={parse_result.source} parse_ms={parse_ms:.1f} engine_ms={engine_ms:.1f} total_ms={total_ms:.1f}"
        )
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

    def _try_fast_parse(self, message: str, session_state: dict[str, Any], answering_field: str | None = None) -> ParseResult | None:
        missing = session_state.get("missing_fields", [])
        if not missing:
            return None
        missing_names = {item["field"] for item in missing}
        text = message.strip()
        updates: dict[str, Any] = {}

        # When the user clicked a specific question, a bare number can be
        # unambiguously assigned to that field
        if answering_field and answering_field in missing_names:
            bare_number = self._coerce_float(text) if answering_field == "budget_amount" else None
            bare_int = self._coerce_int(text) if answering_field == "quantity" else None
            if answering_field == "quantity" and bare_int and bare_int > 0:
                updates["quantity"] = bare_int
                merged = self._merge_request_data(session_state["request_json"], updates)
                return ParseResult(
                    request_json=self._normalise_request(merged, session_state["request_json"].get("request_text", ""), message),
                    source="fast-parse",
                )
            if answering_field == "budget_amount" and bare_number and bare_number > 0:
                updates["budget_amount"] = bare_number
                # Also check for currency in the same message
                for word in re.split(r"[\s,.\-;]+", text):
                    cur = self._normalise_currency(word)
                    if cur:
                        updates["currency"] = cur
                        break
                merged = self._merge_request_data(session_state["request_json"], updates)
                return ParseResult(
                    request_json=self._normalise_request(merged, session_state["request_json"].get("request_text", ""), message),
                    source="fast-parse",
                )
            if answering_field == "currency":
                cur = self._normalise_currency(text)
                if cur:
                    updates["currency"] = cur
                    merged = self._merge_request_data(session_state["request_json"], updates)
                    return ParseResult(
                        request_json=self._normalise_request(merged, session_state["request_json"].get("request_text", ""), message),
                        source="fast-parse",
                    )
            if answering_field == "country":
                resolved = self._resolve_country_code(text)
                if resolved:
                    updates["country"] = self._request_country_code(resolved)
                    merged = self._merge_request_data(session_state["request_json"], updates)
                    return ParseResult(
                        request_json=self._normalise_request(merged, session_state["request_json"].get("request_text", ""), message),
                        source="fast-parse",
                    )
            if answering_field == "category_l2":
                coerced = self._coerce_category(None, text)
                if coerced:
                    updates["category_l1"] = coerced["category_l1"]
                    updates["category_l2"] = coerced["category_l2"]
                    merged = self._merge_request_data(session_state["request_json"], updates)
                    return ParseResult(
                        request_json=self._normalise_request(merged, session_state["request_json"].get("request_text", ""), message),
                        source="fast-parse",
                    )
            # answering_field was set but we couldn't parse it — fall through to general parsing

        # --- quantity (must match before budget so we can exclude quantity numbers from budget matching) ---
        qty_span: tuple[int, int] | None = None
        if "quantity" in missing_names:
            # Prefer number followed by a unit keyword
            qty_match = re.search(
                r"(\d[\d,]*)\s*(?:units?|devices?|sets?|pcs?|pieces?|licenses?|seats?|subscriptions?|instances?)\b",
                text, re.IGNORECASE,
            )
            if not qty_match and "budget_amount" not in missing_names:
                # If budget is already known, a bare number is unambiguously quantity
                qty_match = re.search(r"(\d[\d,]*)", text)
            if qty_match:
                qty = self._coerce_int(qty_match.group(1))
                if qty and qty > 0:
                    updates["quantity"] = qty
                    qty_span = qty_match.span()

        # --- budget (+ optional currency) ---
        if "budget_amount" in missing_names:
            # Try contextual patterns first: "budget is X", currency symbols, k/m suffixes
            budget_patterns = [
                r"budget\s+(?:is|of|:)?\s*[€$]?\s*([\d,]+(?:\.\d+)?)\s*([kKmM])?\s*(eur(?:o|os)?|usd|chf|dollars?)?",
                r"[€$]\s*([\d,]+(?:\.\d+)?)\s*([kKmM])?\s*(eur(?:o|os)?|usd|chf|dollars?)?",
                r"([\d,]+(?:\.\d+)?)\s*([kKmM])\s*(eur(?:o|os)?|usd|chf|dollars?)?",
                r"([\d,]+(?:\.\d+)?)\s*()\s*(eur(?:o|os)?|usd|chf|dollars?)",
            ]
            for pattern in budget_patterns:
                for m in re.finditer(pattern, text, re.IGNORECASE):
                    # Skip if this overlaps with the quantity match
                    if qty_span and not (m.end() <= qty_span[0] or m.start() >= qty_span[1]):
                        continue
                    raw_num = m.group(1).replace(",", "")
                    amount = float(raw_num)
                    multiplier = (m.group(2) or "").lower()
                    if multiplier == "k":
                        amount *= 1000
                    elif multiplier == "m":
                        amount *= 1_000_000
                    if amount > 0:
                        updates["budget_amount"] = amount
                    cur_token = m.group(3)
                    if cur_token:
                        cur = self._normalise_currency(cur_token)
                        if cur:
                            updates["currency"] = cur
                    else:
                        # Check for € or $ prefix
                        prefix_char = text[m.start():m.start()+1] if m.start() < len(text) else ""
                        if prefix_char == "€":
                            updates["currency"] = "EUR"
                        elif prefix_char == "$":
                            updates["currency"] = "USD"
                    break
                if "budget_amount" in updates:
                    break

            # Fallback: use a bare number ONLY when it's the only number in the entire message
            # AND quantity is not also missing (ambiguous which field the number belongs to)
            if "budget_amount" not in updates and "quantity" not in missing_names:
                all_numbers = list(re.finditer(r"[\d,]+(?:\.\d+)?", text))
                if len(all_numbers) == 1:
                    raw_num = all_numbers[0].group(0).replace(",", "")
                    amount = float(raw_num)
                    if amount > 0:
                        updates["budget_amount"] = amount

        # --- currency (standalone) ---
        if "currency" in missing_names and "currency" not in updates:
            for word in re.split(r"[\s,.\-;]+", text):
                cur = self._normalise_currency(word)
                if cur:
                    updates["currency"] = cur
                    break

        # --- country ---
        if "country" in missing_names:
            # Split on commas/semicolons but not periods (preserves "U.S.", "U.K.")
            for candidate in re.split(r"[,;]+", text):
                candidate = candidate.strip().rstrip(".")
                # Strip common prefixes
                candidate = re.sub(r"^(?:deliver(?:y|ed)?\s+(?:to|in)|ship(?:ped)?\s+to|to)\s+", "", candidate, flags=re.IGNORECASE).strip()
                # Strip numbers and unit words that might be in the same segment
                candidate = re.sub(r"\d[\d,]*\s*(?:units?|devices?|sets?|pcs?|pieces?|licenses?|seats?)?\s*", "", candidate, flags=re.IGNORECASE).strip()
                if not candidate or re.match(r"^[\d\s.,]+$", candidate):
                    continue
                resolved = self._resolve_country_code(candidate)
                if resolved:
                    updates["country"] = self._request_country_code(resolved)
                    break

        # --- category ---
        if "category_l2" in missing_names:
            for candidate in re.split(r"[,.\-;]+", text):
                coerced = self._coerce_category(None, candidate.strip())
                if coerced:
                    updates["category_l1"] = coerced["category_l1"]
                    updates["category_l2"] = coerced["category_l2"]
                    break

        if not updates:
            return None

        # Use fast parse if it resolved at least one field and the message is short/simple,
        # or if it resolved all missing fields.
        resolved_fields = set(updates.keys())
        all_resolved = (missing_names - {"currency"}).issubset(resolved_fields) if "budget_amount" in missing_names and "budget_amount" not in resolved_fields else missing_names.issubset(resolved_fields)
        word_count = len(text.split())
        # Short answers (≤6 words) are safe even if they only resolve one field
        # Longer answers must resolve everything or fall back to LLM
        if not all_resolved and word_count > 6:
            return None

        merged = self._merge_request_data(session_state["request_json"], updates)
        return ParseResult(
            request_json=self._normalise_request(merged, session_state["request_json"].get("request_text", ""), message),
            source="fast-parse",
        )

    def _parse_request(self, message: str, session_state: dict[str, Any] | None, answering_field: str | None = None) -> ParseResult:
        if session_state:
            fast = self._try_fast_parse(message, session_state, answering_field)
            if fast is not None:
                return fast
            updated = self._update_with_moonshot(session_state["request_json"], message)
            if updated is not None:
                # Don't let the LLM invent a currency the user didn't mention
                if not session_state["request_json"].get("currency") and updated.get("currency"):
                    if not self._message_mentions_currency(message):
                        updated["currency"] = None
                merged_update = self._merge_request_data(session_state["request_json"], updated)
                return ParseResult(
                    request_json=self._normalise_request(merged_update, session_state["request_json"].get("request_text", ""), message),
                    source="moonshot-update",
                )
            raise RuntimeError("LLM parser unavailable for follow-up clarification.")

        moonshot_result = self._parse_with_moonshot(message)
        if moonshot_result is not None:
            # Don't let the LLM invent a currency the user didn't mention
            if moonshot_result.get("currency"):
                if not self._message_mentions_currency(message):
                    moonshot_result["currency"] = None
            return ParseResult(
                request_json=self._normalise_request(moonshot_result, message),
                source="moonshot",
            )
        raise RuntimeError("LLM parser unavailable for request intake.")

    def _parse_with_moonshot(self, message: str) -> dict[str, Any] | None:
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            return None

        base_url = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1").rstrip("/")
        model = os.getenv("MOONSHOT_MODEL", "kimik2.5")
        system_prompt = (
            "Convert the user's procurement message into JSON only. "
            "Do not add prose or markdown. "
            "Set unknown values to null. "
            "Use category_l2 from this allowed list only: "
            f"{self._category_names}. "
            f"Today's date is {datetime.today().strftime('%Y-%m-%d')}"
            "Return these keys exactly: "
            "category_l1, category_l2, title, quantity, unit_of_measure, "
            "budget_amount, currency, required_by_date, country, site, delivery_countries, "
            "preferred_supplier_mentioned, incumbent_supplier, contract_type_requested, "
            "data_residency_constraint, esg_requirement, business_unit, requester_role, "
            "request_language, scenario_tags."
        )
        call_started = perf_counter()
        result = self._call_moonshot(base_url, api_key, model, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ])
        call_ms = (perf_counter() - call_started) * 1000
        print(f"[moonshot.timing] mode=intake model={model} duration_ms={call_ms:.1f}")
        return result

    def _update_with_moonshot(self, current_request: dict[str, Any], message: str) -> dict[str, Any] | None:
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            return None

        base_url = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1").rstrip("/")
        model = os.getenv("MOONSHOT_MODEL", "kimik2.5")
        system_prompt = (
            "Update an existing procurement request JSON using the user's clarification. "
            "Return JSON only. "
            "Keep existing values unless the clarification changes them. "
            "Do not invent unsupported values. "
            "Preserve the same response keys as the current request."
        )
        user_prompt = json.dumps({
            "current_request": current_request,
            "user_clarification": message,
        })
        call_started = perf_counter()
        result = self._call_moonshot(base_url, api_key, model, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        call_ms = (perf_counter() - call_started) * 1000
        print(f"[moonshot.timing] mode=update model={model} duration_ms={call_ms:.1f}")
        return result

    def _call_moonshot(self, base_url: str, api_key: str, model: str, messages: list[dict[str, str]]) -> dict[str, Any] | None:
        payload = {
            "model": model,
            "temperature": 0,
            "thinking": {
                "type": "disabled"
            },
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
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            detail = self._extract_error_message(body) or exc.reason
            raise MoonshotParserError(f"Moonshot parser request failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise MoonshotParserError(f"Moonshot parser network error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise MoonshotParserError("Moonshot parser request timed out.") from exc
        except json.JSONDecodeError as exc:
            raise MoonshotParserError("Moonshot parser returned invalid JSON.") from exc

        content = raw.get("choices", [{}])[0].get("message", {}).get("content")
        if not isinstance(content, str):
            raise MoonshotParserError("Moonshot parser response did not include message content.")
        try:
            return json.loads(self._strip_json_wrapping(content))
        except json.JSONDecodeError as exc:
            raise MoonshotParserError("Moonshot parser content was not valid JSON.") from exc

    def _strip_json_wrapping(self, content: str) -> str:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        return cleaned

    def _extract_error_message(self, body: str) -> str | None:
        if not body:
            return None
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return body.strip()[:200]
        error_obj = parsed.get("error")
        if isinstance(error_obj, dict):
            message = error_obj.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        message = parsed.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
        return body.strip()[:200]

    def _normalise_request(
        self,
        parsed: dict[str, Any],
        original_message: str,
        follow_up_message: str | None = None,
    ) -> dict[str, Any]:
        combined_message = original_message if not follow_up_message else f"{original_message}\nFollow-up: {follow_up_message}"
        category = self._coerce_category(parsed.get("category_l1"), parsed.get("category_l2"))
        country = self._coerce_country(parsed.get("country"), parsed.get("site"), parsed.get("delivery_countries"))
        currency = self._normalise_currency_value(parsed.get("currency"))
        request_id = parsed.get("request_id") or f"REQ-{uuid.uuid4().hex[:8].upper()}"
        title = parsed.get("title") or (f"{category['category_l2']} request" if category else "Procurement request")

        quantity = self._coerce_int(parsed.get("quantity"))
        budget_amount = self._coerce_float(parsed.get("budget_amount"))
        preferred = self._find_supplier_name(parsed.get("preferred_supplier_mentioned"))
        incumbent = self._find_supplier_name(parsed.get("incumbent_supplier") or "")
        delivery_countries = self._coerce_delivery_countries(parsed.get("delivery_countries"), country)

        return {
            "request_id": request_id,
            "created_at": parsed.get("created_at") or datetime.now(tz=timezone.utc).isoformat(),
            "request_channel": parsed.get("request_channel") or "frontend_chat",
            "request_language": parsed.get("request_language") or "en",
            "business_unit": parsed.get("business_unit") or "Frontend Intake",
            "country": country,
            "site": parsed.get("site") or country,
            "requester_id": parsed.get("requester_id") or "frontend-user",
            "requester_role": parsed.get("requester_role") or "Requester",
            "submitted_for_id": parsed.get("submitted_for_id") or "frontend-user",
            "category_l1": category["category_l1"] if category else None,
            "category_l2": category["category_l2"] if category else None,
            "title": title,
            "request_text": combined_message.strip(),
            "currency": currency,
            "budget_amount": budget_amount,
            "quantity": quantity,
            "unit_of_measure": parsed.get("unit_of_measure") or (category["typical_unit"] if category else None),
            "required_by_date": parsed.get("required_by_date"),
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
                missing.append({"field": field_name, "reason": "invalid", "criteria": criteria, "attempted_value": value})
            elif field_name == "currency" and value not in criteria.get("values", []):
                missing.append({"field": field_name, "reason": "invalid", "criteria": criteria, "attempted_value": value})
            elif field_name == "quantity" and not self._is_positive_number(value):
                missing.append({"field": field_name, "reason": "invalid", "criteria": criteria, "attempted_value": value})
            elif field_name == "budget_amount" and not self._is_positive_number(value):
                missing.append({"field": field_name, "reason": "invalid", "criteria": criteria, "attempted_value": value})
        missing_fields = {item["field"] for item in missing}
        if "currency" in missing_fields and "budget_amount" in missing_fields:
            missing = [item for item in missing if item["field"] != "currency"]
        return missing

    def _get_typical_unit(self, request_json: dict[str, Any]) -> str | None:
        l1 = request_json.get("category_l1", "")
        l2 = request_json.get("category_l2", "")
        for row in self.categories:
            if l1 and l2 and row["category_l1"].lower() == str(l1).lower() and row["category_l2"].lower() == str(l2).lower():
                return row.get("typical_unit") or None
            if l2 and not l1 and row["category_l2"].lower() == str(l2).lower():
                return row.get("typical_unit") or None
        return None

    def _build_follow_up_questions(self, missing_fields: list[dict[str, Any]], request_json: dict[str, Any] | None = None) -> list[dict[str, str]]:
        typical_unit = self._get_typical_unit(request_json) if request_json else None
        questions: list[dict[str, str]] = []
        for item in missing_fields:
            field = item["field"]
            criteria = item["criteria"]
            if field == "category_l2":
                attempted_value = item.get("attempted_value")
                if attempted_value:
                    questions.append({"field": field, "question": f"{attempted_value} is not a supported product category. Please provide a valid category."})
                else:
                    examples = ", ".join(criteria["values"][:5])
                    questions.append({"field": field, "question": f"What product are you buying? Use a category such as {examples}."})
            elif field == "country":
                attempted_value = item.get("attempted_value")
                if item.get("reason") == "invalid" and attempted_value:
                    questions.append({"field": field, "question": f"I interpreted the delivery country as {attempted_value}, but that country is not supported by the current policy dataset."})
                else:
                    questions.append({"field": field, "question": "Which delivery country should I use?"})
            elif field == "quantity":
                attempted_value = item.get("attempted_value")
                if item.get("reason") == "invalid" and attempted_value:
                    questions.append({"field": field, "question": f"I could not use the quantity value {attempted_value}."})
                elif typical_unit:
                    unit_display = typical_unit.replace("_", " ")
                    questions.append({"field": field, "question": f"How many {unit_display}s do you need?"})
                else:
                    questions.append({"field": field, "question": "How many units do you need?"})
            elif field == "budget_amount":
                attempted_value = item.get("attempted_value")
                questions.append({"field": field, "question":
                    f"I could not use the budget value {attempted_value}." if item.get("reason") == "invalid" and attempted_value
                    else "What is your total budget?"
                })
            elif field == "currency":
                attempted_value = item.get("attempted_value")
                questions.append({"field": field, "question":
                    f"I interpreted the currency as {attempted_value}, but we can only place orders in EUR, CHF, or USD"
                    if item.get("reason") == "invalid" and attempted_value
                    else "Which currency should I use?"
                })
        if not questions:
            field_names = ", ".join(item["field"] for item in missing_fields)
            questions.append({"field": "unknown", "question": f"I still need critical request details before I can run supplier matching: please provide valid values for {field_names}."})
        return questions

    def _coerce_category(self, category_l1: str | None, category_l2: str | None) -> dict[str, str] | None:
        if category_l1 and category_l2:
            for row in self.categories:
                if row["category_l1"].lower() == str(category_l1).lower() and row["category_l2"].lower() == str(category_l2).lower():
                    return row
        if category_l2:
            for row in self.categories:
                if row["category_l2"].lower() == str(category_l2).lower():
                    return row
        return None

    def _coerce_country(self, country: str | None, site: str | None, delivery_countries: list[str] | None) -> str | None:
        for candidate in [country, *(delivery_countries or []), site]:
            if not candidate:
                continue
            resolved = self._resolve_country_code(candidate)
            if resolved:
                return self._request_country_code(resolved)
        return None

    def _coerce_delivery_countries(self, countries: Any, fallback_country: str | None) -> list[str]:
        if isinstance(countries, str):
            raw_candidates = [countries]
        elif isinstance(countries, list):
            raw_candidates = countries
        else:
            raw_candidates = []

        normalised: list[str] = []
        for candidate in raw_candidates:
            resolved = self._resolve_country_code(candidate)
            if not resolved:
                continue
            request_code = self._request_country_code(resolved)
            if request_code not in normalised:
                normalised.append(request_code)

        if not normalised and fallback_country:
            return [fallback_country]
        return normalised

    def _resolve_country_code(self, candidate: Any) -> str | None:
        if not isinstance(candidate, str):
            return None
        cleaned = candidate.strip().lower()
        if not cleaned:
            return None
        cleaned = re.sub(r"[()]+", " ", cleaned)
        cleaned = re.sub(r"[.,]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned in CITY_TO_COUNTRY:
            return CITY_TO_COUNTRY[cleaned]
        exact = COUNTRY_ALIASES.get(cleaned)
        if exact:
            return exact

        # Accept phrases like "deliver to Zurich office" or "ship to Switzerland".
        for city, code in sorted(CITY_TO_COUNTRY.items(), key=lambda item: len(item[0]), reverse=True):
            if re.search(rf"\b{re.escape(city)}\b", cleaned):
                return code
        for alias, code in sorted(COUNTRY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", cleaned):
                return code
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
                "pricingMessage": (
                    supplier.get("pricing_note")
                    or (supplier.get("pricing") or {}).get("_pricing_note")
                    or (supplier.get("pricing") or {}).get("_currency_note")
                ),
                "preferred": bool(supplier.get("preferred")),
                "incumbent": bool(supplier.get("incumbent")),
                "policyCompliant": bool(supplier.get("policy_compliant", True)),
                "standardLeadTimeDays": supplier.get("standard_lead_time_days"),
                "expeditedLeadTimeDays": supplier.get("expedited_lead_time_days"),
                "recommendationNote": supplier.get("recommendation_note"),
                "confidencePct": supplier.get("confidence_pct"),
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
                "pricingMessage": "No valid pricing row for this supplier in the requested region and currency.",
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
            approver = escalation.get("escalate_to")
            reason = escalation.get("trigger", "Escalation raised")
            notifications.append({
                "id": index,
                "type": "rejected" if escalation.get("blocking") else "pending",
                "message": (
                    f"{'Blocked' if escalation.get('blocking') else 'Approval required'}: {reason}"
                    + (f" Routed to {approver}." if approver else "")
                ),
                "time": escalation.get("rule", "policy"),
            })
        base = len(notifications)
        for offset, issue in enumerate(engine_output.get("validation", {}).get("issues_detected", []), start=1):
            notifications.append({
                "id": base + offset,
                "type": "rejected" if issue.get("severity") == "critical" else "pending",
                "message": (
                    f"{issue.get('description', 'Validation issue detected')} "
                    f"Action: {issue.get('action_required', 'Review required.')}"
                ),
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

    def _message_mentions_currency(self, message: str) -> bool:
        lowered = message.lower()
        if re.search(r"[€$]", message):
            return True
        currency_terms = {"eur", "euro", "euros", "usd", "dollar", "dollars", "chf", "gbp", "franc", "francs", "pound", "pounds", "sterling"}
        for word in re.split(r"[\s,.\-;]+", lowered):
            if word in currency_terms:
                return True
        return False

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
