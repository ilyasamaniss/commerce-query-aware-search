from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, Optional

from shopify_ml_demo.config import load_ollama_settings
from shopify_ml_demo.schemas import QueryAnalysis, QueryConstraints

_ALLOWED_INTENTS = {"search", "browse", "help", "none"}


class QueryAnalysisError(RuntimeError):
    # Raised when structured query analysis cannot complete successfully.
    pass


def _to_optional_float(value: Any) -> Optional[float]:
    # Convert numeric-like values from model output to float.
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _to_optional_bool(value: Any) -> Optional[bool]:
    # Convert bool-like strings from model output.
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1"}:
            return True
        if text in {"false", "no", "0"}:
            return False
    return None


def _sanitize_analysis(raw: Dict[str, Any], query: str) -> QueryAnalysis:
    # Validate and normalize raw JSON into deterministic structured analysis.
    intent_raw = str(raw.get("intent", "search")).strip().lower()
    intent = intent_raw if intent_raw in _ALLOWED_INTENTS else "search"

    query_rewrite = str(raw.get("query_rewrite", "")).strip() or query.strip()

    constraints_raw = raw.get("constraints", {})
    if not isinstance(constraints_raw, dict):
        constraints_raw = {}

    color_value = constraints_raw.get("color")
    color = str(color_value).strip().lower() if isinstance(color_value, str) and color_value.strip() else None

    constraints = QueryConstraints(
        price_min=_to_optional_float(constraints_raw.get("price_min")),
        price_max=_to_optional_float(constraints_raw.get("price_max")),
        in_stock=_to_optional_bool(constraints_raw.get("in_stock")),
        color=color,
    )

    entities_raw = raw.get("candidate_entities", [])
    candidate_entities: list[str] = []
    if isinstance(entities_raw, list):
        for item in entities_raw:
            if not isinstance(item, str):
                continue
            entity = item.strip().lower()
            if entity and entity not in candidate_entities:
                candidate_entities.append(entity)

    return QueryAnalysis(
        intent=intent,
        query_rewrite=query_rewrite,
        constraints=constraints,
        candidate_entities=candidate_entities,
    )


def analyze_query(query: str) -> QueryAnalysis:
    # Call local Ollama and return validated structured analysis.
    cfg = load_ollama_settings()
    payload = {
        "model": cfg.model,
        "prompt": f"""
You are a query analyzer for an ecommerce search engine.

Your job is to read the user's search query and return ONLY valid JSON.

Return this exact structure:
{{
  "intent": "search" | "browse" | "help" | "none",
  "query_rewrite": "string",
  "candidate_entities": ["string"],
  "constraints": {{
    "price_min": number or null,
    "price_max": number or null,
    "in_stock": true | false | null,
    "color": string or null
  }}
}}

Rules:
- "under 100" means price_max = 100
- "above 50" means price_min = 50
- "between 50 and 100" means price_min = 50 and price_max = 100
- "in stock", "available now", "not sold out" means in_stock = true
- query_rewrite should preserve important retrieval words
- candidate_entities should contain likely Pokemon names implied by the query
- candidate_entities should be ordered from strongest semantic match to weakest semantic match for the user's query
- if several entities are similarly plausible, break ties by general popularity or recognizability
- do not rank a globally popular Pokemon ahead of a clearly better query match
- if the query is broad or ambiguous, return several likely Pokemon names
- if no specific Pokemon can be inferred, return an empty list
- for color, type, move, or descriptive queries, infer likely Pokemon names when possible
- return JSON only, no markdown, no explanation
- for narrow queries, return only the few strongest matches

Examples:
Query: "yellow pokemon"
candidate_entities: ["pikachu", "psyduck", "jolteon"]

Query: "blue pokemon"
candidate_entities: ["squirtle", "vaporeon", "mudkip", "lapras", "gyarados"]

Query: "red dragon pokemon"
candidate_entities: ["charizard", "gyarados", "dragonite"]

Query: "pokemon that knows fly"
candidate_entities: ["charizard", "dragonite", "rayquaza"]

User query:
{query}
""",
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "query_rewrite": {"type": "string"},
                "candidate_entities": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "constraints": {
                    "type": "object",
                    "properties": {
                        "price_min": {"type": ["number", "null"]},
                        "price_max": {"type": ["number", "null"]},
                        "in_stock": {"type": ["boolean", "null"]},
                        "color": {"type": ["string", "null"]},
                    },
                    "required": ["price_min", "price_max", "in_stock", "color"],
                },
            },
            "required": ["intent", "query_rewrite", "candidate_entities", "constraints"],
        },
    }

    req = urllib.request.Request(
        cfg.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise QueryAnalysisError(f"Query analyzer request failed for model '{cfg.model}' at '{cfg.url}': {exc}") from exc

    raw_response = response_data.get("response", "{}")
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise QueryAnalysisError(f"Query analyzer returned invalid JSON for model '{cfg.model}': {raw_response!r}") from exc
    if not isinstance(parsed, dict):
        raise QueryAnalysisError(f"Query analyzer returned non-object JSON for model '{cfg.model}': {type(parsed).__name__}")

    return _sanitize_analysis(parsed, query)
