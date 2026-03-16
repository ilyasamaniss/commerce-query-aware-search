from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from shopify_ml_demo.catalog import fetch_products
from shopify_ml_demo.filtering import apply_constraints
from shopify_ml_demo.query_analysis import analyze_query
from shopify_ml_demo.retrieval import Retriever
from shopify_ml_demo.schemas import QueryAnalysis, SearchResult


@dataclass(frozen=True)
class BaselineSearchEngine:
    # Benchmark engine: raw query retrieval only, no structured filtering.
    retriever: Retriever

    def search(self, prompt: str, *, products: Optional[List[Dict[str, Any]]] = None, product_limit: int = 50, k: int = 3) -> SearchResult:
        trace: List[Dict[str, Any]] = []

        if products is None:
            trace.append({"tool": "fetch_products", "args": {"limit": product_limit}})
            products = fetch_products(limit=product_limit)
            trace.append({"tool": "fetch_products", "result_count": len(products)})
        else:
            trace.append({"tool": "fetch_products", "reused_catalog_count": len(products)})

        trace.append({"tool": "retriever.search", "args": {"query": prompt, "k": k, "kind": self.retriever.__class__.__name__}})
        matches = self.retriever.search(prompt, k=k)
        trace.append({"tool": "retriever.search", "result_count": len(matches)})

        return SearchResult(answer=_format_answer(matches, intent="search"), picked_products=matches, trace=trace)


@dataclass(frozen=True)
class StructuredSearchEngine:
    # Main engine: query analysis + rewritten retrieval + deterministic filtering.
    retriever: Retriever
    analyzer: Callable[[str], QueryAnalysis] = analyze_query
    retrieval_k: int = 25

    def search(self, prompt: str, *, products: Optional[List[Dict[str, Any]]] = None, product_limit: int = 50, k: int = 3) -> SearchResult:
        trace: List[Dict[str, Any]] = []

        if products is None:
            trace.append({"tool": "fetch_products", "args": {"limit": product_limit}})
            products = fetch_products(limit=product_limit)
            trace.append({"tool": "fetch_products", "result_count": len(products)})
        else:
            trace.append({"tool": "fetch_products", "reused_catalog_count": len(products)})

        trace.append({"tool": "query_analysis", "args": {"query": prompt}})
        analysis = self.analyzer(prompt)
        trace.append(
            {
                "tool": "query_analysis",
                "result": {
                    "intent": analysis.intent,
                    "query_rewrite": analysis.query_rewrite,
                    "candidate_entities": analysis.candidate_entities,
                    "constraints": {
                        "price_min": analysis.constraints.price_min,
                        "price_max": analysis.constraints.price_max,
                        "in_stock": analysis.constraints.in_stock,
                        "color": analysis.constraints.color,
                    },
                },
            }
        )

        retrieval_query = _build_retrieval_query(analysis, prompt)
        trace.append({"tool": "retriever.search", "args": {"query": retrieval_query, "k": self.retrieval_k, "kind": self.retriever.__class__.__name__}})
        candidates = self.retriever.search(retrieval_query, k=self.retrieval_k)
        trace.append({"tool": "retriever.search", "result_count": len(candidates)})

        filtered = apply_constraints(candidates, analysis.constraints)
        trace.append({"tool": "deterministic_filter", "input_count": len(candidates), "result_count": len(filtered)})

        final = filtered[:k]
        return SearchResult(answer=_format_answer(final, intent=analysis.intent), picked_products=final, trace=trace)


def _build_retrieval_query(analysis: QueryAnalysis, prompt: str) -> str:
    # Expand retrieval query with candidate entities to improve recall.
    base = analysis.query_rewrite.strip() or prompt.strip()
    entities = [entity.strip() for entity in analysis.candidate_entities if entity.strip()]
    if not entities:
        return base

    # Keep entity expansion bounded and deterministic.
    entity_part = " ".join(entities[:5])
    return f"{base} {entity_part}".strip()


def _format_answer(matches: List[Dict[str, Any]], intent: str) -> str:
    # Render user-facing answer from matched products.
    if not matches:
        return "I couldn’t find matching products in the store catalog for that request."

    heading = "Here are a few options from the store:"
    if intent == "browse":
        heading = "Here are some sets from the catalog:"
    elif intent == "help":
        heading = "I can help with search, filters, and browsing. Here are a few relevant sets:"

    lines = [heading]
    for product in matches:
        min_price = product["priceRange"]["minVariantPrice"]["amount"]
        currency = product["priceRange"]["minVariantPrice"]["currencyCode"]
        lines.append(f"- {product['title']} (${min_price} {currency}) — /products/{product['handle']}")
    return "\n".join(lines)
