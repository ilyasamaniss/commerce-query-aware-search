from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from shopify_ml_demo.evaluation import normalize_query_analyzer
from shopify_ml_demo.retrieval import RETRIEVER_KINDS, Retriever, build_retriever
from shopify_ml_demo.schemas import QueryAnalysis
from shopify_ml_demo.search import BaselineSearchEngine, StructuredSearchEngine

MAX_K = 20


class BadRequest(ValueError):
    # Raised for invalid client parameters; mapped to HTTP 400.
    pass


def product_summary(product: Dict[str, Any]) -> Dict[str, Any]:
    # Reduce a Storefront product to the fields the demo UI renders.
    price = (product.get("priceRange") or {}).get("minVariantPrice") or {}
    image = product.get("featuredImage") or {}
    return {
        "title": product.get("title", ""),
        "handle": product.get("handle", ""),
        "price": float(price["amount"]) if price.get("amount") is not None else None,
        "currency": price.get("currencyCode"),
        "available": bool(product.get("availableForSale", False)),
        "imageUrl": image.get("url"),
        "tags": list(product.get("tags") or []),
    }


@dataclass
class SearchService:
    # Runs one search request over a fixed catalog, caching retrievers per configuration.
    products: List[Dict[str, Any]]
    embedder_factory: Optional[Callable[[], Any]] = None
    analyzer: Optional[Callable[[str], QueryAnalysis]] = None
    _retrievers: Dict[str, Retriever] = field(default_factory=dict)

    def retriever(self, kind: str) -> Retriever:
        if kind not in RETRIEVER_KINDS:
            raise BadRequest(f"retriever must be one of {', '.join(RETRIEVER_KINDS)}")
        if kind not in self._retrievers:
            embedder = self.embedder_factory() if self.embedder_factory and kind in {"dense", "hybrid"} else None
            self._retrievers[kind] = build_retriever(kind, self.products, embedder=embedder)
        return self._retrievers[kind]

    def config(self) -> Dict[str, Any]:
        return {"retrievers": list(RETRIEVER_KINDS), "catalogSize": len(self.products), "maxK": MAX_K}

    def search(self, params: Dict[str, str]) -> Dict[str, Any]:
        query = (params.get("q") or "").strip()
        if not query:
            raise BadRequest("q is required")
        kind = (params.get("retriever") or "bm25").strip().lower()
        try:
            k = int(params.get("k") or 5)
        except ValueError as exc:
            raise BadRequest("k must be an integer") from exc
        if not 1 <= k <= MAX_K:
            raise BadRequest(f"k must be between 1 and {MAX_K}")
        analyzer_name = normalize_query_analyzer(params.get("analyzer"))

        retriever = self.retriever(kind)
        if analyzer_name:
            os.environ["OLLAMA_MODEL"] = analyzer_name
            engine: Any = (
                StructuredSearchEngine(retriever=retriever, analyzer=self.analyzer)
                if self.analyzer is not None
                else StructuredSearchEngine(retriever=retriever)
            )
        else:
            engine = BaselineSearchEngine(retriever=retriever)

        result = engine.search(query, products=self.products, k=k)
        return {
            "query": query,
            "retriever": kind,
            "analyzer": analyzer_name or "none",
            "products": [product_summary(p) for p in result.picked_products],
            "trace": result.trace,
        }


def handle_request(service: SearchService, path: str) -> Tuple[int, Dict[str, Any]]:
    # Route one GET request to a status code and JSON body.
    url = urlparse(path)
    params = {key: values[-1] for key, values in parse_qs(url.query).items()}
    try:
        if url.path == "/api/config":
            return 200, service.config()
        if url.path == "/api/search":
            return 200, service.search(params)
        return 404, {"error": "not found"}
    except BadRequest as exc:
        return 400, {"error": str(exc)}
    except Exception as exc:
        # Analyzer or embedder backends being down surfaces as 503, matching fail-closed search.
        return 503, {"error": str(exc)}


def make_handler(service: SearchService, allow_origin: str = "*") -> type:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            status, body = handle_request(service, self.path)
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", allow_origin)
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def serve(service: SearchService, host: str = "127.0.0.1", port: int = 8000) -> None:
    # Serve the JSON API until interrupted.
    server = ThreadingHTTPServer((host, port), make_handler(service))
    print(f"Search API on http://{host}:{port} ({len(service.products)} products)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
