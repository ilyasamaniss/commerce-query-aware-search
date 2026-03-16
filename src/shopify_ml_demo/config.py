from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv() -> bool:
        # No-op fallback for environments without python-dotenv.
        return False


@dataclass(frozen=True)
class ShopifySettings:
    # Runtime config for Shopify Storefront API access.
    store_domain: str
    storefront_token: str
    api_version: str


@dataclass(frozen=True)
class OllamaSettings:
    # Runtime config for local Ollama query analysis.
    url: str
    model: str


def load_shopify_settings() -> ShopifySettings:
    # Load and validate required Shopify env vars.
    load_dotenv()
    domain = os.getenv("SHOPIFY_STORE_DOMAIN", "").strip()
    token = os.getenv("SHOPIFY_STOREFRONT_TOKEN", "").strip()
    version = os.getenv("SHOPIFY_STOREFRONT_API_VERSION", "2026-01").strip() or "2026-01"

    if not domain:
        raise RuntimeError("Missing SHOPIFY_STORE_DOMAIN in environment")
    if not token:
        raise RuntimeError("Missing SHOPIFY_STOREFRONT_TOKEN in environment")

    return ShopifySettings(store_domain=domain, storefront_token=token, api_version=version)


def load_ollama_settings() -> OllamaSettings:
    # Load optional Ollama settings with sensible local defaults.
    load_dotenv()
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate").strip()
    model = os.getenv("OLLAMA_MODEL", "gemma3:4b").strip()
    return OllamaSettings(url=url, model=model)
