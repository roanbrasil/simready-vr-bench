"""Thin client for the public USD Search instance behind SimReady Central.

Request body mirrors ``usd-search/skills/search/references/search-body.json``.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass

# SimReady Central's own search endpoint (indexes the Hugging Face packages). The general public
# USD Search instance, https://search.simready.omniverse.nvidia.com, indexes Isaac's S3 content library.
DEFAULT_URL = os.environ.get("USD_SEARCH_API_URL", "https://nvidia-simready-central.hf.space")


def resolvable(url: str) -> str:
    """Hits point at Hugging Face *blob* pages (HTML). A USD resolver needs the raw *resolve* URL."""
    if "huggingface.co/" in url and "/blob/" in url:
        return url.replace("/blob/", "/resolve/", 1)
    return url


@dataclass
class Hit:
    name: str
    url: str          # base_key: a resolvable USD URL
    score: float
    caption: str | None
    validated_profile: str | None
    has_rigid_body: bool
    is_robot: bool = False
    image_key: str | None = None   # SigLIP2-matched thumbnail id; /images?asset_url= 404s on the public instance

    @property
    def thumbnail_url(self) -> str:
        if self.image_key:
            return f"{DEFAULT_URL}/images?image_key={self.image_key}"
        return f"{DEFAULT_URL}/images?asset_url={urllib.parse.quote(self.url, safe='')}"


def build_body(query: str, limit: int = 12) -> dict:
    return {
        "hybrid_text_query": query,
        "vector_queries": [{"field_name": "siglip2-embedding.embedding", "query_type": "text", "query": query}],
        "file_extension_include": "usd*",
        "return_metadata": True,
        "return_usd_properties": True,
        "return_vision_generated_metadata": True,
        "return_images": False,
        "limit": limit,
    }


def parse_hits(payload: dict) -> list[Hit]:
    hits = []
    for h in payload.get("hits", []):
        s = h.get("source", {})
        props = {p.get("name"): p.get("value") for p in s.get("usd_properties", [])}
        keys = [mv.get("image") for ex in (h.get("metadata") or {}).get("explanations", [])
                for mv in ex.get("matched_vectors", []) if mv.get("image")]
        cap = (s.get("plugin_rendering_to_vision_metadata_metadata_vlm_generated") or [{}])[0].get("value_text")
        hits.append(Hit(
            name=s.get("name", "?"),
            url=resolvable(s.get("base_key", "")),
            score=float(h.get("rrf_score") or h.get("score") or 0.0),
            caption=cap,
            validated_profile=props.get("simready_metadata_validation__profile"),
            has_rigid_body=any("rigidBody" in (k or "") for k in props),
            is_robot=any((k or "").startswith("isaac:physics:robot") for k in props),
            image_key=keys[0] if keys else None,
        ))
    return hits


def search(query: str, limit: int = 12, url: str = DEFAULT_URL, timeout: float = 30.0) -> list[Hit]:
    req = urllib.request.Request(
        f"{url}/search_hybrid",
        data=json.dumps(build_body(query, limit)).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return parse_hits(json.load(r))
