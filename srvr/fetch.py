"""Download only what the audit needs: USD layers and JSON sidecars, never textures."""
from __future__ import annotations

from pathlib import Path

from .index import IndexEntry

PATTERNS = ["*.json", ".metadata/*.json", "**/*.usd", "**/*.usda", "**/*.usdc", "*.usd", "*.usda", "*.usdc"]


def fetch(entry: IndexEntry, cache: Path) -> Path:
    """Return the local folder of ``entry`` at its pinned revision."""
    from huggingface_hub import snapshot_download

    local = cache / entry.repo.replace("/", "__") / entry.revision
    target = local / entry.asset
    if target.exists() and any(target.rglob("com.nvidia.simready.packaging.json")):
        return target
    snapshot_download(
        entry.repo,
        repo_type="dataset",
        revision=entry.revision,
        allow_patterns=[f"{entry.asset}/{p}" for p in PATTERNS],
        local_dir=str(local),
    )
    return target
