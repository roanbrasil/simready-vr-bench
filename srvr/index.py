"""Parse SimReady Central submission index files.

Each file under ``simready-central/submissions/<namespace>/<dataset>`` holds one
URL per package, pinned to the Hugging Face commit it was reviewed at::

    https://huggingface.co/datasets/<ns>/<ds>/resolve/<sha>/<asset>/com.nvidia.simready.packaging.json
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

_URL = re.compile(
    r"https://huggingface\.co/datasets/(?P<repo>[^/]+/[^/]+)/resolve/(?P<rev>[0-9a-f]{6,40})/"
    r"(?P<asset>.+?)/com\.nvidia\.simready\.packaging\.json"
)


@dataclass(frozen=True)
class IndexEntry:
    repo: str        # "<namespace>/<dataset>" on the HF Hub
    revision: str    # pinned commit sha
    asset: str       # folder name inside the dataset
    oem: str         # namespace, used as the vendor label

    @property
    def key(self) -> str:
        return f"{self.repo}/{self.asset}"


def parse_line(line: str) -> IndexEntry | None:
    m = _URL.search(line.strip())
    if not m:
        return None
    repo = m["repo"]
    return IndexEntry(repo=repo, revision=m["rev"], asset=m["asset"], oem=repo.split("/")[0])


def read_index(submissions_dir: Path) -> list[IndexEntry]:
    entries: list[IndexEntry] = []
    for f in sorted(p for p in submissions_dir.glob("*/*") if p.is_file()):
        for line in f.read_text(encoding="utf-8").splitlines():
            e = parse_line(line)
            if e:
                entries.append(e)
    return entries


def dedupe(entries: list[IndexEntry]) -> list[IndexEntry]:
    """An index may list several reviewed revisions of the same asset; keep the last one listed."""
    last: dict[str, IndexEntry] = {}
    for e in entries:
        last.pop(e.key, None)
        last[e.key] = e
    return list(last.values())


def stratified_sample(entries: list[IndexEntry], per_oem_cap: int, seed: int = 7) -> list[IndexEntry]:
    """Keep every asset of small vendors; cap large ones at ``per_oem_cap`` (random, seeded).

    The two bulk datasets (nvidia, PalatialSim) hold ~93% of the index. Without a cap
    they would dominate every per-vendor number.
    """
    by_oem: dict[str, list[IndexEntry]] = {}
    for e in entries:
        by_oem.setdefault(e.oem, []).append(e)
    rng = random.Random(seed)
    out: list[IndexEntry] = []
    for oem in sorted(by_oem):
        group = by_oem[oem]
        out.extend(group if len(group) <= per_oem_cap else rng.sample(group, per_oem_cap))
    return out
