"""Sample SimReady Central, fetch packages at their pinned commits, audit, write CSV/JSONL.

    python scripts/run_audit.py --submissions <simready-central>/submissions --cap 60 --out results/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srvr.audit import audit_asset  # noqa: E402
from srvr.fetch import fetch  # noqa: E402
from srvr.index import dedupe, read_index, stratified_sample  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=60, help="max assets per vendor namespace")
    ap.add_argument("--cache", type=Path, default=Path("cache"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    raw = read_index(args.submissions)
    entries = dedupe(raw)
    sample = stratified_sample(entries, args.cap)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sample.json").write_text(json.dumps(
        {"index_lines": len(raw), "index_total": len(entries), "sample": [e.__dict__ for e in sample]}, indent=1))
    print(f"index: {len(entries)} packages, sample: {len(sample)}")

    def work(e):
        for attempt in range(3):
            try:
                d = fetch(e, args.cache)
                return audit_asset(e.key, e.oem, d)
            except Exception as ex:  # rate limit / transient
                err = ex
                time.sleep(5 * (attempt + 1))
        from srvr.audit import AssetAudit, Declared
        return AssetAudit(e.key, e.oem, e.asset, None, Declared(), {}, error=f"FETCH: {err}"[:300])

    rows = []
    with ThreadPoolExecutor(args.workers) as ex, open(args.out / "audit.jsonl", "w") as jl:
        futs = [ex.submit(work, e) for e in sample]
        for i, f in enumerate(as_completed(futs), 1):
            a = f.result()
            r = a.to_row()
            rows.append(r)
            jl.write(json.dumps(r, default=str) + "\n")
            if i % 25 == 0:
                print(f"  {i}/{len(sample)}")
    cols = sorted({k for r in rows for k in r})
    with open(args.out / "audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
