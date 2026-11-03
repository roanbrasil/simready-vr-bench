"""Aggregate audit statistics, computed from results/ and the package cache.

    python scripts/summarize.py --results results --cache /tmp/srvr-cache
writes results/summary.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pandas as pd

BY_DESIGN = {"JOINT_WITHOUT_LIMITS"}  # continuous wheels, drums, doors: absence of limits is intended


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--cache", type=Path, required=True)
    a = ap.parse_args()

    meta = json.loads((a.results / "sample.json").read_text())
    df = pd.read_json(a.results / "audit.jsonl", lines=True)
    ok = df[df.error.isna()]
    findings = ok.findings.fillna("").str.split(";").explode()
    findings = findings[findings != ""]

    grasp, gripper, failing_tests = collections.Counter(), [], collections.Counter()
    grasp_reasons = collections.Counter()
    grasp_pkgs = set()
    for e in meta["sample"]:
        base = a.cache / e["repo"].replace("/", "__") / e["revision"] / e["asset"] / ".metadata"
        p = base / f"{e['asset']}_benchmark.json"
        if not p.exists():
            continue
        b = json.loads(p.read_text())
        for t in b.get("results", []):
            if t["status"] == "fail":
                failing_tests[t["test_name"]] += 1
            if t["test_name"] == "grasp_and_lift":
                grasp[t["status"]] += 1
                grasp_pkgs.add(e["asset"])
                if t["status"] == "skipped":
                    m = t["message"]
                    grasp_reasons["no grasp_identifier prims" if "grasp_identifier" in m
                                  else "needs FET_003_NEWTON first" if "FET_003_NEWTON" in m else m[:60]] += 1
        g = [t for t in b.get("results", []) if t["test_name"].startswith("gripper_close")]
        if g:
            v = json.loads((base / f"{e['asset']}_validation.json").read_text())
            fs = next(iter(v.values()))["features_summary"]
            gripper.append({
                "asset": e["asset"], "oem": e["oem"],
                "static_FET_028_STANDARD": fs.get("FET_028_STANDARD", {}).get("passed"),
                "static_FET_028_ISAAC": fs.get("FET_028_ISAAC", {}).get("passed"),
                "runtime_statuses": collections.Counter(t["status"] for t in g),
                "benchmark_passed": b.get("passed"),
            })

    nonaif = ok[ok.profile != "AIF-Entity"]
    bench = ok[ok.has_benchmark_sidecar]
    s = {
        "index_lines": meta["index_lines"], "index_unique": meta["index_total"],
        "audited": int(len(df)), "audit_errors": int(df.error.notna().sum()), "vendors": int(ok.oem.nunique()),
        "per_vendor": ok.oem.value_counts().to_dict(),
        "profiles": ok.profile.value_counts().to_dict(),
        "declared_source": ok.declared_source.value_counts().to_dict(),
        "declared_source_vendors": {k: sorted(v) for k, v in ok.groupby("declared_source").oem.unique().items()},
        "variant_scheme": ok["PhysX.variant_scheme"].value_counts().to_dict(),
        "variant_scheme_vendors": {k: sorted(v) for k, v in ok.groupby("PhysX.variant_scheme").oem.unique().items()},
        "grabbable_nonaif": [int(nonaif["PhysX.grabbable"].sum()), int(len(nonaif))],
        "grabbable_aif": [int(ok[ok.profile == "AIF-Entity"]["PhysX.grabbable"].sum()), int((ok.profile == "AIF-Entity").sum())],
        "findings": findings.value_counts().to_dict(),
        "findings_assets": {f: ok[ok.findings.fillna("").str.contains(f)].key.tolist() for f in findings.unique()},
        "runtime_divergence": int(findings.eq("RUNTIME_BODY_COUNT_DIVERGES").sum()),
        "runtime_compared": {str(k): int(v) for k, v in ok[[f"{rt}.variant_available" for rt in ("PhysX", "Newton", "MuJoCo")]]
                             .fillna(False).astype(bool).sum(axis=1).value_counts().sort_index().items()},
        "benchmark": {
            "packages_with_sidecar": int(len(bench)), "package_passed": int(bench.benchmark_passed.sum()),
            "tests_total": int(bench.benchmark_tests.sum()), "tests_failed": int(bench.benchmark_failed.sum()),
            "tests_skipped": int(bench.benchmark_skipped.sum()),
            "top_failing_tests": dict(failing_tests.most_common(8)),
            "grasp_and_lift": dict(grasp), "grasp_and_lift_packages": len(grasp_pkgs), "grasp_and_lift_skip_reasons": dict(grasp_reasons),
        },
        "grippers": [{**g, "runtime_statuses": dict(g["runtime_statuses"])} for g in gripper],
    }
    (a.results / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    print(json.dumps({k: v for k, v in s.items() if k not in ("findings_assets", "per_vendor")}, indent=1, default=str))


if __name__ == "__main__":
    main()
