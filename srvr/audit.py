"""Contract audit: what a SimReady package *declares* vs what its USD *carries*.

Declared  = the package's own sidecars (``.metadata/<asset>_validation.json``,
            ``<asset>_benchmark.json``) and the root layer's ``SimReady_Metadata``.
Observed  = an independent read of the composed stage with ``usd-core``, once per
            physics runtime variant (PhysX / Newton / MuJoCo), traversing instance
            proxies so instanced colliders are counted.

Nothing here runs a physics engine. The question is narrower and, for a VR grab demo,
more practical: does the composed stage contain what a solver needs to simulate the
object without the application guessing (bodies, colliders, a mass source, joint limits)?
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

RUNTIMES = ("PhysX", "Newton", "MuJoCo")
LIMITED_JOINTS = ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint")


# --------------------------------------------------------------------------- declared
@dataclass
class Declared:
    has_validation_sidecar: bool = False
    declared_source: str = "none"     # "sidecar" | "in_usd" | "none"
    has_benchmark_sidecar: bool = False
    profile: str | None = None
    features: dict[str, bool] = field(default_factory=dict)   # FET id -> passed
    benchmark_passed: bool | None = None
    benchmark_tests: int = 0
    benchmark_failed: int = 0
    benchmark_skipped: int = 0
    benchmark_tests_by_name: list[str] = field(default_factory=list)
    metadata_mass: float | None = None                       # SimReady_Metadata["mass"]
    metadata_category: str | None = None
    license: str | None = None

    def feature(self, fid: str) -> bool | None:
        return self.features.get(fid)


def read_declared(asset_dir: Path, root_usd: Path | None) -> Declared:
    d = Declared()
    asset = asset_dir.name
    meta = asset_dir / ".metadata"
    v = meta / f"{asset}_validation.json"
    if v.exists():
        d.has_validation_sidecar = True
        raw = json.loads(v.read_text(encoding="utf-8"))
        # one key per validated root USD; take the first (packages have one root)
        block = next(iter(raw.values())) if raw else {}
        d.profile = block.get("profile_id")
        for fid, info in (block.get("features_summary") or {}).items():
            d.features[fid] = bool(info.get("passed"))
        d.declared_source = "sidecar"
    b = meta / f"{asset}_benchmark.json"
    if b.exists():
        d.has_benchmark_sidecar = True
        raw = json.loads(b.read_text(encoding="utf-8"))
        d.benchmark_passed = raw.get("passed")
        res = raw.get("results") or []
        d.benchmark_tests = len(res)
        # "skipped" tests carry passed=false but are not-applicable, not failures
        d.benchmark_failed = sum(1 for r in res if r.get("status") == "fail")
        d.benchmark_skipped = sum(1 for r in res if r.get("status") == "skipped")
        d.benchmark_tests_by_name = sorted({r.get("test_name") for r in res if r.get("status") == "pass"})
    pkg = asset_dir / "com.nvidia.simready.packaging.json"
    if pkg.exists():
        d.license = json.loads(pkg.read_text(encoding="utf-8")).get("license")
    if root_usd is not None:
        layer = Sdf.Layer.FindOrOpen(str(root_usd))
        md = dict(layer.customLayerData.get("SimReady_Metadata", {})) if layer else {}
        mass = md.get("mass")
        try:
            d.metadata_mass = float(mass) if mass is not None else None
        except (TypeError, ValueError):
            d.metadata_mass = None
        d.metadata_category = md.get("category")
        val = md.get("validation")
        if val is not None:
            val = _plain(val)
            if d.profile is None:
                d.profile = val.get("profile")
            if d.declared_source == "none":
                feats = _latest_in_usd_features(val)
                if feats:
                    d.features.update(feats)
                    d.declared_source = "in_usd"
    return d


def _plain(o):
    """Vt/Dictionary -> plain python, recursively."""
    try:
        return {str(k): _plain(v) for k, v in dict(o).items()}
    except (TypeError, ValueError):
        return o


_LEGACY = re.compile(r"^FET(\d{3})_BASE_(NEUTRAL|PHYSX|NEWTON|MUJOCO|ISAAC|RTX|MDL|OPENPBR)$")


def normalize_feature_id(fid: str) -> str:
    """In-USD dialect ``FET003_BASE_PHYSX`` -> sidecar dialect ``FET_003_PHYSX`` (NEUTRAL -> STANDARD)."""
    m = _LEGACY.match(fid)
    if not m:
        return fid
    num, rt = m.groups()
    return f"FET_{num}_{'STANDARD' if rt == 'NEUTRAL' else rt}"


def _latest_in_usd_features(val: dict) -> dict[str, bool]:
    runs = val.get("validated_features") or {}
    if not isinstance(runs, dict) or not runs:
        return {}
    latest = runs[sorted(runs)[-1]]
    out = {}
    for fid, info in (latest or {}).items():
        passed = info.get("passed") if isinstance(info, dict) else None
        out[normalize_feature_id(fid)] = str(passed).lower() == "true"
    return out


# --------------------------------------------------------------------------- observed
@dataclass
class Observed:
    runtime: str                      # variant selected ("default" if no variant set)
    variant_available: bool
    variant_scheme: str = "none"      # "per_runtime_sets" | "physics_set" | "none"
    rigid_bodies: int = 0
    kinematic_bodies: int = 0
    bodies_with_collider: int = 0
    bodies_visible_no_collider: int = 0   # has render geometry but nothing to collide with
    bodies_with_explicit_mass: int = 0
    bodies_with_mass_source: int = 0  # mass>0 or density>0 on body/colliders/bound material
    authored_mass_total: float = 0.0  # kg, sum of explicit body masses (stage units applied)
    colliders: int = 0
    joints: int = 0
    limited_joint_candidates: int = 0  # revolute + prismatic
    limited_joints_with_limits: int = 0
    articulation_roots: int = 0
    semantic_prims: int = 0

    @property
    def dynamic_bodies(self) -> int:
        return self.rigid_bodies - self.kinematic_bodies

    @property
    def grabbable(self) -> bool:
        """At least one dynamic body a VR hand could pick up without the app inventing physics."""
        return self.dynamic_bodies > 0 and self.bodies_with_collider > 0 and self.bodies_with_mass_source > 0


def _positive(attr) -> float | None:
    if attr and attr.HasAuthoredValue():
        v = attr.Get()
        if v is not None and math.isfinite(v) and v > 0:
            return float(v)
    return None


def _bound_physics_density(prim: Usd.Prim, stage: Usd.Stage) -> float | None:
    rel = prim.GetRelationship("material:binding:physics") or prim.GetRelationship("material:binding")
    if not rel:
        return None
    for t in rel.GetTargets():
        m = stage.GetPrimAtPath(t)
        if m and m.HasAPI(UsdPhysics.MaterialAPI):
            return _positive(UsdPhysics.MaterialAPI(m).GetDensityAttr())
    return None


def _is_semantic(prim: Usd.Prim) -> bool:
    return any(s.startswith(("SemanticsLabelsAPI", "SemanticsAPI")) for s in prim.GetAppliedSchemas())


def _limits_authored(prim: Usd.Prim) -> bool:
    lo, hi = prim.GetAttribute("physics:lowerLimit"), prim.GetAttribute("physics:upperLimit")
    if not (lo and hi and lo.HasAuthoredValue() and hi.HasAuthoredValue()):
        return False
    a, b = lo.Get(), hi.Get()
    return a is not None and b is not None and math.isfinite(a) and math.isfinite(b) and a < b


def match_runtime_variant(names, runtime: str) -> str | None:
    """Pick the variant of a single ``Physics`` set that means ``runtime``: exact (any case), else prefix.

    Seen in the wild: ``physx``, ``PhysX``, ``Physx_parallel_grip``, ``Newton_compliant``. With several
    prefix matches (e.g. two PhysX grip modes) the first in authored order wins.
    """
    rt = runtime.lower()
    for n in names:
        if n.lower() == rt:
            return n
    for n in names:
        if n.lower().startswith(rt):
            return n
    return None


def observe(root_usd: Path, runtime: str | None) -> Observed:
    stage = Usd.Stage.Open(str(root_usd), Usd.Stage.LoadAll)
    available, scheme = False, "none"
    dp = stage.GetDefaultPrim() or next(iter(stage.GetPseudoRoot().GetChildren()), None)
    if dp is not None and "Physics" in dp.GetVariantSets().GetNames():
        scheme = "physics_set"   # one set; names vary: physx / PhysX / Physx_parallel_grip / Newton_compliant...
        vs = dp.GetVariantSets().GetVariantSet("Physics")
        pick = match_runtime_variant(vs.GetVariantNames(), runtime) if runtime else None
        if pick:
            vs.SetVariantSelection(pick)
            available = True
    elif dp is not None and any(rt in dp.GetVariantSets().GetNames() for rt in RUNTIMES):
        scheme = "per_runtime_sets"
    if runtime and dp and scheme == "per_runtime_sets":
        vsets = dp.GetVariantSets()
        if runtime in vsets.GetNames():
            vs = vsets.GetVariantSet(runtime)
            if "Enabled" in vs.GetVariantNames():
                vs.SetVariantSelection("Enabled")
                available = True
            # keep the other runtimes off so isolation is what the package intends
            for other in RUNTIMES:
                if other != runtime and other in vsets.GetNames():
                    ovs = vsets.GetVariantSet(other)
                    if "Disabled" in ovs.GetVariantNames():
                        ovs.SetVariantSelection("Disabled")
    o = Observed(runtime=runtime or "default", variant_available=available, variant_scheme=scheme)
    kpu = UsdPhysics.GetStageKilogramsPerUnit(stage) or 1.0

    prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)))
    bodies = []
    for p in prims:
        if not p.IsActive():
            continue
        if p.HasAPI(UsdPhysics.CollisionAPI):
            en = UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
            if en is not False:
                o.colliders += 1
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(p)
            if rb.GetRigidBodyEnabledAttr().Get() is not False:
                bodies.append(p)
                if rb.GetKinematicEnabledAttr().Get():
                    o.kinematic_bodies += 1
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            o.articulation_roots += 1
        if p.IsA(UsdPhysics.Joint):
            o.joints += 1
            if p.GetTypeName() in LIMITED_JOINTS:
                o.limited_joint_candidates += 1
                if _limits_authored(p):
                    o.limited_joints_with_limits += 1
        if _is_semantic(p):
            o.semantic_prims += 1

    o.rigid_bodies = len(bodies)
    body_paths = {b.GetPath() for b in bodies}
    for b in bodies:
        has_col, mass_src, has_geom = False, False, False
        explicit = None
        if b.HasAPI(UsdPhysics.MassAPI):
            m = UsdPhysics.MassAPI(b)
            explicit = _positive(m.GetMassAttr())
            mass_src = bool(explicit or _positive(m.GetDensityAttr()))
        if not mass_src and _bound_physics_density(b, stage):
            mass_src = True
        # subtree of this body, stopping at nested bodies (they own their colliders)
        stack = [b]
        while stack:
            q = stack.pop()
            if q.IsActive() and q.IsA(UsdGeom.Gprim) and not q.HasAPI(UsdPhysics.CollisionAPI):
                purpose = UsdGeom.Imageable(q).GetPurposeAttr().Get()
                if purpose in (None, "default", "render"):
                    has_geom = True
            if q.IsActive() and q.HasAPI(UsdPhysics.CollisionAPI) and \
                    UsdPhysics.CollisionAPI(q).GetCollisionEnabledAttr().Get() is not False:
                has_col = True
                if not mass_src:
                    if q.HasAPI(UsdPhysics.MassAPI) and (
                        _positive(UsdPhysics.MassAPI(q).GetMassAttr()) or _positive(UsdPhysics.MassAPI(q).GetDensityAttr())
                    ):
                        mass_src = True
                    elif _bound_physics_density(q, stage):
                        mass_src = True
            for c in q.GetFilteredChildren(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
                if c.GetPath() not in body_paths:
                    stack.append(c)
        o.bodies_with_collider += has_col
        o.bodies_visible_no_collider += (has_geom and not has_col)
        o.bodies_with_mass_source += mass_src
        if explicit:
            o.bodies_with_explicit_mass += 1
            o.authored_mass_total += explicit * kpu
    return o


# --------------------------------------------------------------------------- findings
@dataclass
class AssetAudit:
    key: str
    oem: str
    asset: str
    root_usd: str | None
    declared: Declared
    observed: dict[str, Observed]
    findings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_row(self) -> dict:
        d = self.declared
        row = {
            "key": self.key, "oem": self.oem, "asset": self.asset, "error": self.error,
            "profile": d.profile, "license": d.license, "category": d.metadata_category,
            "has_validation_sidecar": d.has_validation_sidecar, "has_benchmark_sidecar": d.has_benchmark_sidecar,
            "benchmark_passed": d.benchmark_passed, "benchmark_tests": d.benchmark_tests,
            "benchmark_failed": d.benchmark_failed, "benchmark_skipped": d.benchmark_skipped,
            "benchmark_grasp_passed": any("grasp" in t for t in d.benchmark_tests_by_name),
            "metadata_mass": d.metadata_mass,
            "declared_source": d.declared_source,
            "declared_FET_003_PHYSX": d.feature("FET_003_PHYSX"),
            "declared_FET_004_PHYSX": d.feature("FET_004_PHYSX"),
            "declared_FET_011": d.feature("FET_011_STANDARD"),
            "findings": ";".join(self.findings),
        }
        for rt, o in self.observed.items():
            for k, v in asdict(o).items():
                if k != "runtime":
                    row[f"{rt}.{k}"] = v
            row[f"{rt}.grabbable"] = o.grabbable
        return row


def find_root_usd(asset_dir: Path) -> Path | None:
    roots = asset_dir / ".metadata" / "com.nvidia.simready.root_usds.json"
    if roots.exists():
        entries = json.loads(roots.read_text(encoding="utf-8")).get("entries") or []
        for e in entries:
            p = asset_dir / e
            if p.exists():
                return p
    named = sorted(p for p in asset_dir.rglob(f"{asset_dir.name}.usd*") if ".cache" not in p.parts)
    if named:
        return min(named, key=lambda p: len(p.parts))
    for base in (asset_dir / "simready_usd", asset_dir):
        if base.is_dir():
            cands = sorted(x for x in base.iterdir() if x.suffix in (".usd", ".usda", ".usdc"))
            named = [c for c in cands if c.stem == asset_dir.name]
            if named or cands:
                return (named or cands)[0]
    return None


def derive_findings(d: Declared, obs: dict[str, Observed]) -> list[str]:
    f: list[str] = []
    if d.declared_source == "in_usd":
        f.append("CLAIMS_IN_USD_ONLY")
    elif d.declared_source == "none":
        f.append("NO_DECLARED_CLAIMS")
    px = obs.get("PhysX")
    if px is not None:
        if d.feature("FET_003_PHYSX") and px.rigid_bodies == 0:
            f.append("DECLARED_RIGID_BUT_NO_BODY_PHYSX")
        if px.bodies_visible_no_collider:
            f.append("VISIBLE_BODY_WITHOUT_COLLIDER")
        if d.feature("FET_003_PHYSX") and px.rigid_bodies and px.colliders == 0:
            f.append("DECLARED_RIGID_BUT_NO_COLLIDER")
        if px.rigid_bodies and px.bodies_with_mass_source < px.rigid_bodies:
            f.append("BODY_WITHOUT_MASS_SOURCE")
        if px.limited_joint_candidates and px.limited_joints_with_limits < px.limited_joint_candidates:
            f.append("JOINT_WITHOUT_LIMITS")
        if d.metadata_mass and px.rigid_bodies and px.bodies_with_explicit_mass == px.rigid_bodies:
            rel = abs(px.authored_mass_total - d.metadata_mass) / d.metadata_mass
            if rel > 0.10:
                f.append("METADATA_MASS_MISMATCH")
        if d.feature("FET_011_STANDARD") and px.semantic_prims == 0:
            f.append("DECLARED_SEMANTICS_BUT_NONE")
    # runtime parity: body counts should not differ between solvers that the package enables
    counts = {rt: o.rigid_bodies for rt, o in obs.items() if o.variant_available}
    if len(set(counts.values())) > 1:
        f.append("RUNTIME_BODY_COUNT_DIVERGES")
    return f


def audit_asset(key: str, oem: str, asset_dir: Path) -> AssetAudit:
    root = find_root_usd(asset_dir)
    try:
        declared = read_declared(asset_dir, root)
    except Exception as e:
        return AssetAudit(key=key, oem=oem, asset=asset_dir.name, root_usd=str(root) if root else None,
                          declared=Declared(), observed={}, error=f"DECLARED: {type(e).__name__}: {e}"[:300])
    a = AssetAudit(key=key, oem=oem, asset=asset_dir.name, root_usd=str(root) if root else None,
                   declared=declared, observed={})
    if root is None:
        a.error = "NO_ROOT_USD"
        return a
    try:
        for rt in RUNTIMES:
            a.observed[rt] = observe(root, rt)
    except Exception as e:  # malformed layer, missing payload, etc.
        a.error = f"{type(e).__name__}: {e}"[:300]
    a.findings = derive_findings(declared, a.observed)
    return a
