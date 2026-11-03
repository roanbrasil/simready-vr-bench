"""The audit on tiny synthetic packages that reproduce the patterns found in SimReady Central."""
import json
from pathlib import Path

from srvr.audit import audit_asset, normalize_feature_id
from srvr.index import IndexEntry, dedupe, parse_line, stratified_sample

HEADER = '#usda 1.0\n(\n    defaultPrim = "A"\n    metersPerUnit = 1\n    kilogramsPerUnit = 1\n    upAxis = "Z"\n{meta}\n)\n'


def _pkg(tmp: Path, name: str, root_body: str, extra_files: dict | None = None,
         sidecar_features: dict | None = None, meta: str = "") -> Path:
    d = tmp / name
    (d / "simready_usd").mkdir(parents=True)
    (d / ".metadata").mkdir()
    (d / "com.nvidia.simready.packaging.json").write_text(json.dumps({"license": "cc-by-4.0"}))
    (d / "simready_usd" / f"{name}.usda").write_text(HEADER.format(meta=meta) + root_body)
    for rel, txt in (extra_files or {}).items():
        p = d / "simready_usd" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt)
    if sidecar_features is not None:
        (d / ".metadata" / f"{name}_validation.json").write_text(json.dumps({
            f"simready_usd/{name}.usda": {"profile_id": "Robotics-Prop",
                                          "features_summary": {k: {"passed": v} for k, v in sidecar_features.items()}}}))
    return d


GOOD_BOX = '''
def Xform "A" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    float physics:mass = 2.0
    def Cube "Geo" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 0.2
    }
}
'''


def test_good_box_is_grabbable_with_no_findings(tmp_path):
    d = _pkg(tmp_path, "box", GOOD_BOX, sidecar_features={"FET_003_PHYSX": True},
             meta='    customLayerData = {\n        dictionary SimReady_Metadata = {\n            double mass = 2.0\n        }\n    }')
    a = audit_asset("k", "oem", d)
    assert a.error is None and a.findings == []
    px = a.observed["PhysX"]
    assert (px.rigid_bodies, px.bodies_with_collider, px.bodies_with_explicit_mass) == (1, 1, 1)
    assert px.grabbable and a.declared.declared_source == "sidecar"


DEACTIVATED_INSTANCE_COLLIDER = '''
def Xform "A" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    float physics:mass = 30
    def Xform "visual" (
        instanceable = true
        prepend references = @./instances.usda@</Instances/visual>
    )
    {
    }
    def Xform "collision" (
        active = false
        instanceable = true
        prepend references = @./instances.usda@</Instances/collision>
    )
    {
    }
}
'''
INSTANCES = '''#usda 1.0
def Xform "Instances"
{
    def Xform "visual"
    {
        def Mesh "m"
        {
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
            point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        }
    }
    def Xform "collision"
    {
        def Mesh "c" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {
            uniform token purpose = "guide"
        }
    }
}
'''


def test_deactivated_collider_is_caught_even_when_declared_rigid(tmp_path):
    """The Comau S18-17 pattern: collision prims authored, then shipped with active = false."""
    d = _pkg(tmp_path, "arm", DEACTIVATED_INSTANCE_COLLIDER, {"instances.usda": INSTANCES},
             sidecar_features={"FET_003_PHYSX": True})
    a = audit_asset("k", "oem", d)
    px = a.observed["PhysX"]
    assert px.rigid_bodies == 1 and px.colliders == 0 and px.bodies_visible_no_collider == 1
    assert "VISIBLE_BODY_WITHOUT_COLLIDER" in a.findings
    assert "DECLARED_RIGID_BUT_NO_COLLIDER" in a.findings
    assert not px.grabbable


def test_instanced_collider_is_counted_when_active(tmp_path):
    body = DEACTIVATED_INSTANCE_COLLIDER.replace("        active = false\n", "")
    assert "active = false" not in body
    d = _pkg(tmp_path, "arm2", body, {"instances.usda": INSTANCES}, sidecar_features={})
    px = audit_asset("k", "oem", d).observed["PhysX"]
    assert px.colliders == 1 and px.bodies_with_collider == 1 and px.bodies_visible_no_collider == 0


def test_density_only_body_does_not_trigger_mass_mismatch(tmp_path):
    body = '''
def Xform "A"
{
    def Xform "b1" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 1
        def Cube "g" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
        }
    }
    def Xform "b2" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:density = 500
        def Cube "g" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
        }
    }
}
'''
    d = _pkg(tmp_path, "two", body, sidecar_features={},
             meta='    customLayerData = {\n        dictionary SimReady_Metadata = {\n            double mass = 9.0\n        }\n    }')
    a = audit_asset("k", "oem", d)
    px = a.observed["PhysX"]
    assert px.bodies_with_mass_source == 2 and px.bodies_with_explicit_mass == 1
    assert "METADATA_MASS_MISMATCH" not in a.findings


def test_all_explicit_mass_mismatch_is_flagged(tmp_path):
    d = _pkg(tmp_path, "box2", GOOD_BOX, sidecar_features={},
             meta='    customLayerData = {\n        dictionary SimReady_Metadata = {\n            double mass = 5.0\n        }\n    }')
    assert "METADATA_MASS_MISMATCH" in audit_asset("k", "oem", d).findings


def test_in_usd_claims_dialect_is_normalized(tmp_path):
    meta = '''    customLayerData = {
        dictionary SimReady_Metadata = {
            dictionary validation = {
                string profile = "Robotics-Prop"
                dictionary validated_features = {
                    dictionary "2026-01-13" = {
                        dictionary FET003_BASE_PHYSX = {
                            string passed = "False"
                        }
                    }
                    dictionary "2026-01-16" = {
                        dictionary FET003_BASE_NEUTRAL = {
                            string passed = "True"
                        }
                        dictionary FET003_BASE_PHYSX = {
                            string passed = "True"
                        }
                    }
                }
            }
        }
    }'''
    d = _pkg(tmp_path, "legacy", GOOD_BOX, meta=meta)
    a = audit_asset("k", "oem", d)
    assert a.declared.declared_source == "in_usd" and a.declared.profile == "Robotics-Prop"
    assert a.declared.feature("FET_003_PHYSX") is True       # latest run wins
    assert a.declared.feature("FET_003_STANDARD") is True
    assert "CLAIMS_IN_USD_ONLY" in a.findings
    assert normalize_feature_id("FET004_BASE_NEUTRAL") == "FET_004_STANDARD"
    assert normalize_feature_id("FET_022_PHYSX") == "FET_022_PHYSX"


def test_per_runtime_variant_selection(tmp_path):
    body = '''
def Xform "A" (
    variants = {
        string Newton = "Disabled"
        string PhysX = "Disabled"
    }
    prepend variantSets = ["PhysX", "Newton"]
)
{
    def Cube "g"
    {
    }
    variantSet "PhysX" = {
        "Disabled" {
        }
        "Enabled" {
            over "g" (
                prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"]
            )
            {
                float physics:mass = 1
            }
        }
    }
    variantSet "Newton" = {
        "Disabled" {
        }
        "Enabled" {
        }
    }
}
'''
    d = _pkg(tmp_path, "var", body, sidecar_features={})
    a = audit_asset("k", "oem", d)
    assert a.observed["PhysX"].variant_scheme == "per_runtime_sets"
    assert a.observed["PhysX"].rigid_bodies == 1 and a.observed["Newton"].rigid_bodies == 0
    assert "RUNTIME_BODY_COUNT_DIVERGES" in a.findings


def test_index_parse_dedupe_sample():
    u = "https://huggingface.co/datasets/Comau/S18-17/resolve/{}/S18-17/com.nvidia.simready.packaging.json"
    a, b = parse_line(u.format("aaaaaaa1")), parse_line(u.format("bbbbbbb2"))
    assert a.repo == "Comau/S18-17" and a.asset == "S18-17" and a.oem == "Comau"
    assert dedupe([a, b]) == [b]
    many = [IndexEntry("big/x", "r", f"a{i}", "big") for i in range(100)] + [IndexEntry("s/y", "r", "z", "s")]
    s = stratified_sample(many, 10)
    assert len(s) == 11 and stratified_sample(many, 10) == s   # seeded


def test_match_runtime_variant_handles_vendor_naming():
    from srvr.audit import match_runtime_variant
    assert match_runtime_variant(["mujoco", "none", "physics", "physx"], "PhysX") == "physx"
    assert match_runtime_variant(["None", "PhysX"], "PhysX") == "PhysX"
    assert match_runtime_variant(["Newton_compliant", "Physx_parallel_grip"], "PhysX") == "Physx_parallel_grip"
    assert match_runtime_variant(["Newton_compliant", "Physx_parallel_grip"], "Newton") == "Newton_compliant"
    assert match_runtime_variant(["None", "PhysX"], "MuJoCo") is None


def test_unreferenced_collision_prototype_is_not_counted(tmp_path):
    """The Comau 248209f pattern: collision prototypes exist in instances.usda but nothing references them."""
    body = DEACTIVATED_INSTANCE_COLLIDER.replace('''    def Xform "collision" (
        active = false
        instanceable = true
        prepend references = @./instances.usda@</Instances/collision>
    )
    {
    }
''', "")
    assert "collision" not in body
    d = _pkg(tmp_path, "arm3", body, {"instances.usda": INSTANCES}, sidecar_features={"FET_003_PHYSX": True})
    a = audit_asset("k", "oem", d)
    assert a.observed["PhysX"].colliders == 0
    assert "DECLARED_RIGID_BUT_NO_COLLIDER" in a.findings
