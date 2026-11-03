import numpy as np
import pytest
from pxr import Sdf, Usd

from vr.catalog import Browser, Card, layout_arc, ray_pick
from vr.grab import BodyState, Grab, GrabParams
from vr.scene import card_panel_usda, environment_usda, safe_name, spawn_usda
from vr.voice import parse
from vr.xrmath import (matrix_to_pose, pose_to_matrix, quat_mul, ray_from_matrix, rigid_inverse,
                       rotation_vector)


# ----------------------------------------------------------------- xrmath
def test_pose_roundtrip_and_inverse():
    q = np.array([0.1, -0.3, 0.2, 0.9]); q /= np.linalg.norm(q)
    M = pose_to_matrix([1, 2, 3], q)
    p, q2 = matrix_to_pose(M)
    assert np.allclose(p, [1, 2, 3])
    assert np.allclose(q2, q) or np.allclose(q2, -q)
    assert np.allclose(M @ rigid_inverse(M), np.eye(4), atol=1e-9)


def test_row_vector_convention_matches_quaternion():
    # 90 deg about Z: +X -> +Y
    q = np.array([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    M = pose_to_matrix([0, 0, 0], q)
    assert np.allclose(np.array([1, 0, 0]) @ M[:3, :3], [0, 1, 0], atol=1e-9)


def test_rotation_vector_small_and_shortest():
    q = np.array([0, 0, np.sin(0.05), np.cos(0.05)])
    assert np.allclose(rotation_vector(q), [0, 0, 0.1], atol=1e-6)
    assert np.allclose(rotation_vector(-q), [0, 0, 0.1], atol=1e-6)


def test_ray_from_matrix_points_down_minus_z():
    o, d = ray_from_matrix(np.eye(4))
    assert np.allclose(d, [0, 0, -1])


# ----------------------------------------------------------------- grab
def _body(p, kind="rigid", mass=1.0):
    return BodyState("/b", kind, np.array(p, float), np.array([0, 0, 0, 1.0]), mass=mass)


def test_grab_requires_reach():
    g = Grab(GrabParams(reach=0.1))
    assert g.begin(pose_to_matrix([0, 0, 0], [0, 0, 0, 1]), [_body([0.5, 0, 0])]) is None
    assert g.begin(pose_to_matrix([0.45, 0, 0], [0, 0, 0, 1]), [_body([0.5, 0, 0])]) == "/b"


def test_grab_keeps_offset_and_tracks_hand():
    g = Grab()
    hand0 = pose_to_matrix([0.5, 0, 0.9], [0, 0, 0, 1])
    b = _body([0.55, 0.0, 0.85])
    g.begin(hand0, [b])
    hand1 = pose_to_matrix([0.5, 0, 1.2], [0, 0, 0, 1])   # lift 30 cm
    tp, _ = g.target(hand1)
    assert np.allclose(tp, [0.55, 0.0, 1.15])            # offset preserved, no snap
    c = g.update(0.0, 1 / 90, hand1, b)
    assert c.kind == "rigid" and c.linear_velocity[2] > 0
    assert np.linalg.norm(c.linear_velocity) <= g.p.max_lin_speed + 1e-9


def test_rigid_velocity_tracking_converges_in_integrator():
    g, dt = Grab(), 1 / 90
    b = _body([0.5, 0, 0.8])
    g.begin(pose_to_matrix([0.5, 0, 0.8], [0, 0, 0, 1]), [b])
    target_hand = pose_to_matrix([0.7, 0.1, 1.0], [0, 0, 0, 1])
    for i in range(30):
        c = g.update(i * dt, dt, target_hand, b)
        b.position = b.position + c.linear_velocity * dt
    assert np.allclose(b.position, [0.7, 0.1, 1.0], atol=1e-3)


def test_link_spring_is_stable_and_converges():
    """Spring on a 1-DOF prismatic link (drawer along +X): pulled only along the axis it can move."""
    g, dt = Grab(), 1 / 90
    b = _body([0.0, 0, 0], kind="link", mass=2.0)
    g.begin(pose_to_matrix([0, 0, 0], [0, 0, 0, 1]), [b])
    hand = pose_to_matrix([0.3, 0.2, 0], [0, 0, 0, 1])       # pull diagonally
    x, v = 0.0, 0.0
    for i in range(400):
        b.position, b.linear_velocity = np.array([x, 0, 0]), np.array([v, 0, 0])
        c = g.update(i * dt, dt, hand, b)
        a = c.force[0] / b.mass                                # joint removes y component
        v += a * dt
        x = min(max(x + v * dt, 0.0), 0.4)                     # drawer travel limit
    assert abs(x - 0.3) < 5e-3 and abs(v) < 1e-2


def test_release_throws_with_hand_velocity():
    g, dt = Grab(), 1 / 90
    b = _body([0, 0, 1])
    g.begin(pose_to_matrix([0, 0, 1], [0, 0, 0, 1]), [b])
    for i in range(6):
        g.update(i * dt, dt, pose_to_matrix([i * 0.02, 0, 1], [0, 0, 0, 1]), b)
    c = g.end()
    assert c is not None and np.isclose(c.linear_velocity[0], 0.02 / dt, rtol=1e-6)
    assert not g.holding


# ----------------------------------------------------------------- voice
@pytest.mark.parametrize("text,action,query,index", [
    ("Find a red forklift.", "search", "red forklift", None),
    ("search for robot gripper", "search", "robot gripper", None),
    ("procura uma empilhadeira", "search", "empilhadeira", None),
    ("spawn", "spawn", None, None),
    ("Spawn three", "spawn", None, 3),
    ("spawn number 2", "spawn", None, 2),
    ("coloca o dois", "spawn", None, 2),
    ("next", "next", None, None),
    ("anterior", "previous", None, None),
    ("limpa tudo", "clear", None, None),
    ("spawn banana", "unknown", None, None),
    ("", "unknown", None, None),
])
def test_voice_parse(text, action, query, index):
    i = parse(text)
    assert (i.action, i.query, i.index) == (action, query, index)


# ----------------------------------------------------------------- catalog
def test_browser_paging_and_resolve():
    b = Browser(page_size=3)
    b.set_results("x", list("abcdefg"))
    assert b.page_count == 3 and b.visible() == ["a", "b", "c"]
    b.next(); b.next(); b.next()
    assert b.visible() == ["g"] and b.resolve(1) == "g" and b.resolve(2) is None
    b.previous()
    assert b.resolve(3) == "f"


def test_layout_and_ray_pick():
    head = np.array([0, 0, 1.6])
    frames = layout_arc(5, head, np.array([1.0, 0, -0.5]), np.array([0, 0, 1.0]))
    cards = [Card(i + 1, f"c{i}", "u", None, **fr) for i, fr in enumerate(frames)]
    # all cards at the same distance and height, facing the user
    for c in cards:
        assert np.isclose(np.linalg.norm((c.center - head)[:2]), 1.1)
        assert np.dot(c.normal, head - c.center) > 0
    mid = cards[2]
    hit = ray_pick(head, (mid.center - head) / np.linalg.norm(mid.center - head), cards)
    assert hit is mid
    assert ray_pick(head, np.array([-1.0, 0, 0]), cards) is None   # behind the user


# ----------------------------------------------------------------- USDA composition
def _compose(text: str) -> Usd.Stage:
    layer = Sdf.Layer.CreateAnonymous(".usda")
    assert layer.ImportFromString(text)
    return Usd.Stage.Open(layer)


def test_environment_composes_with_colliders():
    st = _compose(environment_usda())
    from pxr import UsdPhysics
    cols = [p for p in st.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert {p.GetName() for p in cols} == {"Floor", "Table"}
    assert st.GetPrimAtPath("/World/Render/Camera").IsValid()


def test_spawn_usda_selects_runtime_variant(tmp_path):
    asset = tmp_path / "a.usda"
    asset.write_text('''#usda 1.0
( defaultPrim = "A" )
def Xform "A" ( variantSets = ["PhysX", "Newton"] )
{
    variantSet "PhysX" = {
        "Disabled" {
        }
        "Enabled" {
            def Cube "Body" (
                prepend apiSchemas = ["PhysicsRigidBodyAPI"]
            )
            {
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
''')
    st = _compose(spawn_usda("box_00", str(asset), pose_to_matrix([1, 2, 3], [0, 0, 0, 1])))
    p = st.GetPrimAtPath("/Spawned/box_00")
    assert p.GetVariantSets().GetVariantSet("PhysX").GetVariantSelection() == "Enabled"
    assert st.GetPrimAtPath("/Spawned/box_00/Body").IsValid()
    from pxr import UsdGeom
    t = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    assert tuple(t) == (1, 2, 3)


def test_card_panel_composes():
    frames = layout_arc(3, np.array([0, 0, 1.6]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))
    cards = [Card(i + 1, f"asset {i}", "https://x/a.usd", "/tmp/t.jpg", badge=b, **fr)
             for i, (fr, b) in enumerate(zip(frames, ["physics", "articulated", ""]))]
    st = _compose(card_panel_usda(cards))
    assert len([p for p in st.Traverse() if p.GetName().startswith("Card_")]) == 3
    assert st.GetPrimAtPath("/Catalog/Panel/Looks/Mat_02/Tex").IsValid()


def test_safe_name():
    assert safe_name("https://x/y/franka_hand.usda", 3) == "franka_hand_03"
    assert safe_name("3d box-v2.usd") == "_3d_box_v2"
