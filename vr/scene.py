"""USDA builders for everything the app authors at runtime.

All strings go through ``ovstage.population.add_usd_reference_from_string`` (or are
written to disk for the environment), so the app never needs a USD authoring library
at runtime. Tests parse them with usd-core to make sure they compose.
"""
from __future__ import annotations

import re

import numpy as np

RUNTIMES = ("PhysX", "Newton", "MuJoCo")


def safe_name(s: str, n: int | None = None) -> str:
    base = re.sub(r"[^A-Za-z0-9_]", "_", s.rsplit("/", 1)[-1].split(".")[0]) or "asset"
    if base[0].isdigit():
        base = "_" + base
    return f"{base}_{n:02d}" if n is not None else base


def _m4(M) -> str:
    rows = np.asarray(M, dtype=float).reshape(4, 4)
    return "(" + ", ".join("(" + ", ".join(f"{v:.6g}" for v in r) + ")" for r in rows) + ")"


def spawn_usda(prim_name: str, asset_url: str, world_M, runtime: str = "PhysX") -> str:
    """A prim that references a SimReady asset, selects one physics runtime, and places it.

    SimReady packages carry ``PhysX`` / ``Newton`` / ``MuJoCo`` variant sets on their
    default prim and ship them ``Disabled`` unless told otherwise; the variant selection
    on the referencing prim is what turns physics on for the solver we run.
    """
    sel = "\n".join(f'            string {rt} = "{"Enabled" if rt == runtime else "Disabled"}"' for rt in RUNTIMES)
    return f'''#usda 1.0
(
    defaultPrim = "Spawned"
    upAxis = "Z"
    metersPerUnit = 1
)
def Xform "Spawned"
{{
    def Xform "{prim_name}" (
        prepend references = @{asset_url}@
        variants = {{
{sel}
        }}
    )
    {{
        matrix4d xformOp:transform = {_m4(world_M)}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}
}}
'''


def environment_usda(table_height: float = 0.75) -> str:
    """Fallback room when no reconstructed environment is supplied: floor + table, static colliders.

    Replace with your own scene via ``--env`` (I used a Blender reconstruction of the room).
    """
    th = table_height
    return f'''#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
    metersPerUnit = 1
    kilogramsPerUnit = 1
)
def Xform "World"
{{
    def PhysicsScene "PhysicsScene"
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }}
    def DistantLight "Sun"
    {{
        float inputs:intensity = 3000
        float3 xformOp:rotateXYZ = (45, 0, 30)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }}
    def DomeLight "Sky"
    {{
        float inputs:intensity = 600
    }}
    def Cube "Floor" (prepend apiSchemas = ["PhysicsCollisionAPI"])
    {{
        double size = 1
        float3 xformOp:scale = (10, 10, 0.02)
        double3 xformOp:translate = (0, 0, -0.01)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [(0.55, 0.55, 0.52)]
    }}
    def Cube "Table" (prepend apiSchemas = ["PhysicsCollisionAPI"])
    {{
        double size = 1
        float3 xformOp:scale = (1.6, 0.8, 0.04)
        double3 xformOp:translate = (0.9, 0, {th - 0.02})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [(0.45, 0.32, 0.22)]
    }}
    def Xform "Render"
    {{
        def Camera "Camera"
        {{
            float3 xformOp:rotateXYZ = (90, 0, -90)
            double3 xformOp:translate = (-0.4, 0, 1.6)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
        }}
    }}
    def Scope "Spawned" {{}}
    def Scope "Catalog" {{}}
}}
'''


def card_panel_usda(cards: list, panel_name: str = "Panel") -> str:
    """Textured quads, one per card, plus a thin frame tinted by the physics badge."""
    tint = {"articulated": (0.95, 0.6, 0.1), "physics": (0.2, 0.75, 0.35), "": (0.6, 0.6, 0.6)}
    parts = []
    for c in cards:
        r, u = np.asarray(c.right), np.asarray(c.up)
        ctr = np.asarray(c.center)
        hw, hh = c.half_w, c.half_h
        pts = [ctr - r * hw - u * hh, ctr + r * hw - u * hh, ctr + r * hw + u * hh, ctr - r * hw + u * hh]
        frame_pts = [ctr - r * (hw + .01) - u * (hh + .01) + c.normal * -0.002,
                     ctr + r * (hw + .01) - u * (hh + .01) + c.normal * -0.002,
                     ctr + r * (hw + .01) + u * (hh + .01) + c.normal * -0.002,
                     ctr - r * (hw + .01) + u * (hh + .01) + c.normal * -0.002]
        p = ", ".join(f"({x:.4f}, {y:.4f}, {z:.4f})" for x, y, z in pts)
        fp = ", ".join(f"({x:.4f}, {y:.4f}, {z:.4f})" for x, y, z in frame_pts)
        tex = c.thumbnail or ""
        col = tint.get(c.badge, tint[""])
        parts.append(f'''
    def Mesh "Card_{c.index:02d}" (prepend apiSchemas = ["MaterialBindingAPI"])
    {{
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [{p}]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (interpolation = "vertex")
        rel material:binding = </Catalog/{panel_name}/Looks/Mat_{c.index:02d}>
        string card:name = "{c.name}"
        string card:url = "{c.url}"
    }}
    def Mesh "Frame_{c.index:02d}"
    {{
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [{fp}]
        color3f[] primvars:displayColor = [({col[0]}, {col[1]}, {col[2]})]
    }}''')
    looks = "".join(f'''
        def Material "Mat_{c.index:02d}"
        {{
            token outputs:surface.connect = </Catalog/{panel_name}/Looks/Mat_{c.index:02d}/PBR.outputs:surface>
            def Shader "PBR"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:emissiveColor.connect = </Catalog/{panel_name}/Looks/Mat_{c.index:02d}/Tex.outputs:rgb>
                color3f inputs:diffuseColor.connect = </Catalog/{panel_name}/Looks/Mat_{c.index:02d}/Tex.outputs:rgb>
                float inputs:roughness = 1
                token outputs:surface
            }}
            def Shader "Tex"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{c.thumbnail or ""}@
                float2 inputs:st.connect = </Catalog/{panel_name}/Looks/Mat_{c.index:02d}/St.outputs:result>
                float3 outputs:rgb
            }}
            def Shader "St"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }}
        }}''' for c in cards)
    return f'''#usda 1.0
(
    defaultPrim = "Catalog"
    upAxis = "Z"
    metersPerUnit = 1
)
def Scope "Catalog"
{{
    def Xform "{panel_name}"
    {{{"".join(parts)}
        def Scope "Looks"
        {{{looks}
        }}
    }}
}}
'''
