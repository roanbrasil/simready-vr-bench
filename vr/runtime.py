"""The real backend: ovstage (scene) + ovrtx (render) + ovphysx (physics) + ovxr (headset).

Written against the published early-access APIs (ovxr 0.1, ovrtx 0.6, ovstage, ovphysx
0.7). Not executed in my environment (no x86 + RTX + headset there): every call whose
contract I could only read, not run, is marked ``VERIFY``. Check those first if something
breaks on your machine.

Data flow per frame (ovphysx does NOT push transforms into ovstage, so the app does):

    begin_frame -> poll controllers -> physics step -> read body poses
      -> write poses (+ spawns, hands) into ovstage at ordinal o
      -> bind_frame(o) seals o and renders -> update_from_ovstage(o) if topology changed
      -> submit_frame
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from .grab import BodyState, Command
from .xrmath import flat, from_flat, pose_to_matrix

ANCHOR = "/World/Render/Camera"


@dataclass
class Frame:
    plan: object
    ordinal: int
    predicted_time: int


class Ordinals:
    """The app is a co-writer with ovxr, so it owns one ordinal per frame and hands it to bind_frame."""

    def __init__(self, start: int):
        self.cur = start

    def next(self) -> int:
        self.cur += 1
        return self.cur


class OvBackend:
    def __init__(self, env_usd: str, runtime: str = "PhysX", dt: float = 1 / 90):
        import ovphysx
        import ovrtx
        import ovstage
        import ovxr
        from ovstage import population
        from ovstage.types import PopulationDomain  # VERIFY: import path of PopulationDomain

        self.ovxr, self.ovstage, self.population, self.ovphysx = ovxr, ovstage, population, ovphysx
        self.dt, self.runtime = dt, runtime
        self._topology_dirty = False
        self._spawned: list[str] = []
        self.input = None

        # ovphysx: the PhysX schemas must be registered with ovstage before the first USD read
        # in the process, or attach_ovstage raises. VERIFY: name of the registration call.
        if hasattr(ovphysx, "register_ovstage_schemas"):
            ovphysx.register_ovstage_schemas()

        self.renderer = ovrtx.Renderer(ovrtx.RendererConfig(read_gpu_transforms=True, keep_system_alive=True))
        self.stage = ovstage.Stage("simready-vr", config=ovstage.StageConfig(
            runtime_default_ordinal_admission_policy=ovstage.OrdinalAdmissionPolicy.NONDECREASING))
        self.renderer.attach_ovstage(self.stage)
        population.open_usd(self.stage, env_usd, ordinal=1, domains=PopulationDomain.ALL)
        self.stage.advance_write_floor(1).wait()
        self.ordinals = Ordinals(1)

        self.physx = ovphysx.PhysX()
        self.physx.wait_task(self.physx.attach_ovstage(self.stage, read_ordinal=1))
        self.physx.warmup()
        self.physx.wait_all()

        self.ctx = ovxr.Context(ovxr.ContextDesc(
            ovrtx_renderer=self.renderer, ovstage=self.stage, application_name="simready-vr",
            display_mode=ovxr.DisplayMode.STEREO, session_mode=ovxr.SessionMode.VR,
            graphics_api=ovxr.GraphicsApi.VULKAN, create_render_products=True,
            tracking_anchor_path=ANCHOR, scene_info_policy=ovxr.SceneInfoPolicy.AUTO,
        ))
        probe = self.ctx.probe_openxr()
        print(f"[xr] {ovxr.result_name(probe.result)} runtime={getattr(probe, 'runtime_name', '?')}", file=sys.stderr)

    # ------------------------------------------------------------------ frame
    def begin(self) -> Frame | None:
        rc, plan = self.ctx.begin_frame()
        if rc == self.ovxr.Result.NOT_READY:
            return None
        if rc != self.ovxr.Result.SUCCESS:
            raise RuntimeError(f"begin_frame: {self.ovxr.result_name(rc)} {self.ctx.last_error}")
        if self.input is None:
            handles = self.ctx.get_openxr_handles()
            if handles is not None:
                from .openxr_input import XrInput
                self.input = XrInput(handles)
        t = int(getattr(plan, "predicted_display_time", 0))  # VERIFY: field name on the frame plan
        return Frame(plan, self.ordinals.next(), t)

    def hands(self, f: Frame):
        return self.input.poll(f.predicted_time) if self.input else {}

    def to_stage(self, pose) -> np.ndarray:
        pos, quat = pose
        return from_flat(self.ctx.transform_openxr_pose(pos, quat))

    def head_matrix(self) -> np.ndarray | None:
        if self.input and self.input.head:
            return self.to_stage(self.input.head)
        return None

    def step_physics(self) -> list[BodyState]:
        self.physx.step_sync(self.dt)
        return self._read_bodies()

    def _read_bodies(self) -> list[BodyState]:
        from ovphysx.types import ObjectScope, SimObjectType
        out: list[BodyState] = []
        kinds = [("rigid", SimObjectType.RIGID_BODY)]
        link_t = getattr(SimObjectType, "ARTICULATION_LINK", None)  # VERIFY
        if link_t is not None:
            kinds.append(("link", link_t))
        for kind, t in kinds:
            with self.physx.read_sync(t, ["position", "orientation", "linearVelocity"], scope=ObjectScope.ALL) as res:
                for g in res.groups:
                    paths = list(g.prim_list)                      # pairing key, per render_handoff tutorial
                    cols = [np.asarray(tt.numpy()) for tt in g.tensors]
                    pos, quat = cols[0].reshape(-1, 3), cols[1].reshape(-1, 4)
                    vel = cols[2].reshape(-1, 3) if len(cols) > 2 else np.zeros_like(pos)
                    for i, p in enumerate(paths):
                        out.append(BodyState(str(p), kind, pos[i].astype(float), quat[i].astype(float), vel[i].astype(float)))
        return out

    def publish_bodies(self, f: Frame, bodies: list[BodyState]) -> None:
        """Copy simulated poses into ovstage so ovrtx renders them this frame."""
        if bodies:
            self.ctx.update_prim_transforms(
                {b.path: flat(pose_to_matrix(b.position, b.orientation)) for b in bodies}, stage_ordinal=f.ordinal)

    def apply(self, cmds: list[Command]) -> None:
        """Write grab commands. VERIFY: ovphysx write-session API shape (query -> write -> commit)."""
        from ovphysx.types import ObjectScope, SimObjectType
        for c in cmds:
            if c.kind == "rigid":
                t, attrs = SimObjectType.RIGID_BODY, {}
                if c.linear_velocity is not None:
                    attrs["linearVelocity"] = np.asarray([c.linear_velocity], dtype=np.float32)
                if c.angular_velocity is not None:
                    attrs["angularVelocity"] = np.asarray([c.angular_velocity], dtype=np.float32)
                attrs["wake"] = np.asarray([1], dtype=np.uint8)
            else:
                t = getattr(SimObjectType, "ARTICULATION_LINK")
                attrs = {"force": np.asarray([c.force], dtype=np.float32)}
                if c.torque is not None:
                    attrs["wrench"] = np.asarray([np.r_[c.force, c.torque]], dtype=np.float32)
                    attrs.pop("force")
            q = self.physx.query(t, scope=ObjectScope.ALL, prim_paths=[c.path])
            with self.physx.write(q, list(attrs)).wait() as session:
                for g in session.groups:
                    for name, tensor in zip(attrs, g.tensors):
                        np.copyto(np.asarray(tensor.numpy()), attrs[name])
                    session.commit(g)

    # ------------------------------------------------------------------ scene edits
    def spawn(self, f: Frame, prim_name: str, usda: str) -> None:
        self.population.add_usd_reference_from_string(self.stage, usda, "/World/Spawned")
        self.population.apply_usd_changes(self.stage, ordinal=f.ordinal)
        self._spawned.append(prim_name)
        self._topology_dirty = True

    def show_cards(self, f: Frame, usda: str) -> None:
        self.population.add_usd_reference_from_string(self.stage, usda, "/World/Catalog")
        self.population.apply_usd_changes(self.stage, ordinal=f.ordinal)

    def clear_spawned(self, f: Frame) -> None:
        # VERIFY: deactivation through ovstage; fallback is to move them far below the floor
        self.ctx.update_prim_transforms(
            {f"/World/Spawned/{n}": flat(pose_to_matrix((0.0, 0.0, -100.0), (0, 0, 0, 1)))
             for n in self._spawned}, stage_ordinal=f.ordinal)
        self._spawned.clear()

    def end(self, f: Frame) -> None:
        rc, sealed = self.ctx.bind_frame(f.plan, self.dt, stage_ordinal=f.ordinal)
        if rc == self.ovxr.Result.NOT_READY:
            self.ctx.cancel_frame(f.plan)
            return
        if self._topology_dirty and sealed:
            try:
                self.physx.wait_task(self.physx.update_from_ovstage(f.ordinal, f.ordinal))
            except Exception as e:  # VERIFY: runtime topology change after warmup
                print(f"[physx] update_from_ovstage failed ({e}); re-attaching", file=sys.stderr)
                self.physx.wait_task(self.physx.detach_ovstage())
                self.physx.wait_task(self.physx.attach_ovstage(self.stage, read_ordinal=f.ordinal))
                self.physx.warmup()
                self.physx.wait_all()
            self._topology_dirty = False
        self.ctx.submit_frame(f.plan)

    def shutdown(self) -> None:
        self.ctx.destroy()
        self.physx.wait_task(self.physx.detach_ovstage())
        self.renderer.destroy()
        self.stage.destroy()
        self.physx.destroy()
