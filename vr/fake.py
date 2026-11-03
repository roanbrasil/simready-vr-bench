"""Headless stand-in for OvBackend, so the app loop runs (and is tested) without a headset.

A scripted right hand reaches for a box on the table, grabs it, lifts, and throws it.
Physics is a toy: gravity, a floor/table plane, velocity integration. Enough to check
that the orchestration in app.py issues the right calls in the right order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grab import BodyState, Command
from .openxr_input import HandState, _edges
from .xrmath import pose_to_matrix


@dataclass
class FakeFrame:
    ordinal: int
    predicted_time: int
    t: float


class FakeBackend:
    def __init__(self, dt: float = 1 / 90, script: list[tuple[float, str]] | None = None):
        self.dt, self.t, self.ord = dt, 0.0, 1
        self.log: list[str] = []
        self.bodies = {"/World/Spawned/box_00": BodyState(
            "/World/Spawned/box_00", "rigid", np.array([0.6, 0.0, 0.80]), np.array([0, 0, 0, 1.0]), mass=1.0)}
        self._prev: dict = {"left": {}, "right": {}}
        self.script = script or []     # (time, utterance) pairs fed to the voice handler
        # head at 1.6 m looking along +X (OpenXR views look down local -Z; -90 deg about Y maps it to +X)
        self.head = ((0.0, 0.0, 1.6), (0.0, -0.70710678, 0.0, 0.70710678))

    # same surface as OvBackend -------------------------------------------------
    def begin(self):
        self.ord += 1
        return FakeFrame(self.ord, int(self.t * 1e9), self.t)

    def _hand_path(self, t: float):
        # 0-1 s reach to box, 1-2 s hold+lift, 2-2.3 s fling forward, then release
        box0 = np.array([0.6, 0.0, 0.80])
        start = np.array([0.2, -0.2, 1.1])
        if t < 1.0:
            p = start + (box0 - start) * t
            grab = 0.0 if t < 0.9 else 1.0
        elif t < 2.0:
            p = box0 + np.array([0.0, 0.0, 0.3 * (t - 1.0)])
            grab = 1.0
        elif t < 2.3:
            p = box0 + np.array([0.0, 0.0, 0.3]) + np.array([2.0, 0.0, 0.5]) * (t - 2.0)
            grab = 1.0
        else:
            p = box0 + np.array([0.6, 0.0, 0.45])
            grab = 0.0
        return p, grab

    def hands(self, f):
        p, g = self._hand_path(f.t)
        r = HandState(located=True, grip=(tuple(p), (0, 0, 0, 1)), aim=(tuple(p), (0, 0, 0, 1)), grab=g)
        _edges(r, self._prev["right"])
        left = HandState()
        _edges(left, self._prev["left"])
        return {"left": left, "right": r}

    def to_stage(self, pose):
        return pose_to_matrix(np.asarray(pose[0]), np.asarray(pose[1]))

    def head_matrix(self):
        return self.to_stage(self.head)

    def step_physics(self):
        for b in self.bodies.values():
            b.linear_velocity = b.linear_velocity + np.array([0, 0, -9.81]) * self.dt
            b.position = b.position + b.linear_velocity * self.dt
            floor = 0.80 if abs(b.position[0] - 0.9) < 0.8 and abs(b.position[1]) < 0.4 else 0.05
            if b.position[2] < floor:
                b.position[2] = floor
                b.linear_velocity = np.zeros(3)
        self.t += self.dt
        return list(self.bodies.values())

    def publish_bodies(self, f, bodies):
        pass

    def apply(self, cmds: list[Command]):
        for c in cmds:
            b = self.bodies[c.path]
            if c.linear_velocity is not None:
                b.linear_velocity = np.asarray(c.linear_velocity, dtype=float)
            self.log.append(f"cmd {c.kind} {c.path}")

    def spawn(self, f, prim_name, usda):
        self.log.append(f"spawn {prim_name}")
        self.bodies[f"/World/Spawned/{prim_name}"] = BodyState(
            f"/World/Spawned/{prim_name}", "rigid", np.array([0.9, 0.0, 1.0]), np.array([0, 0, 0, 1.0]))

    def show_cards(self, f, usda):
        n_cards = usda.count('def Mesh "Card_')
        self.log.append(f"cards {n_cards}")

    def clear_spawned(self, f):
        self.log.append("clear")

    def end(self, f):
        pass

    def shutdown(self):
        self.log.append("shutdown")
