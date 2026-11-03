"""Physics-respecting grab, using only attributes ovphysx documents as writable.

Two cases, because ovphysx exposes them differently:

* Free rigid body (pose/velocity writable): velocity tracking. Each frame we write
  ``linearVelocity``/``angularVelocity`` that would carry the body to the hand target in
  one step, clamped. The body stays dynamic, so it still collides and can't tunnel
  through the table the way a teleported (kinematic) body would.
* Articulation link (pose/velocity read-only, ``force``/``wrench`` writable): a
  critically-damped spring on the grab point. The joints decide where the link can go,
  so pulling a drawer slides it and pulling a finger opens it along its prismatic axis.

The hand keeps the offset it had at grab time (``grab_offset``), so the object does not
snap its origin into the palm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .xrmath import matrix_to_pose, pose_to_matrix, quat_conj, quat_mul, rigid_inverse, rotation_vector


@dataclass
class BodyState:
    path: str
    kind: str                 # "rigid" | "link"
    position: np.ndarray
    orientation: np.ndarray   # xyzw
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 1.0


@dataclass
class Command:
    path: str
    kind: str
    linear_velocity: np.ndarray | None = None
    angular_velocity: np.ndarray | None = None
    force: np.ndarray | None = None
    torque: np.ndarray | None = None


@dataclass
class GrabParams:
    reach: float = 0.12           # m, max distance from hand to body origin to start a grab
    max_lin_speed: float = 6.0    # m/s clamp, keeps a fast hand from launching objects
    max_ang_speed: float = 12.0   # rad/s
    link_stiffness: float = 400.0  # N/m per kg of link mass
    link_ang_stiffness: float = 8.0
    throw_gain: float = 1.0


def pick_body(hand_pos: np.ndarray, bodies: list[BodyState], reach: float) -> BodyState | None:
    best, best_d = None, reach
    for b in bodies:
        d = float(np.linalg.norm(b.position - hand_pos))
        if d <= best_d:
            best, best_d = b, d
    return best


class Grab:
    """One hand's grab state machine: idle -> holding -> idle."""

    def __init__(self, params: GrabParams | None = None):
        self.p = params or GrabParams()
        self.held: str | None = None
        self.kind: str | None = None
        self.offset: np.ndarray | None = None   # body_world = offset @ hand_world
        self._hand_hist: list[tuple[float, np.ndarray]] = []

    @property
    def holding(self) -> bool:
        return self.held is not None

    def begin(self, hand_M: np.ndarray, bodies: list[BodyState]) -> str | None:
        hand_pos = hand_M[3, :3]
        b = pick_body(hand_pos, bodies, self.p.reach)
        if b is None:
            return None
        body_M = pose_to_matrix(b.position, b.orientation)
        self.offset = body_M @ rigid_inverse(hand_M)
        self.held, self.kind = b.path, b.kind
        self._hand_hist.clear()
        return b.path

    def target(self, hand_M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return matrix_to_pose(self.offset @ hand_M)

    def update(self, t: float, dt: float, hand_M: np.ndarray, body: BodyState) -> Command:
        assert self.holding and body.path == self.held
        self._hand_hist.append((t, hand_M[3, :3].copy()))
        self._hand_hist = self._hand_hist[-6:]
        tp, tq = self.target(hand_M)
        dq = quat_mul(tq, quat_conj(body.orientation))
        rotvec = rotation_vector(dq)
        if body.kind == "rigid":
            v = (tp - body.position) / max(dt, 1e-4)
            w = rotvec / max(dt, 1e-4)
            return Command(body.path, "rigid", _clamp(v, self.p.max_lin_speed), _clamp(w, self.p.max_ang_speed))
        # articulation link: critically damped spring, scaled by mass
        k = self.p.link_stiffness * body.mass
        c = 2.0 * np.sqrt(k * body.mass)
        f = k * (tp - body.position) - c * body.linear_velocity
        ka = self.p.link_ang_stiffness * body.mass
        tau = ka * rotvec - 2.0 * np.sqrt(ka * body.mass) * 0.1 * body.angular_velocity
        return Command(body.path, "link", force=f, torque=tau)

    def end(self) -> Command | None:
        """Release. For a free body, hand it the hand's recent velocity so it can be thrown."""
        cmd = None
        if self.kind == "rigid" and len(self._hand_hist) >= 2:
            (t0, p0), (t1, p1) = self._hand_hist[0], self._hand_hist[-1]
            if t1 > t0:
                v = (p1 - p0) / (t1 - t0) * self.p.throw_gain
                cmd = Command(self.held, "rigid", _clamp(v, self.p.max_lin_speed), None)
        self.held = self.kind = self.offset = None
        self._hand_hist.clear()
        return cmd


def _clamp(v: np.ndarray, limit: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n <= limit else v * (limit / n)
