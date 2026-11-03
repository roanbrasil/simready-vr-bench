"""Small rigid-transform helpers in the convention ovxr/ovstage use.

Matrices are 4x4, row-major, row-vector (p' = p @ M), translation in M[3, :3]
(flat indices 12..14). Quaternions are (x, y, z, w), as OpenXR and ovphysx report them.
"""
from __future__ import annotations

import numpy as np


def quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("zero quaternion (unlocated space?)")
    return q / n


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product a*b, xyzw. Applying (a*b) rotates by b first, then a."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_conj(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def quat_to_rot(q) -> np.ndarray:
    """Column-vector rotation matrix R (v' = R v) for unit quaternion q."""
    x, y, z, w = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_to_quat(R) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    return quat_normalize(q)


def pose_to_matrix(pos, quat) -> np.ndarray:
    """Row-vector 4x4 from position + xyzw quaternion."""
    M = np.eye(4)
    M[:3, :3] = quat_to_rot(quat).T  # row-vector convention stores R transposed
    M[3, :3] = pos
    return M


def matrix_to_pose(M) -> tuple[np.ndarray, np.ndarray]:
    M = np.asarray(M, dtype=float).reshape(4, 4)
    return M[3, :3].copy(), rot_to_quat(M[:3, :3].T)


def flat(M) -> list[float]:
    return [float(v) for v in np.asarray(M).reshape(16)]


def from_flat(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(4, 4)


def rigid_inverse(M) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    R = M[:3, :3]
    inv = np.eye(4)
    inv[:3, :3] = R.T
    inv[3, :3] = -M[3, :3] @ R.T
    return inv


def rotation_vector(q_err) -> np.ndarray:
    """Axis*angle (radians) of a unit quaternion, shortest path."""
    q = quat_normalize(q_err)
    if q[3] < 0:
        q = -q
    s = np.linalg.norm(q[:3])
    if s < 1e-9:
        return 2.0 * q[:3]
    angle = 2.0 * np.arctan2(s, q[3])
    return q[:3] / s * angle


def ray_from_matrix(M, forward=(0.0, 0.0, -1.0)) -> tuple[np.ndarray, np.ndarray]:
    """Origin and unit direction of a controller ray (OpenXR aim pose looks down local -Z)."""
    M = np.asarray(M, dtype=float)
    d = np.asarray(forward, dtype=float) @ M[:3, :3]
    return M[3, :3].copy(), d / np.linalg.norm(d)
