"""In-headset catalog: search results laid out as cards on an arc in front of the user.

Cards are ordinary USD quads (authored by ``scene.card_panel_usda``) so ovrtx renders
them like anything else; no extra OpenXR composition layer is needed. Selection is a
controller ray against each card rectangle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Card:
    index: int            # 1-based number shown on the card and used by voice ("spawn three")
    name: str
    url: str
    thumbnail: str | None
    center: np.ndarray    # stage units
    normal: np.ndarray    # faces the user
    right: np.ndarray
    up: np.ndarray
    half_w: float
    half_h: float
    badge: str = ""       # "physics" | "articulated" | "" ; from search-time USD properties


@dataclass
class Browser:
    page_size: int = 8
    query: str | None = None
    hits: list = field(default_factory=list)
    page: int = 0
    selected: int | None = None   # 1-based within the page

    def set_results(self, query: str, hits: list) -> None:
        self.query, self.hits, self.page, self.selected = query, list(hits), 0, None

    @property
    def page_count(self) -> int:
        return max(1, -(-len(self.hits) // self.page_size))

    def visible(self) -> list:
        s = self.page * self.page_size
        return self.hits[s:s + self.page_size]

    def next(self) -> None:
        self.page = min(self.page + 1, self.page_count - 1)
        self.selected = None

    def previous(self) -> None:
        self.page = max(self.page - 1, 0)
        self.selected = None

    def resolve(self, index: int | None):
        """Hit for card ``index`` (1-based) on the current page, else the ray-selected one."""
        idx = index if index is not None else self.selected
        vis = self.visible()
        if idx is None or not (1 <= idx <= len(vis)):
            return None
        return vis[idx - 1]


def layout_arc(n: int, head_pos: np.ndarray, head_forward: np.ndarray, up: np.ndarray,
               radius: float = 1.1, height: float = -0.15, span_deg: float = 100.0,
               card_w: float = 0.24, card_h: float = 0.28) -> list[dict]:
    """Card frames on a horizontal arc centred on the user's heading.

    ``head_forward`` is flattened onto the floor plane so looking down at the table does
    not tilt the whole panel into the floor.
    """
    up = up / np.linalg.norm(up)
    f = head_forward - np.dot(head_forward, up) * up
    if np.linalg.norm(f) < 1e-6:
        f = np.cross(up, [1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.cross(up, [0.0, 1.0, 0.0])
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    out = []
    angles = [0.0] if n == 1 else np.linspace(-span_deg / 2, span_deg / 2, n)
    for a in np.radians(angles):
        d = np.cos(a) * f + np.sin(a) * r
        c = head_pos + radius * d + height * up
        normal = -d
        right = np.cross(up, normal)
        right /= np.linalg.norm(right)
        out.append({"center": c, "normal": normal, "right": right, "up": up,
                    "half_w": card_w / 2, "half_h": card_h / 2})
    return out


def ray_pick(origin: np.ndarray, direction: np.ndarray, cards: list[Card], max_dist: float = 5.0) -> Card | None:
    best, best_t = None, max_dist
    for c in cards:
        denom = float(np.dot(direction, c.normal))
        if abs(denom) < 1e-6:
            continue
        t = float(np.dot(c.center - origin, c.normal)) / denom
        if not (0.0 < t < best_t):
            continue
        p = origin + t * direction - c.center
        if abs(np.dot(p, c.right)) <= c.half_w and abs(np.dot(p, c.up)) <= c.half_h:
            best, best_t = c, t
    return best


def badge_for(hit) -> str:
    if getattr(hit, "is_robot", False) or (getattr(hit, "validated_profile", None) and "Robot" in hit.validated_profile):
        return "articulated"
    return "physics" if getattr(hit, "has_rigid_body", False) else ""
