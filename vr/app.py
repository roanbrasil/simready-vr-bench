"""SimReady Central in VR: voice search -> catalog cards -> spawn -> grab with physics.

    python -m vr.app --env my_room.usda            # headset (x86 + RTX + OpenXR runtime)
    python -m vr.app --dry-run --say "find a cardboard box" --say "spawn one"

Controls (Quest Touch / Index):
    hold A (right) or X (left)  push-to-talk  ("find a robot gripper", "spawn three", "next", "clear")
    trigger on a card           spawn it in front of you
    grip                        grab the nearest body (free bodies follow; articulated links pull against their joints)
    B / Y                       next / previous page
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np

from .catalog import Browser, Card, badge_for, layout_arc, ray_pick
from .grab import Grab
from .scene import card_panel_usda, environment_usda, safe_name, spawn_usda
from .voice import parse
from .xrmath import pose_to_matrix, ray_from_matrix


class App:
    def __init__(self, backend, search_fn, thumbs_dir: Path, runtime: str = "PhysX", transcribe=None, ptt=None):
        self.b, self.search_fn, self.thumbs, self.runtime = backend, search_fn, thumbs_dir, runtime
        self.transcribe, self.ptt = transcribe, ptt
        self.browser = Browser()
        self.cards: list[Card] = []
        self.grabs = {"left": Grab(), "right": Grab()}
        self.spawn_count = 0
        self._results: queue.Queue = queue.Queue()
        self._pending_spawn: list = []
        self._panel_gen = 0
        self.events: list[str] = []
        self.threads: list[threading.Thread] = []

    # ------------------------------------------------------------------ commands
    def handle_utterance(self, text: str) -> None:
        intent = parse(text)
        self.events.append(f"heard '{text}' -> {intent.action}")
        if intent.action == "search":
            th = threading.Thread(target=self._search_worker, args=(intent.query,), daemon=True)
            th.start()
            self.threads.append(th)
        elif intent.action == "spawn":
            hit = self.browser.resolve(intent.index or 1)
            if hit is not None:
                self._pending_spawn.append(hit)
        elif intent.action == "next":
            self.browser.next()
            self._results.put(None)
        elif intent.action == "previous":
            self.browser.previous()
            self._results.put(None)
        elif intent.action == "clear":
            self._pending_spawn.append("CLEAR")

    def _search_worker(self, q: str) -> None:
        try:
            hits = self.search_fn(q)
        except Exception as e:
            self.events.append(f"search failed: {e}")
            return
        self._results.put((q, hits))

    def _thumbnail(self, hit) -> str | None:
        url = getattr(hit, "thumbnail_url", None)
        if not url:
            return None
        self.thumbs.mkdir(parents=True, exist_ok=True)
        p = self.thumbs / f"{safe_name(hit.name)}_{abs(hash(hit.url)) % 10**8}.jpg"
        if not p.exists():
            try:
                urllib.request.urlretrieve(url, p)
            except Exception:
                return None
        return str(p)

    def _rebuild_cards(self, f) -> None:
        head = self.b.head_matrix()
        if head is None:
            return
        fwd = np.array([0.0, 0.0, -1.0]) @ head[:3, :3]  # VIEW space looks down -Z
        vis = self.browser.visible()
        frames = layout_arc(len(vis), head[3, :3], fwd, np.array([0.0, 0.0, 1.0]))
        self.cards = [Card(i + 1, h.name, h.url, self._thumbnail(h), badge=badge_for(h), **fr)
                      for i, (h, fr) in enumerate(zip(vis, frames))]
        self._panel_gen += 1
        self.b.show_cards(f, card_panel_usda(self.cards, panel_name=f"Panel_{self._panel_gen}"))
        self.events.append(f"cards {len(self.cards)} for '{self.browser.query}' page {self.browser.page + 1}")

    def _spawn_pose(self) -> np.ndarray:
        head = self.b.head_matrix()
        if head is None:
            return pose_to_matrix((0.8, 0.0, 1.0), (0, 0, 0, 1))
        fwd = np.array([0.0, 0.0, -1.0]) @ head[:3, :3]
        fwd[2] = 0.0
        fwd = fwd / (np.linalg.norm(fwd) or 1.0)
        pos = head[3, :3] + 0.6 * fwd
        pos[2] = max(head[3, 2] - 0.55, 0.3)   # roughly table height for a standing user
        return pose_to_matrix(pos, (0, 0, 0, 1))

    # ------------------------------------------------------------------ frame
    def tick(self) -> bool:
        f = self.b.begin()
        if f is None:
            return False
        hands = self.b.hands(f)
        bodies = self.b.step_physics()
        by_path = {x.path: x for x in bodies}
        cmds = []
        for side, h in hands.items():
            g = self.grabs[side]
            if h.grip is not None:
                hand_M = self.b.to_stage(h.grip)
                if "grab" in h.pressed and not g.holding:
                    got = g.begin(hand_M, bodies)
                    if got:
                        self.events.append(f"{side} grab {got}")
                if g.holding:
                    if "grab" in h.released or g.held not in by_path:
                        c = g.end()
                        self.events.append(f"{side} release")
                        if c:
                            cmds.append(c)
                    else:
                        cmds.append(g.update(getattr(f, "t", 0.0), self.b.dt, hand_M, by_path[g.held]))
            if h.aim is not None and "select" in h.pressed and self.cards:
                o, d = ray_from_matrix(self.b.to_stage(h.aim))
                card = ray_pick(o, d, self.cards)
                if card:
                    self.browser.selected = card.index
                    self._pending_spawn.append(self.browser.resolve(card.index))
            if "next" in h.pressed:
                self.handle_utterance("next")
            if "prev" in h.pressed:
                self.handle_utterance("previous")
            if self.ptt and "talk" in h.pressed:
                self.ptt.start()
            if self.ptt and "talk" in h.released and self.transcribe:
                text = self.transcribe(self.ptt.stop())
                if text:
                    self.handle_utterance(text)
        if cmds:
            self.b.apply(cmds)
        self.b.publish_bodies(f, bodies)

        while not self._results.empty():
            item = self._results.get_nowait()
            if item is not None:
                self.browser.set_results(*item)
            self._rebuild_cards(f)
        for hit in self._pending_spawn:
            if hit == "CLEAR":
                self.b.clear_spawned(f)
                self.events.append("clear")
                continue
            name = safe_name(hit.name, self.spawn_count)
            self.spawn_count += 1
            self.b.spawn(f, name, spawn_usda(name, hit.url, self._spawn_pose(), self.runtime))
            self.events.append(f"spawn {name}")
        self._pending_spawn.clear()
        self.b.end(f)
        return True


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", help="environment USD (default: a generated floor+table room)")
    ap.add_argument("--runtime", default="PhysX", choices=["PhysX", "Newton", "MuJoCo"])
    ap.add_argument("--dry-run", action="store_true", help="headless fake backend, no headset")
    ap.add_argument("--say", action="append", default=[], help="(dry-run) utterances fed at startup")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--no-voice", action="store_true")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from srvr.search import search

    cache = Path.home() / ".cache" / "simready-vr"
    if args.dry_run:
        from .fake import FakeBackend
        backend = FakeBackend()
        app = App(backend, lambda q: search(q, limit=8), cache / "thumbs", args.runtime)
        for s in args.say:
            app.handle_utterance(s)
            for th in app.threads:      # headless: wait for the search instead of racing it
                th.join(timeout=60)
            app.tick()
        n = args.frames or 360
    else:
        env = args.env
        if env is None:
            cache.mkdir(parents=True, exist_ok=True)
            env = str(cache / "room.usda")
            Path(env).write_text(environment_usda())
        from .runtime import OvBackend
        backend = OvBackend(env, runtime=args.runtime)
        transcribe = ptt = None
        if not args.no_voice:
            from .voice import PushToTalk, Transcriber
            asr = Transcriber()
            transcribe, ptt = asr.transcribe, PushToTalk()
        app = App(backend, lambda q: search(q, limit=24), cache / "thumbs", args.runtime, transcribe, ptt)
        n = args.frames

    done = 0
    try:
        while n == 0 or done < n:
            if app.tick():
                done += 1
    except KeyboardInterrupt:
        pass
    finally:
        backend.shutdown()
        for e in app.events:
            print(e)


if __name__ == "__main__":
    main()
