"""The whole app loop against the fake backend: search -> cards -> spawn -> grab -> throw."""
from dataclasses import dataclass

from vr.app import App
from vr.fake import FakeBackend


@dataclass
class FakeHit:
    name: str
    url: str
    validated_profile: str | None = None
    has_rigid_body: bool = True
    thumbnail_url: str | None = None


def test_end_to_end(tmp_path):
    hits = [FakeHit(f"box_{i}.usd", f"https://example/box_{i}.usd") for i in range(10)]
    backend = FakeBackend()
    app = App(backend, lambda q: hits, tmp_path)
    app.handle_utterance("find a cardboard box")
    # the search runs on a thread; wait for it like the real loop would
    for t in app.threads:
        t.join(timeout=5)
    app.tick()
    assert app.browser.query == "cardboard box" and len(app.cards) == 8
    app.handle_utterance("spawn three")
    for _ in range(300):
        app.tick()
    ev = "\n".join(app.events)
    assert "spawn box_2_00" in ev
    assert "right grab /World/Spawned/box_00" in ev
    assert "right release" in ev
    # released with forward hand velocity -> the box flew forward (+X) before landing
    box = backend.bodies["/World/Spawned/box_00"]
    assert box.position[0] > 0.8
    assert any(l.startswith("cmd rigid") for l in backend.log)
