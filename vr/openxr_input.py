"""Controller input on the OpenXR session that ovxr owns.

ovxr exposes the raw handles (instance, session, reference space) and the
``xrGetInstanceProcAddr`` it was loaded with. Every OpenXR call here goes through that
function pointer, never through pyopenxr's own loader: handles from one loader are not
valid in another loader's trampolines. pyopenxr is used only for its ctypes structs.

ovxr's ownership rules (README "Direct OpenXR input poses"): we may create actions,
action spaces and locate them; we must not wait/begin/end frames or touch session state.
xrAttachSessionActionSets is allowed once per session, so all actions live in one set.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass, field

HANDS = ("left", "right")

# interaction profile -> {action: component suffix}; first match the runtime accepts wins
PROFILES = {
    "/interaction_profiles/oculus/touch_controller": {
        "grab": "squeeze/value", "select": "trigger/value", "talk": "a/click|x/click",
        "next": "b/click", "prev": "y/click"},
    "/interaction_profiles/valve/index_controller": {
        "grab": "squeeze/value", "select": "trigger/value", "talk": "a/click",
        "next": "b/click"},   # Index has no Y; "previous" stays a voice command there
    "/interaction_profiles/khr/simple_controller": {
        "grab": "select/click", "select": "select/click", "talk": "menu/click"},
}


@dataclass
class HandState:
    located: bool = False
    aim: tuple | None = None    # (position xyz meters, orientation xyzw) in ovxr's reference space
    grip: tuple | None = None
    grab: float = 0.0
    select: float = 0.0
    talk: bool = False
    next: bool = False
    prev: bool = False
    pressed: set = field(default_factory=set)    # rising edges this frame
    released: set = field(default_factory=set)


class XrInput:
    def __init__(self, handles):
        import xr  # pyopenxr, structs only
        self.xr = xr
        self.h = handles
        self.instance = ctypes.cast(ctypes.c_void_p(handles.xr_instance), xr.Instance)
        self.session = ctypes.cast(ctypes.c_void_p(handles.xr_session), xr.Session)
        self.base_space = ctypes.cast(ctypes.c_void_p(handles.xr_space), xr.Space)
        self._gipa = xr.PFN_xrGetInstanceProcAddr(handles.xr_get_instance_proc_addr)
        self._fn = {}
        self.paths = {h: self._path(f"/user/hand/{h}") for h in HANDS}
        self._prev_bool: dict = {}
        self._create_actions()

    # -------------------------------------------------------------- plumbing
    def _call(self, name: str, *args):
        if name not in self._fn:
            pfn = getattr(self.xr, f"PFN_{name}")
            out = self.xr.PFN_xrVoidFunction()
            rc = self._gipa(self.instance, name.encode(), ctypes.byref(out))
            if rc != 0:
                raise RuntimeError(f"xrGetInstanceProcAddr({name}) -> {rc}")
            self._fn[name] = ctypes.cast(out, pfn)
        rc = self._fn[name](*args)
        if int(rc) < 0:
            raise RuntimeError(f"{name} -> XrResult {int(rc)}")
        return int(rc)

    def _path(self, s: str):
        p = self.xr.Path()
        self._call("xrStringToPath", self.instance, s.encode(), ctypes.byref(p))
        return p

    # -------------------------------------------------------------- setup
    def _create_actions(self):
        xr = self.xr
        aset = xr.ActionSet()
        self._call("xrCreateActionSet", self.instance,
                   ctypes.byref(xr.ActionSetCreateInfo("simready_vr", "SimReady VR", 0)), ctypes.byref(aset))
        self.action_set = aset
        sub = (xr.Path * 2)(*[self.paths[h] for h in HANDS])
        spec = {"aim": xr.ActionType.POSE_INPUT, "grip": xr.ActionType.POSE_INPUT,
                "grab": xr.ActionType.FLOAT_INPUT, "select": xr.ActionType.FLOAT_INPUT,
                "talk": xr.ActionType.BOOLEAN_INPUT, "next": xr.ActionType.BOOLEAN_INPUT,
                "prev": xr.ActionType.BOOLEAN_INPUT}
        self.actions = {}
        for name, kind in spec.items():
            a = xr.Action()
            info = xr.ActionCreateInfo(action_name=name, action_type=kind, count_subaction_paths=2,
                                       subaction_paths=sub, localized_action_name=name)
            self._call("xrCreateAction", aset, ctypes.byref(info), ctypes.byref(a))
            self.actions[name] = a

        accepted = []
        for profile, comps in PROFILES.items():
            binds = []
            for h in HANDS:
                binds.append((self.actions["aim"], f"/user/hand/{h}/input/aim/pose"))
                binds.append((self.actions["grip"], f"/user/hand/{h}/input/grip/pose"))
                for act, comp in comps.items():
                    options = comp.split("|")
                    chosen = options[0] if h == "right" or len(options) == 1 else options[-1]
                    binds.append((self.actions[act], f"/user/hand/{h}/input/{chosen}"))
            arr = (xr.ActionSuggestedBinding * len(binds))(
                *[xr.ActionSuggestedBinding(a, self._path(p)) for a, p in binds])
            sugg = xr.InteractionProfileSuggestedBinding(
                interaction_profile=self._path(profile), count_suggested_bindings=len(binds),
                suggested_bindings=arr)
            try:
                self._call("xrSuggestInteractionProfileBindings", self.instance, ctypes.byref(sugg))
                accepted.append(profile)
            except RuntimeError:
                pass  # runtime does not know this profile
        if not accepted:
            raise RuntimeError("runtime accepted no interaction profile")
        self.accepted_profiles = accepted

        sets = (xr.ActionSet * 1)(aset)
        self._call("xrAttachSessionActionSets", self.session,
                   ctypes.byref(xr.SessionActionSetsAttachInfo(count_action_sets=1, action_sets=sets)))

        self.spaces = {}
        # VIEW space = the head; used to lay out the catalog arc in front of the user
        view = xr.Space()
        self._call("xrCreateReferenceSpace", self.session,
                   ctypes.byref(xr.ReferenceSpaceCreateInfo(reference_space_type=xr.ReferenceSpaceType.VIEW,
                                                            pose_in_reference_space=xr.Posef())),
                   ctypes.byref(view))
        self.view_space = view
        for pose in ("aim", "grip"):
            for h in HANDS:
                sp = xr.Space()
                info = xr.ActionSpaceCreateInfo(action=self.actions[pose], subaction_path=self.paths[h],
                                                pose_in_action_space=xr.Posef())
                self._call("xrCreateActionSpace", self.session, ctypes.byref(info), ctypes.byref(sp))
                self.spaces[(pose, h)] = sp

    # -------------------------------------------------------------- per frame
    def poll(self, predicted_display_time: int) -> dict[str, HandState]:
        """Call after ovxr's begin_frame, with the plan's predicted display time."""
        xr = self.xr
        active = (xr.ActiveActionSet * 1)(xr.ActiveActionSet(self.action_set, xr.NULL_PATH))
        self._call("xrSyncActions", self.session,
                   ctypes.byref(xr.ActionsSyncInfo(count_active_action_sets=1, active_action_sets=active)))
        out = {}
        for h in HANDS:
            s = HandState()
            for pose in ("aim", "grip"):
                loc = xr.SpaceLocation()
                self._call("xrLocateSpace", self.spaces[(pose, h)], self.base_space,
                           xr.Time(predicted_display_time), ctypes.byref(loc))
                ok = (loc.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT) and \
                     (loc.location_flags & xr.SPACE_LOCATION_ORIENTATION_VALID_BIT)
                if ok:
                    o, p = loc.pose.orientation, loc.pose.position
                    setattr(s, pose, ((p.x, p.y, p.z), (o.x, o.y, o.z, o.w)))
                    s.located = True
            for name in ("grab", "select"):
                st = xr.ActionStateFloat()
                self._call("xrGetActionStateFloat", self.session,
                           ctypes.byref(xr.ActionStateGetInfo(self.actions[name], self.paths[h])), ctypes.byref(st))
                setattr(s, name, float(st.current_state) if st.is_active else 0.0)
            for name in ("talk", "next", "prev"):
                st = xr.ActionStateBoolean()
                self._call("xrGetActionStateBoolean", self.session,
                           ctypes.byref(xr.ActionStateGetInfo(self.actions[name], self.paths[h])), ctypes.byref(st))
                setattr(s, name, bool(st.current_state) and bool(st.is_active))
            _edges(s, self._prev_bool.setdefault(h, {}))
            out[h] = s
        loc = xr.SpaceLocation()
        self._call("xrLocateSpace", self.view_space, self.base_space, xr.Time(predicted_display_time), ctypes.byref(loc))
        self.head = None
        if loc.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
            o, p = loc.pose.orientation, loc.pose.position
            self.head = ((p.x, p.y, p.z), (o.x, o.y, o.z, o.w))
        return out


def _edges(s: HandState, prev: dict, threshold: float = 0.6) -> None:
    """Rising/falling edges, with hysteresis on the analog triggers."""
    cur = {
        "grab": s.grab > (threshold - 0.2 if prev.get("grab") else threshold),
        "select": s.select > (threshold - 0.2 if prev.get("select") else threshold),
        "talk": s.talk, "next": s.next, "prev": s.prev,
    }
    for k, v in cur.items():
        if v and not prev.get(k):
            s.pressed.add(k)
        if not v and prev.get(k):
            s.released.add(k)
        prev[k] = v
