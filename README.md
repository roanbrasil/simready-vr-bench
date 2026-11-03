# simready-vr-bench

Two things in one repo:

1. **`srvr/` + `scripts/`: a contract audit of NVIDIA SimReady Central.** It fetches packages at the commit the public index pins, then compares what each package *declares* (validation/benchmark sidecars, `SimReady_Metadata`) with what its composed USD *contains*, once per physics runtime variant (PhysX / Newton / MuJoCo). It needs only `usd-core` and network access, no GPU.
2. **`vr/`: a VR app** that searches SimReady Central by voice, shows results as cards in the headset, spawns assets with physics and lets you grab them. Built on `ovxr` + `ovrtx` + `ovstage` + `ovphysx`.

## Audit

```bash
pip install usd-core huggingface_hub pandas
git clone --depth 1 https://github.com/NVIDIA-Omniverse/simready-central
python scripts/run_audit.py --submissions simready-central/submissions --cap 60 --cache cache --out results
python scripts/summarize.py --results results --cache cache     # -> results/summary.json
```

About 15 minutes for 333 packages over an unauthenticated HF connection. Set `HF_TOKEN` for higher rate limits.

Findings it reports (`srvr/audit.py::derive_findings`):

| Finding | Meaning |
|---|---|
| `DECLARED_RIGID_BUT_NO_COLLIDER` | validation passes `FET_003_PHYSX` but the PhysX-composed stage has bodies and zero active colliders |
| `VISIBLE_BODY_WITHOUT_COLLIDER` | a rigid body has renderable geometry but no active collider in its subtree |
| `BODY_WITHOUT_MASS_SOURCE` | no explicit mass, and no density on body, collider or bound physics material |
| `METADATA_MASS_MISMATCH` | every body has explicit mass and the sum is >10% off `SimReady_Metadata.mass` |
| `JOINT_WITHOUT_LIMITS` | revolute/prismatic joint without finite limits (often intentional: wheels, drums) |
| `CLAIMS_IN_USD_ONLY` / `NO_DECLARED_CLAIMS` | validation lives only in `SimReady_Metadata` (legacy `FET003_BASE_*` dialect) / nowhere |
| `RUNTIME_BODY_COUNT_DIVERGES` | rigid-body count differs between enabled runtime variants |

## VR app

Requirements: **Linux or Windows x86_64** (ovxr 0.1 has no aarch64 build yet, so DGX Spark / Jetson are out for now), an NVIDIA RTX GPU with a supported driver, an OpenXR runtime (SteamVR, Meta Quest Link, Monado...), and a headset.

```bash
pip install ovrtx ovstage ovphysx usd-core numpy pyopenxr faster-whisper sounddevice
pip install <ovxr wheel from github.com/NVIDIA-Omniverse/ovxr/releases>
export OVX_PHONE_HOME_OPT_OUT=1          # optional: disable ovxr telemetry

python -m vr.app                         # generated floor + table room
python -m vr.app --env my_room.usda      # your own environment (needs /World/Render/Camera as tracking anchor)
python -m vr.app --runtime Newton
python -m vr.app --dry-run --say "find a cardboard box" --say "spawn one"   # no headset, fake backend
```

Controls (Touch / Index): hold **A** (or X) to talk, **trigger** on a card spawns it, **grip** grabs the nearest body within 12 cm, **B / Y** pages. Voice works in EN and PT-BR: "find a robot gripper", "procura uma empilhadeira", "spawn three", "coloca o dois", "next", "clear".

### Status

- Tested (40 tests, `python -m pytest -q tests`): audit logic on synthetic packages, pose math, both grab control laws, voice parser, catalog layout/picking, USDA composition, and the full app loop against `vr/fake.py`.
- **Not executed**: `vr/runtime.py` (ovrtx/ovstage/ovphysx/ovxr) and `vr/openxr_input.py`. They are written against the early-access docs, and every call I could only read, not run, is tagged `VERIFY`. Check these first:
  1. `ovphysx.update_from_ovstage()` with new prims after `warmup()`. If it refuses, the code falls back to detach + re-attach.
  2. the `ovphysx` write-session shape in `OvBackend.apply()`.
  3. the predicted-display-time field name on `ovxr`'s frame plan (`OvBackend.begin()`).
  4. the PhysX-schema registration call before the first USD read (`OvBackend.__init__`).

### Design notes

- `ovphysx` doesn't push transforms into `ovstage`, so each frame the app reads body poses with `read_sync` and writes them back with `ovxr.Context.update_prim_transforms(..., stage_ordinal=o)`.
- The app is a co-writer with ovxr: one ordinal per frame goes to `bind_frame(plan, dt, stage_ordinal=o)`, and the stage uses `NONDECREASING` admission.
- OpenXR input resolves every function through ovxr's `xrGetInstanceProcAddr`. pyopenxr is used only for its ctypes structs, because handles from one loader aren't valid in another.
- Grab: free bodies use velocity tracking (they stay dynamic and keep colliding); articulation links use a critically damped force spring, because link pose and velocity are read-only in ovphysx.
- Spawn only knows the per-runtime variantSet scheme. Packages with a single `Physics` set (URDF-converter output) or no variants spawn with their default physics.

## Layout

```
srvr/      index parsing, selective HF fetch, USD Search client, audit
scripts/   run_audit.py, summarize.py
vr/        app.py (loop), runtime.py (real backend), fake.py, openxr_input.py, grab.py, catalog.py, scene.py, voice.py, xrmath.py
tests/     pytest
results/   audit output written by the scripts (not committed)
```

Licenses: this code is MIT. ovxr/ovrtx/ovphysx/ovstage are NVIDIA proprietary. SimReady assets are mostly CC-BY-4.0, so check each package's `com.nvidia.simready.packaging.json`.
