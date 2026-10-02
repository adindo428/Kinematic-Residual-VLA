# DynamicVLA tracking extension

Audited upstream commit: [`434dc7105cf6d559018d15af5fecbcdad44f036e`](https://github.com/hzxie/DynamicVLA/tree/434dc7105cf6d559018d15af5fecbcdad44f036e), September 30, 2026. [upstream.json](upstream.json) pins the inspected files and permitted patched hashes. The audit checkout is `/tmp/dom-upstream-audit`; it is disposable. The collection code and existing pilot were not changed by this integration.

## Recommendation and current status

Use DOM's Franka, asset conventions, cameras, observation/action schema, and training reader, and extend its task family with `track`. A native upstream generation path is a sensible next implementation milestone. It requires changes to the rollout lifecycle as well as an expert class; adding a YAML task alone will fail. Shared assets and serialization make this a DOM-compatible extension, not an official addition to `hzxie/DOM` or an identical motion distribution.

Implemented here: a minimal tracking instruction-parser patch, an optional disk-only reader dependency patch, and a check against the **actual upstream dataset reader**. The native `simulate.py --task track` extension is **not implemented yet**. Existing collection remains through `dom_tracking collect`.

The pilot passed 18 upstream-reader sample checks: beginning/middle/end of one episode from each split, both absolute and delta actions, three-frame observation histories, five-action chunks, and all three decoded camera streams. Task parsing and legacy pick/place/long-horizon instruction regression checks passed. The report is [the reader validation report](../../verification/upstream-reader-validation.json). Tested reader versions: LeRobot 0.3.3, datasets 3.6.0, PyTorch 2.10.0, torchvision 0.25.0, TorchCodec 0.10.0. This verifies data loading; it does not test a model forward pass, checkpoint resumption, or upstream simulation/replay.

## What the upstream implementation provides

The [generation workflow](https://github.com/hzxie/DynamicVLA/tree/434dc7105cf6d559018d15af5fecbcdad44f036e#simulated-dataset-generation) runs `simulate.py` → raw HDF5/scene JSON → `translate_dataset_seq.py` replay → `create_lerobot_dataset.py --rotation euler` → LeRobot v2.1. Its documented environment is Isaac Sim 5.1.0 / Isaac Lab 2.3.2.

| Upstream location | Existing behavior | Required tracking adaptation |
|---|---|---|
| `simulations/configs/sim_cfg.yaml` | Registers `pick`, `place`, `long-horizon` by expert class path | Register one `track` expert with fixed duration and explicit tracking parameters |
| `simulations/simulate.py:get_state_machine` | Dynamically instantiates a class; `compute(curr_state)` returns pose/quaternion/gripper and `sm_state` | Bridge our expert into this interface; supply tool velocity, Jacobian, limits and controller diagnostics separately |
| `simulate.py:get_env_cfg` and scene configs | Random scenes/table anchors, objects, containers, lighting | Consume our audited eligible manifest and verified fixed layout; zero containers, one fixed object |
| Object initialization and event configuration | Static or initially moving physical objects, friction/optional perturbations | Apply seeded bounded 3D kinematic motion each physics step; fixed orientation; log analytic velocity and acceleration |
| `simulations/configs/env_cfg.py` | `dt=.04`, `decimation=1` (25 Hz) | 200 Hz expert/physics and separate 25 Hz capture; avoid labeling 200 Hz records as 25 FPS |
| `termination_cfg.py:get_done_term` | Recognizes only `object_picked`, `objects_placed` | Add fixed-duration tracking completion with error/contact/limit gates; allow five continuation frames |
| `simulate.py:simulate/main` | Saves successful manipulation episodes; filters stopped moving objects | Preserve complete accepted tracking episodes, remove inappropriate stopped-object filter, retain rejected diagnostics and resumable attempts |
| `translate_dataset_seq.py` | Replays five-frame-ahead source poses; pads the end with final actions; uses manipulation completion terms | Replay the exact target path and accept tracking duration; capture real continuation instead of terminal padding; revalidate errors/cameras |
| `create_lerobot_dataset.py` | Converts HDF5 states/actions and serialized task metadata to DOM features | Keep geometry/camera keys; propagate tracking provenance, commanded controls, split manifests and mode metrics |
| `utils/instruction_generator.py` | Only pick/place/long-horizon; ignores `instruction`; samples container descriptions | Handle `track` before template/container sampling, using its explicit instruction |
| `utils/datasets.py` | Defaults to modulo-1000 train/test selection | Pass our explicit `episodes` lists for all three splits; retain per-selected-episode statistics |

Our current exporter labels the measured pose from the same rollout at `t+5`. Upstream replay instead commands a future pose from a **source** rollout and captures observations in a **second** rollout. These are related action conventions, but the resulting observation/action pairs are not identical. Keep `direct-future-measured` and any future replay export version distinguishable in provenance; validate replay before mixing them. The current pilot need not be discarded or silently relabeled.

Use one primary task: hover 10 cm above the object's top-center, keep the gripper open, and select tilt from lateral velocity with the specified hysteresis. The three orientation modes are outputs of that task. Stationary, neutral, positive/negative lateral, and reversal paths are validation cases. A fixed-downward tracking task can be added later as an explicitly instructed baseline; do not label the existing velocity-conditioned demonstrations as that baseline.

## Apply the reader integration

From the collector directory, create a separate upstream checkout if it does not already exist. For an existing checkout, inspect its revision/local changes first; do not reset it to apply these patches.

```bash
git clone https://github.com/hzxie/DynamicVLA.git ../DynamicVLA
git -C ../DynamicVLA checkout --detach 434dc7105cf6d559018d15af5fecbcdad44f036e
git -C ../DynamicVLA apply --check ../Data_Collection_Dynamic_Objects/integrations/dynamicvla/track_instruction.patch
git -C ../DynamicVLA apply ../Data_Collection_Dynamic_Objects/integrations/dynamicvla/track_instruction.patch
```

The tracking branch returns the explicit instruction before attempting to sample the empty `containers` array. Existing manipulation templates retain their behavior. Missing/empty tracking instructions fail clearly.

Upstream imports `pylibmc` even with caching disabled. If it is unavailable, [optional_memcached.patch](optional_memcached.patch) makes that dependency optional for `pin_memory=False` disk reading, and raises an explicit error if caching is requested without it. The local reader check used this patch. Alternatively install pylibmc and its native libmemcached dependency in the isolated reader environment.

```bash
git -C ../DynamicVLA apply --check ../Data_Collection_Dynamic_Objects/integrations/dynamicvla/optional_memcached.patch
git -C ../DynamicVLA apply ../Data_Collection_Dynamic_Objects/integrations/dynamicvla/optional_memcached.patch
```

Use a separate reader environment with LeRobot 0.3.3 and datasets 3.6.0. Its TorchCodec build must match PyTorch and have FFmpeg support. Do not install training dependencies into Isaac Sim. Run:

```bash
env -u LEROBOT_HOME python integrations/dynamicvla/check_loader.py \
  --upstream ../DynamicVLA --dataset exports/pilot-v1 \
  --report exports/pilot-v1.dynamicvla-validation.json
```

`--instructions-only` checks the patched task parser and source pin without loading Torch/video dependencies. A full check loads the real `utils.datasets.LeRobotDataset`, verifies explicit episode membership, compares absolute/delta action chunks with Parquet values, and decodes synchronized observation histories. Reports must remain outside the immutable export. Run the collector's full `validate` separately; this sample check does not replace file completeness/checksum validation.

## Palmetto reader wiring

After downloading a pinned HF dataset revision, instantiate the upstream reader with the split manifest:

```python
import json
from pathlib import Path
from utils.datasets import LeRobotDataset

root = Path("/YOUR/PALMETTO/SCRATCH/dom-franka-tracking")
splits = json.loads((root / "meta/split_assignments.json").read_text())
info = json.loads((root / "meta/info.json").read_text())
train = LeRobotDataset(
    repo_id="YOUR_NAMESPACE/dom-franka-tracking", root=root,
    episodes=splits["train"], required_features=list(info["features"]),
    pin_memory=False, delta_action=True,
    delta_timestamps={"observation": [-2, -1, 0], "action": [0, 1, 2, 3, 4]},
)
train_stats = train.stats
```

The history/chunk offsets above are the compatibility-test values; match the actual model configuration when training. Upstream calls these offsets `delta_timestamps` but interprets them as frame offsets and divides by FPS internally. Delta actions subtract current XYZ/Euler coordinates from the six pose values, leaving gripper unchanged. Upstream subtraction does not wrap angular differences; preserve its convention unless the model and inference code are changed together.

The higher-level upstream `get_dataset()` factory does not expose `episodes` or `root`. The Palmetto training entry point must forward them or instantiate this reader directly. Merely setting `split="train"` would select 99 pilot episodes rather than our 80. The model training loop and checkpoint configuration have not been modified.

## Native generation acceptance milestone

Implement the table's tracking branches in an isolated upstream checkout, reusing our tested trajectory/controller/metrics modules. Verify the five smoke trajectories and camera synchronization, then export a small upstream-generated/replayed sample and run both validators. Only after comparing action conventions and passing those checks should this replace the current validated collector. Preserve held-out asset exclusions, atomic writes, bounded retries, balanced whole episodes, controller commands, and the existing pilot throughout migration. No custom USD is needed.

## Recovered Kinematic-Residual-VLA fork (October 2, 2026)

The destination repository already contains a modified `third_party/DynamicVLA` fork at repository commit `290e72e` before this collector was added. It supports explicit episode lists through its higher-level factory, unlike the upstream revision audited above. Do not reset that fork or apply the pinned-source checker to it: the checker deliberately requires the original audited upstream commit.

Use [evaluate_checkpoint.py](evaluate_checkpoint.py) for offline saved baseline/residual evaluation on this tracking dataset with the recovered fork. It uses generic LeRobot 0.3.3 for recorded samples and handles the explicit tracking instruction without editing the fork's instruction parser. See [the main README](../../README.md#evaluate-saved-checkpoints) for commands, checkpoint formats, action semantics, metrics and the separate existing DOM simulator workflow. No learned-policy hover rollout bridge is implemented.
