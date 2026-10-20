# DOM-compatible Franka hover tracking

Collect locally in Isaac Sim, export LeRobot **v2.1**, upload to Hugging Face, and load the pinned dataset revision on Palmetto. This is a tracking extension using DOM assets, not an official release of DOM. No model training runs here.

The pilot uses one scene and one object, 100 accepted 10-second episodes, three RGB cameras at 480×360/25 FPS, and a 200 Hz rule-based expert. The gripper stays open. Lateral target velocity selects downward or ±45° tool orientation, with hysteresis. See [implementation.md](implementation.md) for exact conventions and limitations.

Published data: [adi2440/dom-franka-tracking](https://huggingface.co/datasets/adi2440/dom-franka-tracking), revision **`5c46a9f26b5f85edafa53885187cc03fa8706219`**. Git contains code, configuration, tests, integration patches, and small verification reports. USD assets, videos, datasets, checkpoints, environments, and runtime caches are excluded.

## What the pipeline does

1. **Audit assets:** inventory DOM scenes/objects and exclude all assets referenced by the supplied official benchmark configurations. Assemble a verified training-eligible scene with a Franka, one apple, and wrist/opposite/side cameras.
2. **Generate motion:** sample seeded smooth 3D paths with bounded position, velocity and acceleration. Hold object orientation fixed. The tracking reference is 10 cm above the object's top-center along world Z.
3. **Run the expert:** at 200 Hz, combine target-velocity feedforward with PD position feedback, then solve damped least-squares IK with speed, workspace and joint limits. Robot-base lateral velocity selects the tool orientation: positive → −45°, negative → +45°, near zero → downward. Entry/exit thresholds are 0.05/0.03 m/s, with hysteresis. Keep the gripper open.
4. **Record and accept episodes:** capture synchronized RGB/state at 25 Hz and 480×360 while retaining 200 Hz controller diagnostics. Reject episodes failing tracking, contact or limit gates. Select whole trajectories to balance the three modes; do not trim or duplicate frames to force balance.
5. **Export:** write DOM-shaped LeRobot v2.1 Parquet, AV1 videos, metadata, calibration, diagnostics and provenance. Each action is the measured tool pose five recorded frames ahead plus open gripper. Five continuation frames supply the final labels.
6. **Validate and publish:** check hashes, timing, cameras, labels, episode splits and reader compatibility; upload to Hugging Face and verify a downloaded sample. Download the exact revision for training/evaluation on Palmetto.

| Export feature | Shape / convention |
|---|---|
| `observation.state` | 6: robot-base tool XYZ + `xyz` Euler angles, meters/radians |
| `observation.environment_state` | 9: object XYZ + Euler angles + linear velocity |
| `action` | 7: future measured tool XYZ + Euler angles + gripper (+1 open) |
| `observation.images.{wrist_cam,opst_cam,side_cam}` | RGB video, 25 FPS, 480×360 |
| Timestamp and frame/episode/task indexes | Synchronized identifiers; task metadata has `task="track"` |

Observation-tool and IK-control offsets are different (0.1034 m and 0.107 m along hand Z). The implementation converts between them explicitly. Actual issued commands remain separate from future-pose training labels. This matters when interpreting policy predictions or building a closed-loop policy adapter.

## Setup

Run the examples from this directory. All YAML paths resolve relative to the YAML file. CLI data/output paths resolve relative to the current directory.

```bash
# From a clone of https://github.com/adindo428/Kinematic-Residual-VLA
cd third_party/Data_Collection_Dynamic_Objects
python -m venv .venv-tools
.venv-tools/bin/python -m pip install -e '.[test]'
.venv-tools/bin/python -m pytest -q
```

Simulation needs an NVIDIA GPU, Isaac Sim **5.1**, and a separate Isaac Lab **2.3.2** installation/environment. Follow the [Isaac Lab v2.3.2 installation instructions](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/index.html), referencing the installed Isaac Sim directory. Do not upgrade a shared training environment. Install this package into the collection interpreter:

```bash
# Replace with the separate 2.3.2 checkout, not an arbitrary existing installation.
export DOM_ISAACLAB=/path/to/IsaacLab-2.3.2
"$DOM_ISAACLAB/isaaclab.sh" -p -m pip install -e .
```

The discovered `/home/adi2440/IsaacLab` checkout is 2.3.0. `--allow-version-mismatch` permits inspection and disposable smoke runs on that checkout, but never training collection. A headless desktop still needs a functioning GPU driver. Cloud-referenced Franka assets must be reachable, or set `scene.robot_usd` to a complete local copy.

## Assets and scene preparation

Expected existing directories:

```text
third_party/
  DOM-3D-Scenes/           # *.usd plus the original texture directory
  DOM-3D-Objects/objects/ # category/object.usd and textures
  DOM-Test/              # tests/, scenes/, test-envs.txt
  Data_Collection_Dynamic_Objects/
```

```bash
.venv-tools/bin/python -m dom_tracking inspect-scene
"$DOM_ISAACLAB/isaaclab.sh" -p run.py inspect-scene --assemble
```

The first command verifies the benchmark inventory and writes `manifests/assets.json`. It excludes **every** object/scene referenced by any supplied test case, plus the `unseen` directory and non-target categories. This conservative policy excludes some otherwise valid training assets too. It does not recover the authors' full original training split. With the current downloads it finds 2,747 eligible scenes and 50 eligible target objects.

The second command checks USD dependencies, chooses a table edge in an eligible scene, and writes `manifests/layout.json`. Assets remain unchanged; no custom USD is required. Geometry checks do not certify physics reachability: smoke runs must pass before a pilot. View the scene with `--set scene.headless=false`. An assembled layout is always marked as awaiting physics validation.

## Smoke tests and collection

Use a separate output directory for each smoke trajectory. Stationary, positive, negative, neutral, and switch trajectories are available. Positive/negative tests use short episodes to stay within workspace bounds.

Run all five checks in one simulator session with `collect --smoke-suite --set collection.output=../runs/smoke-suite`. This forces disposable smoke mode, uses two-second episodes (ten seconds for switching), checks expected mode coverage, and writes `smoke_suite.json`.

```bash
"$DOM_ISAACLAB/isaaclab.sh" -p run.py collect --smoke-suite \
  --set collection.output=../runs/smoke-suite

"$DOM_ISAACLAB/isaaclab.sh" -p run.py collect \
  --set collection.purpose=smoke --set collection.episodes=1 \
  --set collection.seconds=2 --set collection.max_attempts=3 \
  --set trajectory.kind=stationary --set collection.output=../runs/smoke-stationary

# Repeat with positive, negative, neutral and distinct output paths.
# Use collection.seconds=10 for the switch test to allow settled tilt intervals.
# Inspect videos and rejection diagnostics before running the pilot.
"$DOM_ISAACLAB/isaaclab.sh" -p run.py collect
```

Default settings are in [configs/pilot.yaml](configs/pilot.yaml). Use repeated `--set group.key=value` flags for workspace, speed/acceleration, thresholds, gains, seed, episode count, camera dimensions, and raw log format. Example: `--set collection.raw_format=jsonl`. The DOM exporter requires 480×360, 25 FPS, and AV1 even though raw collection is configurable.

Interrupted runs resume with the same command, configuration, assets, and software versions. Never edit a run in place to change gains or thresholds; start a new output directory. Partial episodes are quarantined, rejected attempts retain diagnostics, and attempt seeds are not reused. Only one collector can own a run directory.

Inspect `runs/pilot/report.json` and each episode's metadata. Mode frequencies target 33.3% each within ±5 percentage points. Defaults require RMSE ≤2 cm, maximum error ≤5 cm, no contacts above 0.5 N on monitored bodies, no joint-limit violations, and ≤5° orientation error after a 0.75-second transition allowance. Frustum inclusion is checked for each camera; it is not an occlusion detector. Failed gates are reported and are never silently relaxed.

## Export and validation

```bash
.venv-tools/bin/python -m dom_tracking export-dom --run runs/pilot --output exports/pilot-v1
.venv-tools/bin/python -m dom_tracking validate --dataset exports/pilot-v1
```

Exports are immutable: choose a fresh destination for each export. For disposable loader tests, a smoke run can be exported with `--allow-smoke`; upload rejects such datasets.

Use a separate **LeRobot 0.3.3** environment for v2.1 loader validation. The currently installed 0.5.x loader targets a newer format; the collector refuses automatic migration.

```bash
python -m venv .venv-loader
.venv-loader/bin/python -m pip install -r requirements-loader.txt
.venv-loader/bin/python -m pip install --no-deps -e .
env -u LEROBOT_HOME .venv-loader/bin/python -m dom_tracking validate \
  --dataset exports/pilot-v1 --loader-check --report exports/pilot-v1.validation.json
```

The validator checks all file hashes, rows, action labels, frame counts, timestamps, camera streams, splits, and several actual LeRobot samples. Keep the validation report outside the dataset folder to preserve checksums.

The command removes the legacy `LEROBOT_HOME` variable for this process because LeRobot 0.3.3 rejects it. Use `HF_LEROBOT_HOME` if you need a custom cache location.

`action` is a **measured tool pose five frames (0.2 seconds) in the future**, plus open-gripper value +1. It is not the controller's issued velocity/joint command. Actual commands and full state are retained under `diagnostics/`. Five extra simulation frames supply final labels without duplicating or padding an episode's tail.

## Hugging Face upload

Authenticate using your existing Hugging Face credentials (`hf auth login` in a current tools environment). Tokens never belong in YAML or committed files. Use a dedicated dataset repository:

```bash
.venv-tools/bin/python -m dom_tracking upload \
  --dataset exports/pilot-v1 --repo-id YOUR_NAMESPACE/dom-franka-tracking \
  --validation-report exports/pilot-v1.validation.json
```

Repositories are private by default; `--public` explicitly selects public publication. A successful upload records an immutable commit and downloaded-sample checks in `exports/pilot-v1.upload.json`. The uploader checks every remote filename/size and re-downloads one episode's three videos, Parquet, and metadata. Retrying uses Hugging Face's content-addressed upload machinery. Do not reuse a repository containing unrelated files. Raw USD assets are not uploaded.

## Palmetto handoff

Use the repository ID and exact revision from the upload receipt:

```python
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "adi2440/dom-franka-tracking"
revision = "5c46a9f26b5f85edafa53885187cc03fa8706219"
root = Path("/YOUR/PALMETTO/SCRATCH/dom-franka-tracking")
snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision, local_dir=root,
                  ignore_patterns=[".gitattributes"])
splits = json.loads((root / "meta/split_assignments.json").read_text())
dataset = LeRobotDataset(repo_id=repo_id, root=root, revision=revision,
                        episodes=splits["train"], video_backend="pyav")
train_stats = dataset.stats  # statistics for the selected episodes in LeRobot 0.3.3
print(dataset[0]["action"].shape)  # (7,)
```

Use a v2.1-compatible training loader. `meta/info.json` lists all episodes as `train` for DOM format compatibility; **select the separate split manifest explicitly** to prevent validation/test overlap. The 100-episode pilot yields 80/10/10 disjoint episodes with shared asset and trajectory support.

Use statistics aggregated over the selected training episodes for normalization. `meta/stats.json` describes the entire export; `meta/episodes_stats.jsonl` provides the per-episode statistics needed to exclude validation/test data from normalization.

The upstream DynamicVLA reader has now been inspected at commit `434dc7105cf6d559018d15af5fecbcdad44f036e`. It requires a tracking instruction-parser patch and explicit episode selection. See [the integration audit, patches, and reader check](integrations/dynamicvla/README.md). Its actual reader passed 18 sample checks with camera histories and absolute/delta action chunks. The checkpoint evaluator below targets the recovered fork already present in this GitHub repository; the older pinned reader audit remains separate historical evidence.

## Evaluate saved checkpoints

There are three distinct checks. **Offline evaluation on our tracking recordings is provided below.** The rule-based smoke suite checks the collector itself. The existing DynamicVLA Isaac Sim server evaluates manipulation benchmarks; it has not yet been adapted to drive this hover-tracking environment with a checkpoint.

| Check | Command / entry point | What the result means |
|---|---|---|
| Collector smoke suite | `run.py collect --smoke-suite` | Expert/controller tracking errors and physical validity |
| Saved baseline/residual checkpoint on tracking data | `integrations/dynamicvla/evaluate_checkpoint.py` | Predicted action agreement with held-out recorded labels |
| Checkpoint in original DOM simulation | `../DynamicVLA/simulations/evaluate.py` + `scripts/inference.py` | Closed-loop pick/place benchmark performance |

### Checkpoint prerequisites

Use the **separate training/inference environment**, with this repository's existing `third_party/DynamicVLA` dependencies and LeRobot 0.3.3. On Palmetto, see [DynamicVLA/RECOVERY.md](../DynamicVLA/RECOVERY.md) and its environment setup. Do not install model-training dependencies into the Isaac Sim interpreter.

The evaluator accepts a local saved **DynamicVLA baseline or residual** directory containing `config.json` and `model.safetensors`. The saved config must specify six-value Euler tool states, seven-value pose/gripper actions, observation history, action chunk size, and `use_delta_action`. Residual settings and normalization buffers come from the checkpoint. Arbitrary architectures, quaternion-action checkpoints and isolated optimizer `.pt` files are not interchangeable with this contract.

If evaluating an epoch file such as `model.epoch015.safetensors`, put a copy in a fresh directory as `model.safetensors` alongside the matching run's `config.json`. Preserve the original checkpoint files. The evaluator records weight/config hashes so results identify the exact checkpoint. No checkpoint weights are included in this repository.

### Offline evaluation on held-out tracking episodes

Download the pinned data once. From the collector directory, in the inference environment:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="adi2440/dom-franka-tracking", repo_type="dataset",
    revision="5c46a9f26b5f85edafa53885187cc03fa8706219",
    local_dir="datasets/dom-franka-tracking",
    ignore_patterns=[".gitattributes"],
)
PY

# Validate the downloaded data with the collector tools environment.
.venv-tools/bin/python -m dom_tracking validate --dataset datasets/dom-franka-tracking

# Bounded checkpoint smoke check; python here is the inference interpreter.
env -u LEROBOT_HOME python integrations/dynamicvla/evaluate_checkpoint.py \
  --upstream ../DynamicVLA --checkpoint /path/to/saved_checkpoint \
  --dataset datasets/dom-franka-tracking --split validation \
  --max-frames 25 --device cuda --seed 42 \
  --output runs/evaluation/checkpoint-smoke.json

# Evaluate the full validation split for checkpoint selection.
env -u LEROBOT_HOME python integrations/dynamicvla/evaluate_checkpoint.py \
  --upstream ../DynamicVLA --checkpoint /path/to/saved_checkpoint \
  --dataset datasets/dom-franka-tracking --split validation \
  --device cuda --seed 42 --output runs/evaluation/checkpoint-validation.json

# After choosing a checkpoint, use --split test and a fresh output filename.
```

The script selects episodes from `meta/split_assignments.json`, reconstructs observation histories at the checkpoint's frame offsets, resizes camera inputs to checkpoint feature sizes, and calls `predict_action_chunk` with observations and the recorded tracking instruction. It does not pass ground-truth actions into the model. It keeps checkpoint normalization unchanged, converts delta predictions back to absolute pose exactly once, and excludes padded action steps at episode ends. No patch to the sibling DynamicVLA files is needed: this evaluator uses the generic LeRobot reader and handles tracking instruction metadata itself.

Reports contain position-label RMSE/max error in meters, geodesic orientation-label RMSE/max error in degrees, gripper-label RMSE, per-episode position errors, evaluated counts, split IDs, seeds, checkpoint/source hashes and software versions. `--max-frames` bounds a smoke run; `--stride` subsamples anchors deterministically. Keep both identical when comparing checkpoints. The default evaluates all frames, with each unpadded action-chunk element contributing to the metrics. Use the same seed for baseline/residual comparisons; GPU/software differences can still affect reproducibility.

These are **offline label errors**, not the collector's physical tracking RMSE, collision rate, or success rate. The model sees recorded expert observations rather than observations caused by its own actions. The evaluator's numerical, padding and input-isolation tests pass; no real saved checkpoint was available locally for a model-forward verification. Run the bounded check with your checkpoint/environment before a full evaluation.

The snapshot examples omit Hugging Face's repository-generated `.gitattributes`; it is not part of the collector's checksummed export. Keep the remaining dataset files intact so `validate` can verify the complete inventory.

### Existing Isaac Sim closed-loop checkpoint evaluation

For the original DOM manipulation benchmark, run two processes on the same machine. From this collector directory, first start the server using the Isaac Lab interpreter (the scene, objects and test JSON must already be downloaded):

```bash
"$DOM_ISAACLAB/isaaclab.sh" -p ../DynamicVLA/simulations/evaluate.py \
  --headless --enable_cameras --device cuda:0 --num_envs 1 \
  --scene_dir ../DOM-Test/scenes --object_dir ../DOM-3D-Objects/objects \
  --env_cfg ../DOM-Test/tests/1-1_place_franka_apple13d_O02_01048988_09b0.json \
  --output_dir runs/evaluation/dom-server --save -n 1 \
  --host 127.0.0.1 --img_port 3186 --act_port 3188
```

Then, in the separate inference environment, from the same directory:

```bash
python ../DynamicVLA/scripts/inference.py \
  -p /path/to/saved_checkpoint -a checkpoint-eval -r euler -d \
  --host 127.0.0.1 --img_port 3186 --act_port 3188 \
  -o runs/evaluation/dom-policy
```

The `-d` flag is for a delta-action checkpoint (`use_delta_action=true`); omit it for an absolute-action checkpoint. Keep Euler representation consistent with the checkpoint. Start with one test and non-streaming inference; preserve both server/client logs and result files. If using a test-list `.txt`, its entries must resolve from the server working directory; the direct JSON example avoids that path ambiguity. The existing Palmetto launcher is [evaluate_closed_loop_dom_l40.sbatch](../DynamicVLA/slurm/evaluate_closed_loop_dom_l40.sbatch); its cluster paths and checkpoint selection are specific to that recovered workflow.

**These commands run DOM placement evaluation, not hover tracking.** A policy-driven hover rollout still needs a bridge that supplies this collector's camera/state histories at 25 Hz, resets the policy between episodes, converts returned observation-tool poses to the control tool point, executes bounded IK at 200 Hz, and records the same tracking/contact/limit metrics for every trial without acceptance filtering. Adding `--task track` to the existing server or a checkpoint path to `collect` does not implement that bridge. See [the integration milestones](integrations/dynamicvla/README.md). No simulated learned-policy performance is claimed here.

## Local verification — September 30, 2026

The complete test suite passed in the isolated Python 3.11 collection environment: **21 passed**. The USD geometry test requires `pxr` and skips in a tools-only interpreter. The selected scene has 334 collision prims, including three for the supporting table; the selected apple has one.

All five simulator cases in [the expert smoke-suite report](verification/expert-smoke-suite.json) passed under Isaac Sim 5.1 / Isaac Lab 2.3.2:

| Case | Position RMSE | Maximum position error |
|---|---:|---:|
| Stationary | 0.013 mm | 0.013 mm |
| Positive lateral | 2.36 mm | 6.37 mm |
| Negative lateral | 2.65 mm | 6.52 mm |
| Neutral | 1.12 mm | 1.83 mm |
| Mode switching | 2.51 mm | 8.10 mm |

All had zero reported contact force, no joint/workspace violations, and 100% anchor-frustum inclusion in each camera. Maximum settled orientation error was below 0.58°. The 250-frame switch export in `exports/switch-v1` passed full validation and real LeRobot 0.3.3 row/video loading; see `exports/switch-v1.validation.json`. These smoke exports are deliberately blocked from upload.

Runtime checks used the isolated environment `/tmp/dom-tracking-sim-env` and Isaac Lab tag `v2.3.2`, commit `37ddf626871758333d6ed89cf64ad702aef127d0`, under `/tmp/dom-tracking-IsaacLab-2.3.2`. These temporary paths are verification artifacts; follow Setup for a persistent installation. Existing shared simulation/training environments were not upgraded.

The pilot completed **100 accepted ten-second episodes with zero rejections**: 25,000 synchronized observations, three cameras, and five continuation observations per episode for labels. Aggregate position RMSE is **3.023 mm**; maximum error is **31.851 mm**. Recorded mode frequencies for −45° / 0° / +45° are **33.344% / 33.316% / 33.340%**. Maximum settled orientation error is 0.738°, maximum monitored contact force is zero, and no joint/workspace violations occurred. All target anchors stayed inside all three camera frusta. See [the pilot report](verification/pilot-report.json). The local tracking plot is generated under `runs/pilot/controller_preview.png` and is excluded from Git.

The local complete export is `exports/pilot-v1` (download it from the Hugging Face link below), approximately 457 MiB, with 300 AV1 videos and 80/10/10 episode assignments. **Full validation and the LeRobot 0.3.3 loader check passed**; see [the dataset validation report](verification/dataset-validation.json). This checks every file checksum, every video frame, calibration/timing, and every future-pose label, then loads actual LeRobot samples. The dataset is published publicly at [adi2440/dom-franka-tracking](https://huggingface.co/datasets/adi2440/dom-franka-tracking), revision `5c46a9f26b5f85edafa53885187cc03fa8706219`. All 711 expected remote files were checked for presence/size; downloaded episode-0 metadata, Parquet, and all three videos matched local checksums, and the videos decoded. Anonymous repository access was verified. See [the upload receipt](verification/huggingface-upload.json).

Earlier recordings under `runs/calibration-debug` have stale camera metadata and are explicitly invalid for training. Other diagnostic/smoke directories retain failed attempts for inspection. Scene validation evidence is recorded in `manifests/physics_validation.json`.

The exact source used for collection is preserved under `runs/pilot/collector_source`, matching its recorded source hash. After collection, video-statistics calculation was accelerated using exact uint8 histograms; a regression test checks its results against all decoded pixels. This changes postprocessing performance without changing collected records or action labels.

The September 30 upstream audit used `/tmp/dom-upstream-audit`. The October 2 GitHub publication includes this folder alongside the destination repository's existing recovered `third_party/DynamicVLA` fork; that fork was preserved unchanged. The patched upstream reader check is recorded in [the upstream reader report](verification/upstream-reader-validation.json). Native upstream `--task track` generation/replay and training entry-point wiring remain pending. Hugging Face publication and revision verification are complete; no Palmetto training run has been performed.

## Sources

- [DOM dataset and schema](https://huggingface.co/datasets/hzxie/DOM)
- [DynamicVLA implementation](https://github.com/hzxie/DynamicVLA)
- [LeRobot v0.3.3 / dataset v2.1](https://github.com/huggingface/lerobot/tree/v0.3.3)

Keep source attribution and asset license information with any redistribution. This project does not assign a new license to the downloaded assets.
