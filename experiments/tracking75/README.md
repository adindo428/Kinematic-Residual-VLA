# Four-model tracking evaluation

This source snapshot reproduces the 75-epoch hover tracking comparison: DOM-VLA, DOM-VLA+Velocity, KR-VLA-Coordinate, and KR-VLA-Coordinate+Velocity. It includes the frozen simulator collector, strict checkpoint loader, current scoring, plotting/video helpers and local launcher. Historical Field code is retained for research history; the current nine-dimension comparison uses the four listed models.

**This is a tracking adaptation, not the official DOM manipulation benchmark.** For the public manipulation baseline use [the official DOM guide](../../evaluation/official_dom/README.md).

## Dependencies

Follow [the two-environment setup](../../evaluation/official_dom/README.md#1-install-two-environments). Policy: Python 3.10, Torch 2.7.1, LeRobot 0.3.3, Transformers 5.2.0. Simulator: Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.2. Install these additional dependencies in the **simulation** environment:

```bash
"$ISAACLAB" -p -m pip install \
  av==16.0.1 pyarrow==19.0.1 Pillow==11.3.0 PyYAML==6.0.2 pyzmq==26.4.0 scipy
```

`ISAACLAB` must select the simulation environment; `POLICY_PYTHON` must be the policy environment's absolute Python path. The collector enforces Isaac Lab 2.3.2. Run the commands from the repository root.

## Copy trained checkpoints

Weights are not committed to GitHub. Each checkpoint directory needs **config.json**, **model.best_validation_action_mse.safetensors**, and **training_ready.json**. Preserve all three bytes exactly. The marker verifies completion of 75 epochs and config/weight hashes; an incomplete or altered checkpoint is refused. Copy whole checkpoint directories to the lab machine, for example:

```bash
export CHECKPOINTS="$HOME/krvla-evaluation/checkpoints"
mkdir -p "$CHECKPOINTS"
scp -r aparame@slogin.palmetto.clemson.edu:/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_full20/checkpoints/tracking_dom75_full20_baseline_16623336 "$CHECKPOINTS/baseline"
scp -r aparame@slogin.palmetto.clemson.edu:/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_velocity/checkpoints/tracking_dom75_full20_velocity_16680047 "$CHECKPOINTS/velocity"
scp -r aparame@slogin.palmetto.clemson.edu:/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_full20/resume_outputs/checkpoints/tracking_dom75_full20_coordinate_16623338 "$CHECKPOINTS/coordinate"
scp -r aparame@slogin.palmetto.clemson.edu:/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_coordinate_velocity/checkpoints/tracking_dom75_full20_coordinate_velocity_16748131 "$CHECKPOINTS/coordinate_velocity"
```

These commands may require Clemson SSH/Duo authentication. Do not rename weight files or regenerate markers to bypass verification. Updated DOM100 checkpoints have a different training/checkpoint contract; use their evaluator instead.

## Assets and preparation

Download the [DOM scene assets](https://gateway.infinitescript.com/?f=DOM-3D-Scenes) and [DOM object assets](https://gateway.infinitescript.com/?f=DOM-3D-Objects). Preserve complete textures/dependencies. The frozen room is `00004f89-9aa5-43c2-ae3c-129586be8aaa.usd`; the original target is `objects/apple/apple01.usd`. VG also uses `apple02`–`apple05` and `objects/unseen/apple99.usd`. A missing file is an asset setup issue, not a failed policy trial.

```bash
export KRVLA_REPO="$PWD"
export TRACK_RUNNER="$KRVLA_REPO/experiments/tracking75/local.py"
export TRACK_RESULTS="$HOME/krvla-evaluation/results/tracking-local-01"
export SCENES="$HOME/krvla-evaluation/assets/scenes"
export OBJECTS="$HOME/krvla-evaluation/assets/objects"
"$POLICY_PYTHON" "$TRACK_RUNNER" prepare \
  --root "$TRACK_RESULTS" --scenes "$SCENES" --objects "$OBJECTS" \
  --baseline "$CHECKPOINTS/baseline" --velocity "$CHECKPOINTS/velocity" \
  --coordinate "$CHECKPOINTS/coordinate" --coordinate-velocity "$CHECKPOINTS/coordinate_velocity"
```

Preparation verifies the frozen room/target hashes, each model's declared architecture and checkpoint completion, then creates 90 matched shards. `templates/tracking_setup.json` carries the original camera, robot, room layout and controller parameters. The local wrapper replaces paths; it does not change trajectories or scoring. If the frozen room is not part of the smaller DOM test asset bundle, use the full scene archive. An optional `--robot-usd` selects an existing local Franka mirror.

## Start with one pilot shard

```bash
"$POLICY_PYTHON" "$TRACK_RUNNER" run --root "$TRACK_RESULTS" \
  --isaaclab "$ISAACLAB" --policy-python "$POLICY_PYTHON" --mode pilot --shard 0 --dry-run
"$POLICY_PYTHON" "$TRACK_RUNNER" run --root "$TRACK_RESULTS" \
  --isaaclab "$ISAACLAB" --policy-python "$POLICY_PYTHON" --mode pilot --shard 0
```

Shard 0 runs one CR trial per model. Inspect its logs, three-view videos and `verification.json` before the rest. Models are loaded sequentially, not simultaneously. On the 3080, monitor GPU memory with `nvidia-smi` while cameras and policy run together; local GPU capacity has not been measured here.

Run all 13 pilot shards, skipping the already verified one:

```bash
"$POLICY_PYTHON" "$TRACK_RUNNER" run --root "$TRACK_RESULTS" \
  --isaaclab "$ISAACLAB" --policy-python "$POLICY_PYTHON" --mode pilot --resume-verified
```

The pilot covers all nine dimensions and all five VG objects (52 rollouts). Full evaluation requires `pilot_verified.json`.

## Full evaluation

```bash
"$POLICY_PYTHON" "$TRACK_RUNNER" run --root "$TRACK_RESULTS" \
  --isaaclab "$ISAACLAB" --policy-python "$POLICY_PYTHON" --mode full
```

The launcher runs serially on one GPU. A shard is one dimension/setup with 20 episodes per model: 80 rollouts. Ninety shards yield 7,200 total rollouts, 1,800 per model. Add `--dimension CR` for the ten CR setups, or `--shard 0` for a single full shard. `--resume-verified` rechecks and skips complete shards; partial outputs are refused. Use a fresh root after a failed partial rollout, keeping the previous logs as evidence. Default RPC port is 27000; `--port` changes it. Ctrl-C cleans up the simulator and policy worker.

Results are under `pilot/000/` and `full/000/` through `full/089/`: manifests, logs, raw 200 Hz telemetry, chunks, initial states, per-episode metrics and videos. Full runs record trial 0 of each setup; pilot episodes all record video. After all full shards verify, the launcher writes `tracking_table1.csv`, `.md`, `.png` and `summary.json`. You can regenerate them with:

```bash
"$POLICY_PYTHON" "$KRVLA_REPO/experiments/tracking75/source/scripts/summarize_benchmark.py" \
  --root "$TRACK_RESULTS"
```

## Protocol and metrics

Success (%) is the arithmetic mean of each episode's time percentage with 3D EE-to-hover-target distance ≤3 cm. It includes the initial second. Early termination's remaining requested time earns zero credit. This is occupancy, not binary episode success. Completion is a separate column. Time is first entry within 3 cm, capped at the episode horizon if never acquired; warmup often makes this zero, so do not call it task completion time.

The hover target is 10 cm above the apple's visual top. Both policies start after the same three-second expert warmup, then receive no expert assistance. Two RGB views and language plus measured EE pose are neural inputs. Velocity models also receive current object velocity; Coordinate models use current velocity for analytic drift. The neural policy receives no exact target pose or future target path. Inputs use current simulator velocity, not a learned visual velocity estimator.

RGB is 25 Hz, physics/scoring 200 Hz, prediction labels begin at +0.2 s, action spacing is 0.04 s, and replanning is every 20 RGB frames (0.8 s). Simulation pauses during policy inference; streaming is disabled. This differs from the official manipulation streaming protocol.

| Dimension | Tracking condition |
|---|---|
| CR | Lateral direction reversals on a four-second sinusoid |
| DA | Three-axis movement with changing speed |
| LS | 30-second continuous tracking; others use ten seconds |
| VU | Exposure/contrast perturbations and a peripheral image mask |
| SR | Wider target workspace locations |
| MP | Brief RGB freezes while current state/velocity remain available |
| VG | Five held-out apple geometries in the training room |
| MG | 1.5–2× trajectory time scaling |
| DR | Scripted displacement disturbance proxy |

Ten setups vary seeded workspace placement/trajectory parameters in **one room**. They are not ten different scenes, and LS does not evaluate pick/place sequencing. For matching full DOM manipulation metrics, use the official evaluator.

The copied source manifests are archival evidence from Palmetto. Local path wrappers, benchmark asset-path parameterization and documentation were added for this publication; those historical hashes are not the hash of the repackaged tree. Newly prepared protocols hash the actual checkpoint/assets used for execution.

## Training and historical tools

See [training instructions](../../docs/TRAINING.md). The source's `scripts/` also retains offline evaluation, previous comparison plots/video merge tools, and data preparation utilities. `palmetto_reference/` and source `slurm/` are cluster references with historical paths; `local.py` is the workstation entry point. Existing comparison reuse scripts expect previously saved results and should not be treated as fresh benchmark launchers.
