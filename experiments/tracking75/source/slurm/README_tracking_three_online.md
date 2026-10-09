# Three-model live closed-loop tracking evaluation

Run from `/home/aparame/Research/Dynamic_VLA/third_party/Kinematic-Residual-VLA/third_party/DynamicVLA`:

```bash
sbatch slurm/evaluate_tracking_three_online_l40.sbatch
```

This requests one L40, eight CPUs, 32 GB RAM and eight hours. It first checks checkpoints/assets/splits, exercises inference on real recorded observations, runs nine two-second live smoke trials, then evaluates all 90 full trials: three methods × ten held-out trajectories × speeds 1×, 1.5× and 2×. A physical policy termination counts as a failure and does not censor the remaining trials. Infrastructure or mismatched-start failures stop the job.

## Methods and checkpoint selection

| Key / display name | Formulation |
| --- | --- |
| `baseline` / **DOM-VLA** (blue) | Standard neural action flow; no analytic object-velocity drift. |
| `residual` / **KR-VLA-Field** (orange) | Case 1: train in raw action coordinates with a drift-adjusted target; apply analytic drift during reverse-flow integration. |
| `coordinate` / **KR-VLA-Coordinate** (purple) | Train/sample in residual action coordinates; restore analytic drift once after integration. |

All actions are Cartesian end-effector pose plus gripper. A common damped-IK servo converts poses to joint commands. Drift affects XYZ only, uses the checkpoint's training normalization/centering, and keeps the zero-based `k × 0.04 s` grid. Labels and execution retain the measured-pose +0.2-second horizon. The evaluator uses overhead/wrist observations at offsets `[-2, 0]` and replans every 20 frames. Both 20- and 50-action prediction chunks are supported.

Defaults explicitly point to the new 75-epoch runs, not the older `tracking_current` alias:

```text
/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_full20/checkpoints/
  tracking_dom75_full20_baseline_16623336
  tracking_dom75_full20_residual_16623337
  tracking_dom75_full20_coordinate_16623338
```

Each must contain `config.json`, `model.best_validation_action_mse.safetensors`, and a matching `training_ready.json` confirming 75 epochs. The loader strictly reloads all weights and normalizers. It never substitutes `model.safetensors` or another method's checkpoint. Full evaluation is not automatically submitted by restoring these files.

At restoration time baseline was ready, Field was still running, and Coordinate had no completion marker. Its training job had failed, and the previous checkpoint-promotion job was cancelled. Finish/verify the training set before running the default full comparison; this restoration does not restart training jobs.

For an explicitly selected older completed checkpoint set, set `BASELINE`, `RESIDUAL`, `COORDINATE` and `REQUIRE_COMPLETE_TRAINING=0`. This disables only the 75-epoch completion-marker requirement; validation-selected weights and method/formulation checks remain mandatory. Do not use a moving, still-training checkpoint for a reported benchmark.

```bash
# Read-only readiness/asset/split check, no GPU allocation:
MODE=preflight RUN_ID=manual_check bash slurm/evaluate_tracking_three_online_l40.sbatch

# Only nine short live trials, after selecting the desired checkpoint set:
MODE=smoke sbatch slurm/evaluate_tracking_three_online_l40.sbatch

# Optional overrides for custom checkpoint folders:
BASELINE=/path/to/baseline RESIDUAL=/path/to/field COORDINATE=/path/to/coordinate \
  REQUIRE_COMPLETE_TRAINING=0 sbatch slurm/evaluate_tracking_three_online_l40.sbatch
```

## Scenario and output contract

This restores the comparison on the original 100-episode apple tracking dataset and its test trajectories. It does not switch to the newly collected five-object paired-mode dataset. The original scene and object hashes are checked against dataset provenance. OOD traverses the same spatial paths faster: velocity scales by 1.5/2 and acceleration by 2.25/4. Reported telemetry includes the fraction actually exceeding the training maximum speed.

The simulator pauses during inference. Residual drift uses the current simulator-measured object velocity in robot-base coordinates; it never receives future object positions or action labels. This is a simulator-state velocity source, not a vision velocity estimator or a latency-aware real-robot test.

Default output root: `/home/aparame/Research/Dynamic_VLA/RESULTS/tracking_current_eval/`:

- `policy_smoke/<job>/`: checkpoint hashes, finite-inference checks and predicted shapes.
- `online_smoke/<job>/`: nine short simulated trials and three comparison videos.
- `online/<job>/`: full comparison, manifest, JSON/CSV metrics, plots and videos.

A direct `MODE=smoke` run uses `smoke/<job>/`. Set `OUTPUT_ROOT` to choose a different parent. Existing run directories are never overwritten.

Each rollout records hover RMSE/p95/max, orientation error, completion/termination, contact and joint/workspace violations, observed speeds, inference latency, per-step telemetry, predicted chunks and three camera videos. Completed-only mean errors are explicitly accompanied by completion counts; low error in a truncated rollout is not a full-horizon result. The legacy `success` field denotes the configured strict hover/orientation/safety thresholds, not grasp-distance success.

`comparison_videos/` shows **left DOM-VLA, middle KR-VLA-Field, right KR-VLA-Coordinate**, with persistent method/scenario labels and termination cards. `tracking_comparison.png` labels accuracy and completion separately. Every generated run includes a README explaining the methods and limitations.

## Code map and restoration

- `slurm/evaluate_tracking_three_online_l40.sbatch`: public entry point and staged checks.
- `slurm/evaluate_tracking_case1_online_l40.sbatch`: shared runtime launcher (now accepts all three methods).
- `scripts/prepare_tracking_online.py`: checkpoint, original-asset and held-out-trajectory manifest.
- `scripts/tracking_checkpoint.py`: validation-selected checkpoint loading and readiness verification.
- `scripts/tracking_online_policy.py`: live observation history, residual velocity inputs, seeded sampling and policy RPC.
- `scripts/tracking_online_sim.py`: Isaac Sim rollouts, termination handling, telemetry and camera recording.
- `scripts/tracking_online_common.py`: +0.2-second pose interpolation, shared IK servo and metrics.
- `scripts/summarize_tracking_online_three.py`, `scripts/tracking_visuals.py`: comparisons, plots and labeled videos.

The original simulator and policy implementation were recovered from the saved session's source writes and edits. The restored simulator, model configuration, model implementation and residual-flow helper match the earlier recorded source hashes. Necessary restoration changes update checkout paths, support 20-action checkpoints, isolate inference from missing training utilities, use strict validation-selected loading, and check completion before default evaluation. `verification/tracking_evaluator_restoration.json` records tests, source hashes and the short GPU verification job. Its mixed-generation smoke results are code verification, not the new matched scientific comparison.
