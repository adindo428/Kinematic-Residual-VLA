# Four-method tracking comparison: 75 epochs, 20-action chunks

This comparison adds **DOM-VLA+Velocity** to the three existing formulations. All models predict 20 Cartesian pose/gripper actions, execute/replan every 20 camera observations, and use the original 100-episode apple tracking dataset with its supplied 80/10/10 splits. The earlier 30-epoch/50-action comparison remains in `RESULTS/tracking_comparison_latest`.

| Key | Name/color | Neural velocity input | Analytic action displacement |
| --- | --- | --- | --- |
| baseline | DOM-VLA / blue | None; motion can be inferred from RGB history | None |
| velocity | DOM-VLA+Velocity / green | Current robot-base XYZ velocity, standardized using training episodes, in state-projection slots 6:9 | None; standard action target and flow integration |
| residual | KR-VLA-Field / orange | No dedicated velocity input in this existing checkpoint | Drift-adjusted target; inject XYZ displacement during integration |
| coordinate | KR-VLA-Coordinate / purple | No dedicated velocity input in this existing checkpoint | Residual coordinates during learning/sampling; restore displacement once |

The new baseline uses the existing 32-input state projection: normalized six-dimensional end-effector pose in slots 0:6, normalized current XYZ object velocity in slots 6:9, zeros elsewhere. It adds no parameters or visual tokens. All model weights are initialized from `hzxie/dynamic-vla-DOM`, not from a tracking-fine-tuned model. The original baseline checkpoint supplies only the already-fitted training-split normalization buffers. Velocity normalization is fitted on the 80 training episodes and saved in the config. Missing/nonfinite velocity or future-horizon velocity inputs are rejected.

## Matched training

- 75 epochs; full vision/text/connector/state/action fine-tuning.
- Batch size 8, 44 accumulation steps, effective batch 352; seed 42.
- AdamW: learning rate 1e-4, betas (0.9, 0.95), epsilon 1e-8, weight decay 1e-10.
- Cosine schedule, 1,000 warmup optimizer steps; same image augmentation.
- Entire validation split each epoch; lowest validation action-MSE checkpoint selected; test reserved for evaluation.
- One A100, 8 CPUs, 64 GB RAM, 24 hours. Two data workers avoid prior host-memory failures.
- Config: `configs/dynamicvla_dom_tracking_velocity75.yaml`.
- Launcher: `slurm/train_tracking_velocity75_a100.sbatch`.
- Checkpoints: `/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_velocity/checkpoints/tracking_dom75_full20_velocity_<training-job>/`.

The baseline and Field 75-epoch runs finished training. Field's original Slurm job failed in its post-training release script; `verify_tracking_training.py` verifies saved epoch/optimizer state, finite tensors, validation selection, strict reload and validation inference before writing a readiness marker. Coordinate is resuming through job 16666006 and writes to `/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_full20/resume_outputs/checkpoints/tracking_dom75_full20_coordinate_16623338`. Evaluation must depend on that resume job as well as the new baseline run.

## Submission

Run from `/home/aparame/Research/Dynamic_VLA/third_party/Kinematic-Residual-VLA/third_party/DynamicVLA`. The initial sequence is GPU smoke → training → evaluation; evaluation also waits for Coordinate's existing resume job. Exact submitted job IDs and dependencies are recorded in `verification/tracking_four_launch.json`.

```bash
sbatch slurm/smoke_tracking_velocity_a100.sbatch
# After the smoke passes:
sbatch slurm/train_tracking_velocity75_a100.sbatch
# Supply the new checkpoint and dependencies explicitly:
VELOCITY=/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_velocity/checkpoints/tracking_dom75_full20_velocity_TRAIN_JOB \
  sbatch --dependency=afterok:TRAIN_JOB:16666006 slurm/evaluate_tracking_four_l40.sbatch
```

The four-method evaluation requests one L40, 8 CPUs and 32 GB for 8 hours. All four methods run on the same GPU and simulator configuration.

## Outputs and evaluation

`RESULTS/tracking_comparison_four_75epoch/<evaluation-job>/` contains:

- `offline_imitation/`: `comparison.json`, `per_episode.csv`, four prediction NPZ files, `iid_xyz_error.png`, and interpretation README.
- `live_tracking/`: `summary.json`, `per_episode.csv`, `tracking_comparison.png`, exact manifest, per-model IID/1.5x/2x rollout folders, telemetry and three camera videos per trial.
- `live_tracking/comparison_videos/`: 30 four-panel videos, ordered **DOM-VLA, DOM-VLA+Velocity, KR-VLA-Field, KR-VLA-Coordinate**. Method names/colors remain visible after termination.
- `verification/`: real-data policy inference and 12 short live smoke trials before the full evaluation.
- `provenance/`: completion/selection/checkpoint-hash records for all four models.

The full comparison has 120 trials (four models × ten held-out trajectories × three speeds). OOD time-scales identical spatial paths: velocity is multiplied by 1.5/2 and acceleration by 2.25/4. Offline imitation is IID only; editing velocity while keeping recorded action labels would not be a valid offline OOD test.

**Velocity is not camera-estimated in this experiment.** Per the matched-comparison choice, training/offline evaluation use recorded current velocity and closed-loop evaluation uses current simulator velocity. All three velocity-using methods receive the same type of current robot-base XYZ signal; no future trajectory fields enter the policy. A vision-based estimator is separate future work. RGB images remain unmasked for every method.

The simulator pauses during neural inference, matching the earlier evaluation. Report actual inference latency without calling this a real-time deployment test. Mean hover errors use completed episodes only; completion/termination counts include every attempted trial. A completed episode is not tracking success, and no grasp-distance metric is reported as tracking success. Offline action MSE mixes Cartesian/Euler/gripper channels; XYZ imitation errors are also reported separately. The per-episode CSV and paired comparisons preserve failures and matched-episode counts.

Default jobs do not overwrite existing output directories or use older checkpoints as fallbacks. The frozen-backbone 30-epoch comparison and its plots/videos remain available separately.

## Current submission: RTX 6000 Ada (2026-10-06)

Per the user's sky02c suggestion, **job 16741015** runs checkpoint verification, offline inference, all four closed-loop IID/OOD policies and Isaac Sim rendering on one RTX 6000 Ada. Resources: `--gpus=rtx_6000:1`, 8 CPUs, 32 GB RAM, 8 hours. Slurm forbids phase constraints; the requested GPU type currently maps to the six online skyphase02c nodes. These nodes were fully allocated at submission, so the job may wait for capacity.

Launcher: `slurm/evaluate_tracking_four_rtx6000.sbatch`. It invokes the shared evaluation driver as Bash; the shared driver's L40 SBATCH header does not apply. CUDA device visibility is explicitly passed into the simulator container to preserve Slurm's allocation on these multi-GPU nodes.

All metrics, plots, and videos go to `RESULTS/tracking_comparison_four_75epoch/16741015/`. The previous L40 job 16740663 was cancelled before starting. No training was restarted; there are no training dependencies.

### Additional L40S evaluation

At the user’s request, **job 16741254** runs the same complete evaluation on one L40S, alongside RTX 6000 job 16741015. It was submitted with `--job-name=tracking-four-l40s --gpus=l40s:1` overriding the shared launcher’s header. Outputs are separate: `RESULTS/tracking_comparison_four_75epoch/16741254/`. Neither evaluation replaces or overwrites the other.
