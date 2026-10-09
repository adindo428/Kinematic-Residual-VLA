# Research handoff: October 9, 2026

Read this at the start of the first VS Code coding conversation on the Linux workstation. The immediate goal is to establish correct DOM manipulation inference and diagnose the failed grasps, then continue the KRVLA experiments. An RTX PRO 6000 Blackwell is expected; verify the exact card with preflight.

## Start-up order

Use [the workstation guide](README.md). Set local storage paths, install separate policy Python 3.10 and simulator Python 3.11 environments, run CUDA/decoder and simulator probes, download the pinned public checkpoint/assets, then run the nine-trial DOM pilot. Investigate physical correctness before launching the full suite. The current environment declaration is Sim 5.1 / Lab 2.3.2; the paper's reference was Sim 4.5 / Lab 2.2.1 with separate A6000 GPUs.

## Findings established on Palmetto

The SIF's native running API and version files confirmed **Isaac Sim 5.1.0-rc.19** and Lab version file **2.3.2**. The current GitHub README changed to that stack after the paper evaluation. Author clarification: [issue 29](https://github.com/hzxie/DynamicVLA/issues/29). No evidence establishes that retraining the released checkpoint is necessary yet.

Original full manipulation evaluation jobs 16868047/16868048 were stopped to diagnose failures. Completed CR and DA each had **0/200 official successes**. They are preserved. Do not report an unfinished nine-dimension score as a completed benchmark.

Isolated three-case, two-repeat diagnostics:

| Configuration | Job | Official completions |
|---|---|---:|
| Original, shared L40S | 16894064 | 0/6 |
| Corrected gripper, shared L40S | 16894064 | 0/6 |
| Corrected gripper and aligned queue, shared L40S | 16894064 | 0/6 |
| Corrected gripper and aligned queue, two L40S GPUs | 16894531 | 0/6 |

Both evaluation jobs completed with exit 0 and no infrastructure failures. The dual run and its 0/6 result were checked again while preparing this handoff. Client-selected action indices matched received observation indices in all arms. That does not establish zero observation-to-inference latency or zero server application lag. Small samples cannot isolate the entire cause.

The gripper correction changed finger velocity limit from 0.05 to 0.2 m/s and set static/dynamic/viscous friction to zero for the 5.1 calibration proposed in [PR 30](https://github.com/hzxie/DynamicVLA/pull/30). Observed summed finger-gap closing speed increased roughly 0.13 to 0.42 m/s. Effort, stiffness, damping, physics timestep, weights and official success definition stayed fixed. Queue alignment only executes the current indexed command and discards expired commands; model-side latency scaling/sleep remained enabled. Both corrections failed to restore official completion in these diagnostic samples.

One original apple trial actually grasped and lifted the object about 18 cm, then failed to complete. Official placement requires the object to be placed **and** the EE to reach its finishing pose within 1.5 cm (typically `[0.3, 0, 0.3]` in robot-relative coordinates). Do not characterize every failure as an empty grasp. Separate missed contact, unstable hold, placement and finishing-pose failure in the next analysis. Keep success scoring unchanged for paper comparisons.

Next investigations should measure object/finger geometry at closure, action coordinate/delta conventions, normalization, action application timestamps, rendering and physical differences from the reference runtime. A faster GPU is a useful timing control, but the completed dual-GPU failures argue against treating GPU sharing as the complete explanation. Do not introduce continuity constraints or retraining just to mask a simulator/inference mismatch.

Results on Palmetto:
`/home/aparame/Research/Dynamic_VLA/RESULTS/official_dom_diagnostic/20261009`

[Open diagnostic results](https://ondemand.rcd.clemson.edu/pun/sys/dashboard/files/fs//home/aparame/Research/Dynamic_VLA/RESULTS/official_dom_diagnostic/20261009)

Frozen source and drivers:
`/scratch/aparame/Dynamic_VLA/runs/official_dom_diagnostic_20261009`

Results may be home symlinks into scratch after the completed archive jobs 16894612/16894613. Durable results archive: private [Hugging Face dataset](https://huggingface.co/datasets/adi2440/krvla-evaluation-results). Check individual archive receipts before assuming a particular file is backed up. Scratch purges after 30 days without access; it is not durable storage.

## Source/model contracts

- `evaluation/official_dom/scripts/local.py`: pinned upstream manipulation evaluator, public checkpoint, no retraining. Source commit `434dc7105cf6d559018d15af5fecbcdad44f036e`; model revision `c9609d94c5940eb42436a31927d7e80ae454279e`; weight SHA256 `e3da8b1e0326d7a3ee6873f7fa17cdf7015201251901be5107377cb851ce3af8`.
- `workstation/diagnostic.py`: portable original/gripper/gripper_aligned diagnostic, separate source/results and source/asset hash checks.
- `experiments/tracking75/source`: frozen code for the trained 75-epoch tracking models. Models are DOM-VLA, DOM-VLA+Velocity, KRVLA-Coordinate and KRVLA-Coordinate+Velocity. Use this source to load those checkpoints.
- `third_party/DynamicVLA`: ongoing DOM100 training development; six variants (`dom100`, `kr100`, `domv100`, `krv100`, `krp100`, `krvp100`). Consult DOM100 docs for checkpoint/training contracts. Do not substitute it for the frozen official baseline or tracking models.

The current Coordinate+Velocity neural residual is conditioned on object velocity. Residual independence from velocity is not assumed. Common inputs are RGB, language and measured robot/EE data; velocity variants also receive current object velocity. Exact target pose and future trajectory are not neural inputs.

Tracking adaptation success is **mean per-episode time occupancy within 3 cm of the moving hover target**, including the first second and counting any unexecuted requested time as zero. This is not binary manipulation completion. Tracking hover is 10 cm above the apple's visual top, with three seconds expert warmup, 25 Hz RGB and 200 Hz physics. Simulation pauses during inference and streaming is disabled. The Table 1 stress adaptation uses one room with ten seeded setups, not the original nine manipulation dimensions.

Tracking full array 16830026 and summary 16830027 write to:
`/home/aparame/Research/Dynamic_VLA/RESULTS/tracking_table1_75epoch/20261008`
Their current status was not refreshed for this handoff. Do not cancel these jobs to free diagnostic GPUs. Coordinate+Velocity comparison lives in `RESULTS/tracking_comparison_four_75epoch/Coordinate_Conditioned_Velocity/16787490`. The four checkpoint transfer commands are in the tracking README.

## Continuing across machines

Published repository: [Kinematic-Residual-VLA](https://github.com/adindo428/Kinematic-Residual-VLA). Start the new conversation with: “Read workstation/HANDOFF.md and workstation/README.md; continue the DOM grasp diagnostics on this Linux RTX PRO 6000 machine.” Then inspect the actual environment and current Git status. Preserve local edits and previous results. Keep code in Git and machine-specific paths in ignored `workstation/env.local.sh`; weights/assets/results stay outside Git.

This handoff contains the experiment context needed for continuation. It does not transfer an active chat or running process. Inspect Palmetto statuses when needed; job states can change after the timestamp above.
