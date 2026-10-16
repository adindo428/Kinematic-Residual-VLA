# Research handoff: October 10, 2026

## Current SSD workstation baseline

The local machine has **two RTX A6000 48 GB GPUs**, Ubuntu 22.04, and NVIDIA
driver 550.163.01. The active profile is **Isaac Sim 4.5.0 / Isaac Lab 2.2.1**,
with both Python environments and all project data under the SSD checkout's
`scratch/dom45`. Use `workstation/dom-paper45.sh` with the ignored
`workstation/env.local.sh`. GPU 0 runs simulation; GPU 1 runs policy inference.
The unchanged upstream evaluator is pinned to
`a27a06d2ca74d0e987a5e552e01013073e93cfd8`, before the 5.1 migration.

The released checkpoint, CUDA/video decoding, native simulator versions, and
all official test configurations passed local checks. DOM-Test, DOM-3D-Objects,
and DOM-3D-Scenes are fully downloaded and extracted; the complete scene archive
is kept separately from the immutable evaluation scene subset. The official
robot/ground/marker USD dependencies are mirrored locally. Lab 2.2.1's rendering
experience selects NVIDIA's 5.0 asset root even with Sim 4.5; the mirror preserves
those original asset files and relative references.

The pilot verified **9 trials, 3 successes, 0 infrastructure failures**. The
full **1,800-trial** evaluation was launched as user systemd service
`krvla-dom-paper45-20261010.service`. Results are in
`scratch/dom45/results/dom-paper45-a6000-20261010-02`; inspect live service status,
`suite_status.json`, and per-dimension trial records before claiming completion.
Full comparison CSV/Markdown/PNG are generated only after every trial verifies.
The pilot percentage is not the Table 1 score. A startup attempt completed zero
full trials because a pilot simulator child retained the port; its logs are
preserved in `failed-startups`, and launcher/process-group cleanup was corrected
before restarting. Keep the machine on and the login session active during the
run. The following sections preserve the earlier Palmetto/Sim 5.1 investigation.

### Queued full-DOM KRVLA + Velocity training

The user authorized all DOM episodes on SSD and `krv100` training after baseline
evaluation. On October 11 they excluded side-camera videos, matching the
released baseline's wrist/opposite inputs. Required coverage is 621,926 files /
193.74 GB, including all metadata/parquets and both views. Previously downloaded
side-camera files remain on SSD. `krvla-dom-full-download-20261010.service`
downloads only missing required files at revision
`bbf6baa15f7253746d271a6e958e037afa537b1e`, with per-file hashes and resumable
SQLite inventory. Pinned resolver GETs avoid extra HEAD requests. It handles Hub
rate limits; inspect actual `download_status.json` before claiming completion.
`krvla-krv100-after-eval-20261010.service` is queued, waiting for baseline/data
completion and the user's own W&B login. The initial PC account was
`omkarmayekar555`; the user chose to log in with their own account locally via
`bash workstation/wandb-ssd.sh login`. Credentials go to the SSD's ignored
`config/wandb/netrc`. Do not print or commit the key.

The Palmetto import completed from the user's partial cache. HF fills missing
wrist/opposite files. The user authorized returning completed data to
`/scratch/aparame/.cache/huggingface/lerobot/hzxie/DOM` through Clemson's transfer
node. `krvla-dom-palmetto-backfill-20261011.service` waits for local completion,
copies missing/wrong-sized required files, checks every remote pinned hash,
repairs mismatches, and records verified completion. No side-camera download or
upload is queued. Inspect `training/dom-full/palmetto_backfill_status.json` and
`logs/dom-palmetto-backfill.log` under `scratch/dom45`. After the upload completed,
the user requested local profiling proceed alongside remote checksum verification.
That remote stage is no longer a local profiling/training prerequisite.

Use `bash workstation/krv100-ssd.sh` for progress. The source snapshot, request,
future family/checkpoints, and logs are under
`scratch/dom45/training/krv100-full-20261010`. The existing recipe has LR `3e-5`,
batch 352, horizon 20, BF16, seed 42, uniform frames, a 100-epoch cap, and the
documented validation early stopping. The local DDP entry point admits two
A6000s but still requires runtime-specific decoded backward/optimizer/tail,
inference, checkpoint/RNG and 15% headroom checks. GPU readiness and production
wait until the baseline finishes. The coordinator verifies W&B write access,
clones a separate SSD training environment, and freezes full-data splits/stats
before mandatory two-GPU throughput profiling and those checks. The user asked
to maximize GPU usage before training: `workstation/profile_krv100.py` sweeps
microbatch/loader/runtime settings, repeats the three fastest candidates, selects
confirmed samples/second with 15% headroom, and captures separate CPU/CUDA rank
traces. Trial optimizer updates are discarded; effective batch 352 and the
scientific recipe remain fixed. The production launcher must match the frozen
`profiling/selected_runtime.json` approval. Inspect `profiling/comparison.csv`
and `profiling/profiling_report.md` for measured results after the stage runs.
No production training or local GPU throughput claim has been established yet.
A preliminary timing report will use warm repeated measurements, excluding
validation/checkpoint I/O; full-dataset training may be lengthy.

### October 11 profiling incident

Local preparation completed and Palmetto checksum verification completed.
Profiling stopped after recording 18 trials: rank 0's video decoder failed on
`videos/chunk-195/observation.images.wrist_cam/episode_195417.mp4`; rank 1 then
timed out in NCCL. Six required-view files are published empty 48-byte MP4s,
despite matching the pinned checksums. Opposite episodes: 194633, 195603. Wrist
episodes: 194760, 195154, 195405, 195417. All are in the training split, accounting
for 662 frames. Fresh pinned HF retrieval returns the same empty failing file.
See `diagnostics/published-empty-videos.json` under the SSD training root.
The user approved excluding all six episodes, retaining 207,300 episodes /
23,551,555 frames. `approved_exclusions.json` records exact IDs/file hashes; the
coordinator requires these six exclusions with no additional omissions. Raw
SSD/Palmetto caches remain intact. Old request/source/family/profile/stage logs
are archived under `archives/20261011T205207Z-before-six-approved-exclusions` in
the training root. Local preparation was restarted with fresh grouped splits
and training-only statistics. Unchanged phase files are reused after checking
the prior manifest, dataset/config, parquet and phase hashes. Profiling runs
fresh under the revised family fingerprint, then automatic training follows
only after readiness checks. Errors now identify the video path/size, and
profiling avoids collective cleanup on failure so torchrun can terminate peers
promptly. Inspect live progress before claiming any stage has completed.

### October 12: resume completed profiling through repaired readiness

The revised six-exclusion family and all 25 performance trials/confirmations
finished. Final readiness timed out serializing the full dataset metadata into
workers for its eight-episode fixture. Compacting only unused fixture records
exposed differing per-GPU checkpoint keys (usable VRAM differs slightly) and
rank-local BatchNorm buffers being compared to the canonical leader checkpoint.
`workstation/readiness_krv100.py` fixes these orchestration assumptions, retains
the frozen numerical check, and passed the complete two-rank check including
the uneven tail, inference, strict reload and RNG restoration. Shared approval:
`family/distributed-profiles/253c537fc612db67.json` in the SSD training root.
Diagnostic attempts/logs are preserved under `diagnostics`; they did not update
production weights. The frozen numerical source and prepared family remain
unchanged. The repaired supervisor reused all 25 measurements and the completed
readiness record, with hashes/protocol/hardware validated and an explicit
`profiling/controller_resume.json` provenance record. The previously unfinished
timeline capture completed at 00:56 EDT on October 12, saving both rank traces.
Production launched on both GPUs at 00:56 EDT and is loading its dataset at this
snapshot. The winner is microbatch 32/GPU, eight workers/GPU, prefetch 2, pinned
memory, fused AdamW, BF16 and global batch 352; repeated warm throughput is
58.7177 samples/second. All 25 prior trials were reused, with only the missing
timeline trial added. Production
uses `workstation/train_krv100.py` to validate both GPU identities and the shared
readiness key before invoking frozen training. Check current status before
reporting optimizer progress or training completion; inspect the SSD training
root's `training.log` and `profiling/profiling_report.md`.

The first production launch failed at 00:59 EDT before updates because W&B 0.30
removed `wandb.util.generate_id`. `train_krv100.py` now installs a compatibility
alias to the SDK's `wandb.sdk.lib.runid.generate_id`; frozen numerical source
and all completed approvals remain intact. The 09:37 EDT restart reused
preparation/profiling, registered W&B run `n9a980u7` at 09:40 EDT, saved the
initial checkpoint. Real progress was verified at 09:51 EDT: 20 optimizer
updates / 7,040 samples, finite loss 0.76808, latest rate 57.93 samples/second,
and sampled GPU utilization 100% / 98%. No preparation or GPU profiling was
repeated for this fix. Status now shows startup separately from logged optimizer
counters and saved checkpoint counters; logs can advance beyond the latest
checkpoint between scheduled saves. Use live state for subsequent progress.

**Latest failure, October 12 at 09:58 EDT:** update 85 hit episode 195616's
opposite video, which has 33 frames for 109 parquet rows. The last logged update
is 80; the latest saved checkpoint is still update zero. A complete MP4 sample
table audit of all 414,600 retained views found three additional short videos:
wrist 195200 (3/106), opposite 195616 (33/109), wrist 195674 (2/107).
Exact decoding and fresh pinned-release downloads confirmed these counts and
official hashes. See `diagnostics/published-short-videos.json` and
`diagnostics/video-frame-counts.json` in the SSD training root. The user approved
these three additional exclusions. The exact selection now excludes nine
episodes / 984 frames and retains 207,297 episodes / 23,551,233 frames.
Prior family, failed run and full search are preserved under
`archives/20261012T142420Z-before-three-approved-short-video-exclusions`.
The coordinator restarted at 10:24 EDT using `revise_dom_family.py`: reuse
records/hashes/weights/phase caches, rebuild grouped splits and recalculate
training-only stats. Numerical source remains frozen. Reconfirm only the
previously selected runtime twice and rerun complete readiness on the revised
family; historical timeline traces retain their old manifest and new checksums.
Training then starts automatically from released weights in a new run because
the archived update-zero checkpoint belongs to the old family. Raw SSD and
Palmetto bytes remain unchanged. Check live status before claiming new updates.
Launcher checkpoint cadence now uses the canonical writer every five minutes
at completed optimizer boundaries; reviewed launcher SHA is in `request.json`.

The revised family completed at 10:40 EDT: manifest `8198f7c9569f0330ab05fc7b22c89b34fb55192e87836ea34a7112aca3fcc7e2`,
203,151 train episodes / 23,080,253 train frames. The first unfinished
reconfirmation was stopped during worker startup because it copied the entire
unused per-episode statistics table to all 16 workers. That attempt is archived
under `unfinished-confirmation-before-metadata-compaction-20261012T145010Z`.
`training_metadata.py` now clears only `meta.episodes_stats` after constructor
aggregation, retaining `base.stats`, metadata aggregate stats, tasks, episode
indices, hashes, phases and every input. Source inspection confirms only the
constructor reads the removed table. Both profiling and production use this
same adapter; request/search/controller record its SHA. No completed trial or
dataset preparation was repeated. The 10:50 EDT restart reused the family and
statistics. First reconfirmation passed at 10:54 EDT: 56.74 samples/s, GPU means
93.75% / 93.67%, peak combined process RAM 47.66 GiB, headroom passed. Second
confirmation and complete readiness gate automatic production. Inspect live logs.

Recovery succeeded: both confirmations passed at mean 56.999 samples/s with
headroom; complete readiness passed at 11:01 EDT and automatic production
launched. Historical timelines remain archived under their original manifest
with explicit hashes; no full runtime search or timeline replay was performed.
New W&B run is `2qh0ijb1` under `aparame-clemson-university/Kinematic-Residual-VLA`.
First optimizer update completed at 11:05:41 EDT. At 11:10:54, training had
50 updates / 17,600 samples, finite loss 0.723888 and latest rate 57.08 samples/s.
The five-minute checkpoint adapter saved update 40 / 14,080 samples at 11:09:52:
generation `u000000040-cb16ddbb`, fingerprint
`6f96cc469773756667357a9bb49292fb2eeff0979fb811fb4056fb8c252d37c4`.
Both A6000s are training in the user service. Check live progress before reporting
completion; no epoch or full training run is complete as of this observation.

## Current user budget: one epoch, then Table 1 (October 13)

The user shortened this run to one completed epoch and authorized a dependent
closed-loop Table 1 evaluation. Read SSD `training_control.json` and
`epoch1_table1_request.json`. Recipe/manifest/source/experiment fingerprints
remain unchanged; retain the original LR-scheduler horizon when resuming.
`epoch_training_control.py` stops after held-out epoch-one validation, writes a
protected final/epoch-001 generation and user-limit completion, and prevents
epoch-two updates. Launcher recovery handles an interruption between the final
atomic save and completion publication. Source numerical code remains frozen.
The controller was adopted by graceful SIGUSR1 checkpointing at update 16,654;
the same W&B run `2qh0ijb1` resumed and reached 16,720 updates at 15:21 EDT.
No completed dataset preparation or throughput profiling was repeated.

Dependency service: `krvla-epoch1-table1-20261013.service`, currently waiting.
Wrapper: `workstation/run-krvla-table1-after-epoch1.sh`; runner: `krvla_table1.py`.
Gate: successful training status/completion, epoch-one held-out validation,
exact total training frames/updates, matching final and epoch-001 pointers.
No GPU allocation while waiting. Pilot: nine trials. Full suite: the same 1800
Table 1 trials as the baseline, Sim 4.5.0 / Lab 2.2.1, separate simulation and
streaming inference GPUs. This evaluates public Table 1 configurations, not
reserved dataset-episode simulator bindings.

Result root:
`scratch/dom45/results/krvla-velocity-epoch001-paper45-a6000-20261013-01`.
Outputs include per-dimension trials/logs/videos, summary and Table 1 comparison
files, plus `table1_baseline_comparison.md/.csv` against the completed local DOM.
Object pose/velocity are additional live simulator inputs in robot coordinates;
an observation-only wrapper attaches them without physics/scoring changes.
Policy source is the frozen KRVLA snapshot; strict loading/finite action probe
gates simulation. Streaming warmup supplies synthetic auxiliary state only for
the internal dummy batch; real missing velocity remains an error. Queued hashes
are checked at startup and evaluation; do not silently edit bound code or inputs.
The evaluator owns a file lock, so duplicate launches cannot compete for results.
Check live status before claiming epoch completion or any closed-loop outcome.


## Previous workstation plan (October 9)

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

## Contact friction follow-up

The user subsequently requested increased gripper friction and another VLA test. We are testing **finger–object contact friction**, distinct from the earlier joint-friction calibration. New Palmetto evaluation **16899593** has a fresh unchanged-contact control plus finger static/dynamic coefficients 1.0 and 2.0, using the corrected/aligned configuration and one shared L40S for all arms. Three cases × two repeats × three conditions = 18 trials. At submission it was pending priority; check current status before reporting results. Archive job 16899599 follows it. Results are `RESULTS/official_dom_friction/20261009`.

The physics helper checks native readback before/after each trial and preserves object materials, restitution, nonfinger materials, combine modes and joint parameters. [Friction diagnostic guide](../evaluation/official_dom/diagnostics/friction_20261009/README.md) includes the Linux continuation commands (`--finger-friction 1 2` during preparation). No learned-policy benefit has been established at submission.
