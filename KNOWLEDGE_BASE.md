# Project knowledge base

**Local DOM baseline setup (October 10, 2026).** The SSD workstation profile in
`workstation/env.local.sh` uses Isaac Sim **4.5.0**, Isaac Lab **2.2.1**, and two
RTX A6000 GPUs: GPU 0 for simulation and GPU 1 for released DynamicVLA inference.
All project environments, downloads, assets, weights, caches, temporary files,
and results live under `scratch/dom45` on the Linux SSD. The official DOM test,
object, and scene archives are downloaded and extracted. Upstream evaluation
source is pinned to `a27a06d2ca74d0e987a5e552e01013073e93cfd8`, before the Sim 5.1
migration; the released checkpoint is unchanged. A nine-trial pilot verified all
trial accounting, with 3 successes and no infrastructure failures. That pilot
does not establish the Table 1 score. The 1,800-trial suite was started as user
service `krvla-dom-paper45-20261010.service`; inspect its live status and
`scratch/dom45/results/dom-paper45-a6000-20261010-02/suite_status.json` before
reporting progress or completion. See `workstation/README.md` for commands.

**Queued local training (updated October 11, 2026).** The user authorized all DOM
episodes and training **KRVLA + Velocity (`krv100`)** after baseline evaluation.
Dataset revision `bbf6baa15f7253746d271a6e958e037afa537b1e` includes 207,306 episodes;
the user subsequently excluded side-camera downloads, matching the released
baseline and our training inputs: wrist and opposite cameras. Required coverage
is 621,926 files / 193.74 GB, including all metadata/parquets and both views.
Previously downloaded side-camera files are retained. Service
`krvla-dom-full-download-20261010` saves inventory/checksum progress under
`scratch/dom45/training/dom-full`. Service `krvla-krv100-after-eval-20261010` waits
for complete verified baseline/two-camera data and
the user's own W&B login via
`bash workstation/wandb-ssd.sh login`; the PC's earlier `omkarmayekar555` login was
not the selected account. The SSD login was subsequently confirmed as `aparame`,
with visible project `aparame-clemson-university/Kinematic-Residual-VLA`;
write access is checked again by a readiness run before GPU profiling/training.
Inspect `bash workstation/krv100-ssd.sh` for live state.
The Palmetto import completed; its source lacked some wrist/opposite videos.
Only missing or mismatched required files are downloaded from HF using pinned
resolver GETs to reduce request overhead, with rate-limit resets respected.
Optional `bash workstation/hf-ssd.sh` stores an HF login on SSD. The user
authorized completing `/scratch/aparame/.cache/huggingface/lerobot/hzxie/DOM`
after local completion: service `krvla-dom-palmetto-backfill-20261011` copies
required files and verifies pinned hashes remotely, repairing mismatches. It
does not add side-camera videos. Once the upload finished, the user requested
local preparation/profiling proceed independently of remote checksum verification.
The coordinator no longer requires verified Palmetto completion; remote checks
continue in parallel. Inspect `palmetto_backfill_status.json` in the SSD's
`training/dom-full` directory before claiming remote completion.
Training uses the existing fixed-rate DOM100 recipe, a new full-data family,
and mandatory two-A6000 throughput profiling plus actual readiness checks.
Profiling compares microbatches, loader workers, prefetch/pinned transfers and
fused AdamW; repeated warm optimizer measurements select the fastest safe
configuration while retaining effective batch 352 and 15% memory headroom.
Reports, trial metrics and CPU/CUDA timeline traces are saved in the training
root's `profiling` directory. Production has not started at
preparation time. Source, request, logs, status, and future checkpoints are under
`scratch/dom45/training/krv100-full-20261010`.

**Local profiling failure (October 11, 2026).** Environment preparation and the
207,306-episode family completed; the SSD and Palmetto required-view caches match
all published file hashes. Profiling recorded 18 trials, then stopped on an
empty wrist-camera video for episode 195417. The peer NCCL timeout followed the
data-loader failure. Six required-view MP4s in the pinned release contain only
48-byte headers and no frames: opposite episodes 194633/195603 and wrist episodes
194760/195154/195405/195417. All six match the published SHA256
`10a394e1002fb5eef69251b70181b9b1ea302e2aff2146e77f629392e34cf516`;
a fresh pinned HF GET confirms the failing file, so this is not a transfer
checksum mismatch. They affect six training episodes / 662 frames. Diagnostic
evidence is under the SSD training root's `diagnostics/published-empty-videos.json`.
Production has not started. Excluding these episodes requires a revised family,
statistics and profiling provenance. The user subsequently **approved all six
exclusions**. `approved_exclusions.json` in the SSD training root records the
exact episode IDs, affected files/hashes and authorization; the revised request
requires exactly those exclusions and no additional omitted episodes. The new
family retains 207,300 episodes / 23,551,555 frames, excluding 662 frames.
The previous request, source, family, profiling results and stage logs were
preserved in `archives/20261011T205207Z-before-six-approved-exclusions` under the
training root. Raw SSD/Palmetto caches remain complete at the pinned revision.
The coordinator was resumed to rebuild splits and training-only statistics,
then run fresh profiling and automatic training after readiness checks. Phase
caches are reused only when the old manifest/artifact hashes, dataset/config,
per-episode parquet and phase hashes match; prior performance approvals are not
reused. Decoder errors now include the path/size. Failed profiling ranks exit
without waiting in collective cleanup, allowing torchrun to stop their peers.
Inspect live status before claiming the rebuild/profiling/training completed.

**Readiness recovery (October 12, 2026).** The six-exclusion family finished
preparation with fingerprint
`4ed681446a66e14ff8e0abf43bc71867845ccb2cc22089d53c9f460194879f54`;
25 profiling trials, including all repeated confirmations, completed. The
final readiness stage timed out while serializing full-release metadata for its
eight-episode fixture. After compacting only unused fixture records, two more
readiness assumptions were exposed: GPU 0/1 report different usable VRAM
(51,033,931,776 / 51,031,572,480 bytes), causing rank-local checkpoint keys to
diverge; and local BatchNorm buffers differ with `broadcast_buffers=False`.
`workstation/readiness_krv100.py` uses the frozen numerical check, limits the
fixture's metadata/hashes to its eight episodes, shares the leader checkpoint
path/fingerprint, aligns buffers only for canonical checkpoint comparison, and
records both GPUs' identities. The complete check passed on both GPUs: 353
samples in two updates, strict reload, per-rank RNG restoration, future inputs
ignored, and memory headroom. Its shared profile is
`family/distributed-profiles/253c537fc612db67.json` under the SSD training root.
Fatal adapter errors exit promptly so CUDA/NCCL destructors cannot hide them.

The numerical source digest (`620c3c9e...`) and all prepared family artifacts
remain unchanged. `profiling/controller_resume.json` records the controller-only
repair. The supervisor validates every unchanged measurement input and cached
result hash, retains the actual original producer identity, and reused all 25
trials plus the completed readiness record. W&B write verification is reused
after confirming the same account/team. The remaining timeline capture completed
on October 12 at 00:56 EDT, with both rank traces saved. Production launched on
both GPUs at 00:56 EDT and is loading its dataset at this snapshot. Selected
runtime: microbatch 32/GPU, eight loader workers/GPU, prefetch 2, pinned memory,
fused AdamW and BF16, retaining global batch 352. Two warm confirmations average
58.7177 samples/second; actual epoch throughput includes additional I/O. This
recovery reused all 25 prior trials; only the missing timeline trial was added.
`workstation/train_krv100.py` validates both GPUs against the performance and
shared readiness records, then invokes the frozen `train_distributed` function.
The status command shows `Completed profile trials reused`.
Production log: `training.log` in the SSD training root. Inspect live progress
before claiming optimizer updates or training completion.

**W&B startup fix (October 12, 2026).** The 00:56 production launch stopped at
00:59 before any optimizer updates: W&B 0.30 removed
`wandb.util.generate_id`, which the frozen training code uses. The workstation
launcher now supplies the installed `wandb.sdk.lib.runid.generate_id` helper at
the historical location, preserving numerical source, family and profiling
approvals. The coordinator restarted at 09:37 EDT using the completed selection
and cached data/statistics/W&B readiness; no GPU profiling or preparation was
repeated. W&B training run `n9a980u7` initialized at 09:40 EDT under
`aparame-clemson-university/Kinematic-Residual-VLA`, and the initial checkpoint
was saved. Actual optimizer progress was verified at 09:51 EDT: 20 updates /
7,040 samples, finite loss 0.76808, latest rate 57.93 samples/second, with sampled
GPU utilization 100% / 98%. The first update included worker startup; later
updates reached the expected warm throughput. The status command
now distinguishes startup from logged optimizer progress and displays latest
logged updates/samples/loss/rate plus the latest saved checkpoint's counters.
Check live logs before claiming optimizer updates; a launch alone is not proof
that training has progressed. Initial checkpoint counters remain zero until
the first scheduled save; logged optimizer counters show more recent progress.

**Short-video failure (October 12, 2026, 09:58 EDT).** The next failure occurred
at optimizer update 85: episode 195616's opposite video has only 33 frames for
109 parquet rows. Last logged progress was 80 updates / 28,160 samples at
61.04 samples/s; the last saved checkpoint remains update zero. The completed
audit of all 414,600 retained camera videos found exactly three MP4 sample-count
mismatches: wrist 195200 (3/106), opposite 195616 (33/109), wrist 195674 (2/107).
Exact decoding and fresh pinned-release downloads confirm all three; their
hashes match the official inventory. Evidence is in the SSD training root's
`diagnostics/video-frame-counts.json` and `diagnostics/published-short-videos.json`.
The user approved these three additional exclusions (322 frames), for nine
excluded episodes / 984 frames total. The revised selection retains 207,297
episodes / 23,551,233 frames. The prior family, failed training run and complete
profiling search are preserved under
`archives/20261012T142420Z-before-three-approved-short-video-exclusions`.
`workstation/revise_dom_family.py` reuses episode records, video/parquet hashes,
weights and phase caches; it rebuilds grouped splits and recalculates training
statistics once. Numerical training source remains frozen at `620c3c9e…`.
The coordinator restarted at 10:24 EDT. After family revision, the profiler
reconfirms the previously selected runtime with two fresh ten-update runs and
the complete two-GPU readiness check. The full runtime search and both original
timeline traces are retained with their original manifest provenance. Fresh
confirmations/readiness must pass on the revised family before automatic training.
The new family starts a new training run from released weights; the failed
run's checkpoint at update zero belongs to the archived, different family.
The production launcher now uses the existing atomic checkpoint/RNG writer at
completed optimizer boundaries every 300 seconds, including resumed runs;
its new SHA and cadence are recorded in `request.json`. This control change
does not change model, loss, optimizer, scheduler, sampling or data numerics.

**Worker startup repair (October 12, 2026).** The nine-exclusion family completed
at 10:40 EDT with fingerprint `8198f7c9569f0330ab05fc7b22c89b34fb55192e87836ea34a7112aca3fcc7e2`,
203,151 training episodes / 23,080,253 training frames (grouped resplitting also
changes held-out membership). The full numerical snapshot is still `620c3c9e…`.
The first unfinished reconfirmation spent its startup copying all 207,306
unused per-episode statistics entries into each of 16 workers. It had not
reached measured updates when stopped. That unfinished attempt is preserved in
`archives/unfinished-confirmation-before-metadata-compaction-20261012T145010Z`.
`workstation/training_metadata.py` clears only `meta.episodes_stats` after the
frozen constructor has computed and retained `base.stats`; episode metadata,
tasks, hashes, phases, inputs and both aggregate statistics dictionaries stay
unchanged. The frozen adapter references episode statistics only in its
constructor. `profile_krv100_worker.py` applies this adapter before running the
unchanged frozen profiling entry point; production applies the same adapter.
Adapter hashes are recorded in request/search/controller, and production checks
that its metadata adapter matches the fresh performance approval. The 10:50 EDT
restart reused the completed family/statistics. First confirmation passed at
10:54 EDT: 56.740 samples/s, mean GPU utilization 93.75% / 93.67%, peak combined
process RAM 47.66 GiB, required memory headroom passed. Check live logs for the
second confirmation, readiness and actual training progress before reporting them.

**Recovery verified (October 12, 2026, 11:10 EDT).** Both fresh confirmations
passed (56.740 / 57.258 samples/s; mean 56.999), preserving 15% GPU/host memory
headroom. Their combined process RAM peaks are 47.66 / 47.39 GiB versus the
prior search's 147.36 / 148.03 GiB. Complete two-GPU readiness passed again
on the revised family and published approval at 11:01 EDT. Both historical
timeline files remain archived with their original manifest and new file hashes;
they were not relabeled as measurements of the revised data. Production launched
automatically. New W&B training run is `2qh0ijb1`:
https://wandb.ai/aparame-clemson-university/Kinematic-Residual-VLA/runs/2qh0ijb1
First update completed at 11:05:41 EDT (57.7 seconds including worker startup).
Progress verified at 11:10:54: 50 updates / 17,600 samples, finite loss 0.723888,
latest warm rate 57.08 samples/s. The five-minute checkpoint adapter saved update
40 / 14,080 samples at 11:09:52 using the canonical atomic writer. Latest
generation is `u000000040-cb16ddbb`, experiment fingerprint
`6f96cc469773756667357a9bb49292fb2eeff0979fb811fb4056fb8c252d37c4`.
Training continues in the user service; this is a historical progress observation,
not a claim that an epoch or the full run is complete. Check live status/logs.

**One-epoch budget and dependent Table 1 evaluation (October 13, 2026).** The
user requested one epoch for now, then closed-loop Table 1 statistics for the
currently trained KRVLA+Velocity checkpoint. `training_control.json` in the SSD
training root sets one completed epoch, tied to manifest/recipe fingerprints.
The frozen recipe and its original 100-epoch LR-scheduler horizon are retained
so existing optimizer/scheduler/RNG progress and the first-epoch LR trajectory
remain unchanged. `epoch_training_control.py` intercepts the completed,
validated epoch boundary, saves final/epoch-001 pointers atomically, and exits
before any epoch-two optimizer update. W&B closes successfully with
`completion_reason=user_epoch_limit`. The launcher recovers a terminal save if
interrupted before completion publication. This is a user budget limit, not
early stopping based on convergence.
Training was gracefully checkpointed at update 16,654 / 5,862,208 samples and
resumed in the same W&B run `2qh0ijb1`; at 15:21 EDT it had reached update 16,720.
Completed preparation and GPU profiling were reused. The frozen numerical
snapshot remains `620c3c9e…`, recipe `3237dd89…`, experiment `6f96cc46…`.

The independent user service `krvla-epoch1-table1-20261013.service` runs
`workstation/run-krvla-table1-after-epoch1.sh`. Its request is
`epoch1_table1_request.json` in the SSD training root. It waits for successful
training completion, completed held-out validation, exact epoch-one sample and
update counts, and matching final/epoch-001 checkpoint pointers before any GPU
allocation. Then a nine-trial pilot gates the complete 1800-trial DOM suite:
the same 90 published configurations, seeds, 20 repeats/configuration, Isaac
Sim 4.5.0 / Isaac Lab 2.2.1, GPU 0 simulator / GPU 1 streaming policy, Euler/delta
controller and success/time scoring as the completed baseline. All results go
to `scratch/dom45/results/krvla-velocity-epoch001-paper45-a6000-20261013-01`.
The benchmark is distinct from the training-split held-out loss evaluation;
it does not pretend to map public Table 1 scenes to reserved dataset episodes.

`krvla_table1_server.py` adds current robot-frame target-object pose and actual
linear velocity to the observation payload, reusing values the official
simulator already computes; physics/scoring remain upstream. These are
additional privileged policy inputs and are recorded explicitly in protocol.
The trained checkpoint's frozen policy/inference source is used, not the
released DOM policy class. `inference_krvla.py` / `krvla_policy_runtime.py` supply
an auxiliary zero object state only for the streaming process's internal
synthetic warmup (the frozen warmup enumerates input_features, excluding the
auxiliary environment field). Real observations never get missing-velocity
fallbacks. The GPU probe uses strict checkpoint loading, preserves tied aliases
and normalizers, and exercises a finite 20x7 action chunk with velocity present.
Queued code, source, baseline protocol and model hashes are checked. Final
`table1_comparison.*` plus `table1_baseline_comparison.md/.csv` compare the trained
policy to the paper and local DOM baseline. No closed-loop results exist yet;
the service currently reports `waiting_epoch_one`. `krv100-ssd.sh` shows the
budget, dependency state and result path.

**Current training state (September 22, 2026).** Palmetto source lives at
`/home/aparame/Research/Dynamic_VLA/third_party/DynamicVLA`; large dataset
caches live in `/scratch/aparame/.cache/huggingface`, checkpoints in
`/scratch/aparame/Dynamic_VLA/runs`, and Slurm/W&B logs in
`/home/aparame/Research/Dynamic_VLA/logs`. The reconstructed 0.10/0.18 split is
present and validated for counts and disjointness (3,810/119,778). Full-finetune
launchers use explicit split configs and the home checkout. Read
`third_party/DynamicVLA/RECOVERY.md` for submission order and cautions; no
GPU training has yet validated this recovered workflow.

This document separates the paper's research claim from the recovered implementation and the historical run notes. It reflects files inspected on September 22, 2026; it is not a record of a successful rerun.

## Sources of truth

1. `paper/iclr2025_conference.tex`: research hypothesis, mathematical definitions, theorem, limitations, and proposed evaluation. This is an unfinished manuscript without measured results.
2. `third_party/DynamicVLA/RECOVERY.md`: current recovery status and the paired nominal/residual workflow. Prefer this over historical paths and commands in the original README.
3. `third_party/DynamicVLA/README.md`: upstream DynamicVLA overview plus historical local experiment instructions. Some old experiments and configs have not been fully recovered.
4. `third_party/DynamicVLA/PROACTIVE_RESIDUAL_MVP.md`: historical speed-OOD plan. It contains paths and split/config names that may not exist in this recovery.
5. The pasted prior chat described a residual rolling checkpoint at saved epoch 15 and baseline at epoch 14. Those were historical observations, **not** recovered checkpoint files.

## Research model

The context \(c=(I,\ell,q)\) comprises visual observation, language, and proprioception. The target \(x_1\) is a flattened horizon-\(H\) action chunk. The method models target velocity \(v^o\) as a predictable displacement \(d_c(v^o)\) in action space rather than only as conditioning.

The paper's mean-preserving residual target is

\[
\widetilde x_1=x_1-d_c(v^o)+\bar d_c,\qquad
\bar d_c=\mathbb E[d_c(V^o)\mid c].
\]

A conditional flow-matching action expert generates \(\widetilde x_1\), then inference adds \(d_c(v^o)-\bar d_c\) to reconstruct the raw action. The joint-space proposal uses a local Jacobian pseudoinverse, frame transform, and horizon lift. The recovered Cartesian implementation in `proactive_vla/residual_flow.py` instead applies drift to XYZ channels and leaves rotation and gripper to the learned model; its implementation supports velocity, trajectory, and lookahead modes. The code's reverse-time flow convention differs from the paper's forward-time notation, so check signs and time schedules when comparing equations with code.

The variance-reduction theorem requires the local decomposition \(X_1=R_c+d_c(V^o)\) and conditional independence of \(R_c\) and velocity. Under those assumptions, residualized conditional laws coincide and velocity-induced Wasserstein variability vanishes; covariance trace decreases by the variance of the projected displacement. This does not imply empirical success under noisy estimates, nonlinear dynamics, or velocity-dependent grasp strategies.

## Recovered paired-velocity workflow

The nominal and residual configs are `third_party/DynamicVLA/configs/dynamicvla_dom_paired_velocity.yaml` and `third_party/DynamicVLA/configs/residual_dynamicvla_dom_paired_velocity.yaml`. Both fine-tune the pretrained `hzxie/dynamic-vla-DOM` model for 30 epochs; the baseline disables residual flow, while the residual config enables it. They are **not** from-scratch comparisons.

The entry points are:

| Purpose | File |
| --- | --- |
| GPU type/count selection (A100, H100, H200; 4–8 GPUs) | `third_party/DynamicVLA/scripts/submit_paired_velocity_pair.sh` |
| Slurm allocation and runtime batch tuning | `third_party/DynamicVLA/slurm/train_paired_velocity_pair_h100.sbatch` |
| Sequential nominal/residual stages and evaluation switches | `third_party/DynamicVLA/scripts/train_paired_velocity_pair.sh` |
| Model training, checkpoint state | `third_party/DynamicVLA/core/train.py`, `third_party/DynamicVLA/run.py` |
| Residual action projection | `third_party/DynamicVLA/proactive_vla/residual_flow.py` |
| Observed IID/OOD comparison | `third_party/DynamicVLA/scripts/evaluate_baseline_residual_iid_ood.py` |
| Counterfactual velocity diagnostic | `third_party/DynamicVLA/scripts/evaluate_residual_paired_velocity_sweep.py` |

The wrapper selects 16 samples/GPU for A100 below 70 GiB, 32 for A100 at least 70 GiB or H100, and 64 for H200, with accumulation targeting global batch ~1024. These are unverified heuristics, not benchmarked maximum-performance values. The underlying sbatch filename says H100 but the wrapper overrides its GPU request. In this copy the sbatch enters `SLURM_SUBMIT_DIR`, so invoke the wrapper from the DynamicVLA project root. The older `RECOVERY.md` workflow still shows the scratch path; adapt its commands to the checkout being used.

The stage script defaults to `AUTO_RESUME=0`; `RESUME_RESIDUAL_ONLY=1` skips baseline and requires `config.json`, `model.safetensors`, and `training_state.pt` in the residual checkpoint directory. The last file contains optimizer/scheduler progress. Do not infer resumability from a high-numbered `model.epoch*.safetensors` alone; historical overlapping jobs left stale snapshots. Do not submit duplicate jobs writing the same checkpoint directories.

## Data, outputs, and storage

The DOM cache is under `/scratch/aparame/.cache/huggingface/lerobot/hzxie/DOM`. The reconstructed 0.10/0.18 split is present; all 3,810 training parquets and required two-camera MP4 files passed an existence preflight. Video decoding, pretrained weights, and GPU training are not yet validated. Historical README sections mention 0.12/0.13 m/s splits but are no longer active. `iid_eval` draws from training episodes, not a held-out IID test set.

Keep source, manuscript, configurations, and irreplaceable summaries in this Git repository and on backed-up storage. Keep large caches, datasets, and checkpoints on scratch for active work, but back up trained weights elsewhere because Palmetto scratch is purged and not a durable archive. This repository excludes generated runs, weights, and secrets via `.gitignore`. No lost trained weights were recovered.

## Open issues before rerunning

- Validate real video decoding and pretrained model load on an allocated compute node, then a small GPU batch before a full submission.
- Back up checkpoints to durable storage; scratch is not archival.
- Verify W&B project/account settings, Slurm partition/account, resource limits, and run flags before submission.
- Resolve the paper's truncated introduction, legacy “ProactiveVLA” naming, placeholder text, and apparent CFM loss discrepancy (`x_s-x_0` versus the stated path derivative `x_1-x_0`).
- Run empirical baseline/residual comparisons before claiming improved success or speed generalization.
