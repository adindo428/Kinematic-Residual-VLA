# Project knowledge base

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

The recovered guide reports 207,306 DOM episode parquet files under `/scratch/aparame/.cache/huggingface/lerobot/hzxie/DOM`; video and pretrained weights were **not** fully verified. It describes a reconstructable speed split with train/IID mean speed at most 0.10 m/s and test/OOD at least 0.18 m/s, but the split file itself is absent from this copy. Historical README sections also mention 0.12/0.13 m/s splits: do not conflate the experiments. The recovered guide says `iid_eval` draws from training-distribution episodes, not a held-out IID test set.

Keep source, manuscript, configurations, and irreplaceable summaries in this Git repository and on backed-up storage. Keep large caches, datasets, and checkpoints on scratch for active work, but back up trained weights elsewhere because Palmetto scratch is purged and not a durable archive. This repository excludes generated runs, weights, and secrets via `.gitignore`. No lost trained weights were recovered.

## Open issues before rerunning

- Recover or regenerate the DOM split and validate all required data/video and pretrained model assets.
- Fix the compute-node environment (`libnppicc.so.12` was missing on the recovery host), then run an import smoke test and CPU tests. The recovery notes report 14 CPU tests passing with video-decoder stubs, but no GPU run.
- Reconcile hard-coded scratch paths with the permanent source location; select an explicit location for outputs and periodic durable checkpoint backups.
- Verify W&B project/account settings, Slurm partition/account, resource limits, and run flags before submission.
- Resolve the paper's truncated introduction, legacy “ProactiveVLA” naming, placeholder text, and apparent CFM loss discrepancy (`x_s-x_0` versus the stated path derivative `x_1-x_0`).
- Run empirical baseline/residual comparisons before claiming improved success or speed generalization.
