# Recovered paired DOM training workflow

## Current Palmetto layout and 0.10/0.18 training

The canonical source checkout is
`/home/aparame/Research/Dynamic_VLA/third_party/DynamicVLA` (underscore; the
hyphenated `/home/aparame/Research/Dynamic-VLA` does not currently exist).
DOM data and Hugging Face caches remain under
`/scratch/aparame/.cache/huggingface`. Checkpoints and generated data stay
under `/scratch/aparame/Dynamic_VLA/runs`; Slurm and W&B logs go to
`/home/aparame/Research/Dynamic_VLA/logs`. The training code's `DIR.OUTPUT`
also puts checkpoints there on scratch; its built-in TensorBoard logs and
run summaries may share that output root. Scratch is purgeable; back up
important checkpoints.
The old scratch source tree is recovery material, not the launch directory.

`configs/dom_speed_ood_0p10_0p18_split.json` contains a reconstructed explicit
partition of 3,810 train episodes at mean speed <= 0.10 m/s and 119,778 test
episodes at >= 0.18 m/s. The derived eval split reserves 5,000 OOD validation
episodes and 114,778 OOD test episodes. `iid_eval` samples 512 training episodes
and is not held-out. The full-finetune configs now use this split rather than
rescanning speeds or using the obsolete 0.12/0.13 threshold. Old thresholds
remain in historical prose only, not active full-finetune configs.

On Palmetto, first ensure the log directory exists **before** `sbatch` opens
its output files. Submit CPU prefetch, verify its completion and data/video
availability, then submit one of the GPU profiles:

```bash
cd /home/aparame/Research/Dynamic_VLA/third_party/DynamicVLA
mkdir -p /home/aparame/Research/Dynamic_VLA/logs/slurm
sbatch slurm/preprocess_full_finetune_speed_ood_pair.sbatch
# Once preprocessing succeeds, check locally before choosing one GPU profile:
source scripts/palmetto_env.sh
python scripts/check_train_readiness.py
sbatch slurm/train_full_finetune_speed_ood_pair_h100.sbatch
# sbatch slurm/train_full_finetune_speed_ood_pair_a100.sbatch
# sbatch slurm/train_full_finetune_speed_ood_pair_h200.sbatch
```

The optional `slurm/train_full_finetune_speed_ood_pair_a100_4gpu.sbatch`
requests four A100 GPUs, uses batch 8 × accumulation 32 per GPU (effective
batch 1,024), and writes checkpoints to a separate
`/scratch/aparame/Dynamic_VLA/runs/a100_4gpu` tree. It may run alongside the
H100 job without overwriting its checkpoints. Submit the two jobs separately;
Slurm does not make either an automatic fallback for the other.

The GPU job trains baseline then residual, each for 30 epochs by default.
Every epoch it logs bounded observed `Eval/iid_eval/*` and `Eval/ood_val/*`
metrics to W&B and selects the best checkpoint by `ood_val/action_mse`.
`Eval/*/final_ee_object_proximity_rate_5cm` is only a predicted end-effector
proximity diagnostic against a constant-velocity object extrapolation, not
pick/place or closed-loop task success. Older runs logged the same diagnostic
as `final_success_rate`; do not compare that label to simulator success.
`EVAL_MAX_BATCHES: 8` limits each validation split per process (up to 2,048
samples per split with 8 GPUs and per-rank batch size 32); `ood_test`
remains untouched during training. The nominal `iid_eval` subset overlaps
training episodes and is not independent validation.
For a bounded single-H100 smoke run first, submit
`sbatch slurm/smoke_full_finetune_speed_ood_h100.sbatch` from the same directory.
It validates (or reconstructs if missing) the full 0.10/0.18 split, checks
training assets, selects eight deterministic low-speed episodes, disables
W&B, and writes its isolated config/checkpoint under
`/scratch/aparame/Dynamic_VLA/runs/smoke/<job-id>`. It does not train on the
119,778 held-out test episodes or resume a full experiment.
The three GPU profiles use 8 A100 (16×16), 8 H100 (32×8), or 2 H200 (64×16)
per-device batch × accumulation, respectively. These sizes are unverified for
the new split. Preprocessing and GPU jobs must not overlap; training also
checks the cache and may fetch missing video assets. `RUN_OFFLINE_COMPARE`
defaults to 0 because the full held-out OOD test is enormous; reserve `ood_val`
for selection and `ood_test` for final reporting. No training was submitted
as part of this recovery.

`scripts/palmetto_env.sh` loads `gcc/12.3.0` and `cuda/12.3.0`, then activates
the existing `dynamicvla` environment. Both modules are needed: CUDA supplies
`libnppicc.so.12`, while GCC provides the C++ runtime needed by the TorchCodec
video decoder. A top-level `import torchcodec` alone is insufficient; test
`import torchcodec.decoders`. No package update was needed for that import.
`pip check` still reports pre-existing fsspec/datasets and
rerun-sdk/LeRobot/NumPy version conflicts; do not upgrade them blindly on
Rocky 8. A cached DOM MP4 decoded successfully on the login node; repeat this
on an allocated compute node before large training.
Automatic resume requires `config.json`, `model.safetensors`, and
`training_state.pt`. No prior trained checkpoints were recovered.

**The remaining instructions below describe the historical paired workflow;
use the paths and precautions above instead.** From-scratch configs were not
recovered and their launcher exits rather than training the wrong experiment.

Recovered on 2026-09-22 from the public DynamicVLA repository and surviving local
development-session records. The original README is preserved in `README.md`;
use this guide for the recovered paired nominal/residual workflow. Other historical
experiments mentioned in the old README are not all verified or fully recovered.

## What is restored

- Nominal configuration: `configs/dynamicvla_dom_paired_velocity.yaml`.
- Residual configuration: `configs/residual_dynamicvla_dom_paired_velocity.yaml`.
- Shared training/model code, residual-flow implementation, dataset helpers,
  IID/OOD evaluation, and paired counterfactual velocity sweep.
- `scripts/train_paired_velocity_pair.sh`, GPU-aware submission wrapper, and
  `slurm/train_paired_velocity_pair_h100.sbatch`.
- Residual-only resume with `RESUME_RESIDUAL_ONLY=1` and general `AUTO_RESUME=1`.
- CPU tests and the historical dependency specifications.

Both configurations fine-tune `hzxie/dynamic-vla-DOM`; these are not from-scratch
or pretraining-leakage-free experiments. The baseline disables residual flow;
the residual configuration enables it. Both default to 30 epochs.

## Data and environment

The existing environment is `/home/aparame/.conda/envs/dynamicvla`. The cached
207,306 DOM episode parquet files survived under
`/scratch/aparame/.cache/huggingface/lerobot/hzxie/DOM`. This does not establish
that every video or pretrained-model artifact is available locally.

```bash
cd /home/aparame/Research/Dynamic_VLA/third_party/DynamicVLA
source scripts/palmetto_env.sh
```

The earlier login-node TorchCodec failure is resolved by loading both GCC 12.3
and CUDA 12.3 via `scripts/palmetto_env.sh`. `torchcodec.decoders` imports
successfully. Real video decoding and GPU memory usage still need compute-node
validation; unit tests stub video decoding.

The deleted low/high-speed split can be reconstructed locally with:

```bash
python scripts/rebuild_dom_speed_split.py
```

The reconstruction refuses to overwrite an existing split and checks the
historical counts: 3,810 training episodes (mean speed <= 0.10 m/s) and 119,778
high-speed episodes (>= 0.18 m/s). The derived eval file selects 512 training
episodes for `iid_eval`, 5,000 high-speed episodes for `ood_val`, and the
remaining 114,778 for `ood_test`. `iid_eval` is training-distribution monitoring,
not a held-out IID set. OOD test episodes are excluded from training-time model
selection. Counts alone do not prove byte-for-byte identity with the lost split.

## Train nominal, then residual

```bash
bash scripts/submit_paired_velocity_pair.sh h200 4
```

Supported GPU types are `a100`, `h100`, and `h200`, with 4 through 8 GPUs on one
node. Additional arguments go to `sbatch`, for example `--partition=...` or
`--account=...` when your cluster requires them. The historical `_h100` filename
is retained, but the wrapper overrides its GPU resource request.

The recovered profiles set per-GPU batch size to 16 for A100 40GB, 32 for A100
80GB/H100, and 64 for H200. Process count follows the detected allocation;
gradient accumulation targets an effective global batch near 1,024. These are
heuristics, **not benchmark-proven maximum-performance settings**. Actual memory
use must be checked on a compute node. `DVLA_BATCH_SIZE`,
`DVLA_GRAD_ACCUM_STEPS`, and `DVLA_N_WORKERS` remain overridable.

Checkpoint destinations are separate:

```text
/scratch/aparame/Dynamic_VLA/runs/checkpoints/dynamicvla_dom_paired_velocity/           # nominal
/scratch/aparame/Dynamic_VLA/runs/checkpoints/residual_dynamicvla_dom_paired_velocity/  # residual
```

The job runs both trainings sequentially and then observed IID/OOD evaluation
and the paired sweep. Set `RUN_OBSERVED_EVAL=0 RUN_PAIRED_SWEEP=0` to train only.
The configs retain the original W&B project/account settings; authenticate or
set `WANDB.MODE: offline` (or `WANDB.ENABLED: false`) in both YAML configs.

## Resume only the residual model

**No deleted trained checkpoint was recovered.** You cannot resume the previous
epoch-14 run using source files alone. First restore a backup of its checkpoint
directory, containing `config.json`, `model.safetensors`, and `training_state.pt`.
The latter is required to restore optimizer/scheduler progress.

```bash
RESUME_RESIDUAL_ONLY=1 RUN_OBSERVED_EVAL=0 RUN_PAIRED_SWEEP=0 \
  bash scripts/submit_paired_velocity_pair.sh h200 4
```

This skips nominal training and resumes from the residual checkpoint destination
above. Alternatively set `RESIDUAL_CKPT=/absolute/path/to/restored/checkpoint`.
Evaluation is disabled in this example because it also needs the trained nominal
checkpoint. Enable it once both trained checkpoints are available.

`AUTO_RESUME=1` applies to the ordinary two-stage workflow. Resume uses the current
checkpoint files in each experiment directory; it does not search arbitrary
folders for the most recently modified epoch file. Changing GPU count/batch
geometry on resume is supported operationally but is not a bit-identical replay.

## Evaluate a restored or newly trained pair

```bash
sbatch slurm/evaluate_residual_paired_velocity_sweep_h100.sbatch
```

This defaults to one H100 and the two paired checkpoint destinations. Override
the Slurm request at submission if needed. The sweep changes velocity metadata
without rerendering images, so it is a counterfactual diagnostic, not a replacement
for observed-data OOD evaluation.

## Recovery limits

Recovery combined upstream files with local patches and recorded file snapshots;
it is not a filesystem undelete or a complete backup of the old workspace.
No Slurm jobs were submitted, cancelled, or modified. No GPU training was run.
Keep a copy of the recovered source archive outside scratch storage, and back up
future checkpoint directories separately; the source archive excludes weights.
