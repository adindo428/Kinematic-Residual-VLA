# Recovered paired DOM training workflow

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
cd /scratch/aparame/Dynamic_VLA/third_party/DynamicVLA
conda activate dynamicvla
export HF_HOME=/scratch/aparame/.cache/huggingface
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
mkdir -p runs/slurm_logs
```

The full import smoke test on this recovery host is blocked by TorchCodec's
missing CUDA library `libnppicc.so.12`. Before submitting training, ensure the
compute-node environment exposes the matching CUDA 12 NPP libraries and that
`python run.py --help` succeeds. `requirements_rocky8.txt` includes
`nvidia-npp-cu12`; no packages were installed or changed during recovery.
Unit tests stub the video decoder and do not validate real video decoding.

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
runs/checkpoints/dynamicvla_dom_paired_velocity/           # nominal
runs/checkpoints/residual_dynamicvla_dom_paired_velocity/  # residual
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
