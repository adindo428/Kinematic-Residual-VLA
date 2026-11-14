# DOMv real post-training on two Palmetto V100s

Slurm job: **20504544**, submitted from CPU allocation **20145158** on
**node1347**. GPU request: two V100s, 16 CPU cores, 64 GB RAM,
maximum 72 hours. The batch first performs a disposable distributed GPU preflight;
production loads the original base again only after that subprocess succeeds.

Run directory:
`/scratch/aparame/Dynamic_VLA/domv-real-v100x2-20261114`.

## Access and Palmetto policy

Follow [AI Code Assistants on Palmetto](https://docs.rcd.clemson.edu/palmetto/ai_code_assistants/)
and the [data transfer rules](https://docs.rcd.clemson.edu/palmetto/transfer/overview/).
Global rules are installed in `/home/ailab/.codex/AGENTS.md` and
`/home/aparame/.codex/AGENTS.md`.

All remote setup, editing, dependency checks, extraction, hashing, scheduler
queries and submissions run in the allocated CPU job. Login is control transport
only. The user explicitly permitted a lightweight `srun --jobid=20145158
--overlap` attach client because both direct and nested SSH authentication to the
CPU node failed. The CPU helper verifies the node and job environment on every
invocation, closes remote stdin and bounds control-script size. It cannot be
used for streamed bulk input. If the CPU allocation expires, obtain another
allocated CPU endpoint; never fall back to a login-node shell.

The archive was ultimately transferred directly using SFTP to
`hpcdtn01.rcd.clemson.edu`, then verified and extracted on node1347. An earlier
throttled login-routed stream was stopped after the supplied policy clarified
that bulk data must not pass through login hosts. Its incomplete file was
removed. Throttling does not make a bulk login transfer compliant. Future bulk
transfers must use a direct DTN or Globus. DTNs are file-transfer endpoints only;
no diagnostics, extraction, hashing, environment changes or Codex runs there.

Control socket: `/home/ailab/.ssh/palmetto-slogin.sock`.
DTN socket: `/home/ailab/.ssh/palmetto-dtn.sock`.
No passwords, keys or authentication caches are copied into the run.

## Initialization, data and optimization

This is DOMv: original DOM architecture with current object velocity in the
existing padded state slots, **no analytical KR**. Both local KRV and remote DOMv
start independently from the original released `hzxie/dynamic-vla-DOM`, revision
`c9609d94c5940eb42436a31927d7e80ae454279e`. SHA256:
`e3da8b1e0326d7a3ee6873f7fa17cdf7015201251901be5107377cb851ce3af8`.
No previously fine-tuned weights or optimizer/EMA state initialize this run.

The frozen real dataset is identical to local KRV: all 859 released static/moving
episodes across three tasks; 706 train / 80 validation / 73 test; 101,781 eligible
training anchors. Recorded prompts, train-only normalization, camera/history
layout [-2, 0], 20-action horizon and 25 Hz timing are retained. The CPU preflight
verified the original checkpoint, archive checksum, every episode's parquet
hash, quality/sampling hashes, split counts and a decoded 384-by-384 RGB sample.
Test episodes remain outside optimization and selection.

The quality-focused recipe is retained: native flow-only objective, full-model
training, action/expert LR 3e-5 versus backbone LR 3e-6, AdamW, 1,000 optimizer-update
warm-up, cosine decay, gradient clipping, capped task/source/phase sampling and
EMA. Maximum 60 epochs, earliest stopping at 15, patience 12. Selection compares
raw and EMA physical action errors on the same 512 anchors and three noise seeds
as the local KRV run. Full held-out flow validation is also logged.

V100 execution uses **FP32 master parameters and optimizer state, FP16 autocast and dynamic gradient scaling**. The released BF16 backbone is widened without changing its initial values. Saved parameter/autocast precision is recorded in checkpoint provenance and restored by the local inference loader. Two-process
NCCL DDP uses microbatch 2 per rank with sixteen accumulated passes, global effective
batch **64**. The final accumulation window is normalized by its actual sample
count. One seeded global weighted draw is sharded across ranks and padded from
101,781 to **101,784** draws to keep rank microbatches equal; this is three extra
replacement draws per epoch, not new demonstrations. Gradients synchronize only
at an optimizer boundary. EMA and physical selection run on rank zero; training
uses both GPUs. The preflight checks native sampler parity, finite FP16 updates
and synchronized checksums across all model parameters, discarding all its updates.

The existing user-owned `dynamicvla` conda environment is used without modification.
A per-run CPU TorchCodec 0.5 overlay avoids the shared environment's unavailable
CUDA/NPP decoder dependency. PyTorch is 2.7.1+cu126 with compiled sm_70 support.
The overlay is isolated under `runtime-deps/`. The GPU W&B producer remains offline; a separately requested CPU-only live-sync
job streams that local record to the dashboard. No workstation credentials were
transferred; syncing uses the existing Palmetto credentials. See the W&B section
below for its allocation and run ID.

## Monitoring and recovery

From the workstation, while CPU job 20145158 is still valid:

```bash
python3 workstation/palmetto_cpu.py --script workstation/palmetto/domv_status.sh
```

Use bounded, occasional checks rather than rapid Slurm polling loops. The helper
executes on the CPU allocation, not login or GPU nodes. Important remote records:

- `submission.json`, `cpu_preflight.json`, `disposable-check/preflight.json`;
- `logs/train-20504544.out` and `.err`;
- `domv/status.json`, `best_checkpoint.json`, `latest_checkpoint.json`;
- `domv/epoch_history/` and `training_state.pt`.

Selected native checkpoints retain deployable normalization/config and raw/EMA
provenance. Resume restores the raw training weights, optimizer, GradScaler, EMA
and per-rank RNG state at the last completed epoch. Partial epochs are excluded
from evaluation. A Slurm restart reruns the disposable check, then resumes if a
completed training state exists. A wall-time expiration is not an automatic
completion; resubmit through a CPU allocation when more training is required.

DOMv was removed from the workstation's waiting queue after the Palmetto job was
accepted. The active KRV process was left untouched. A replacement local monitor
records the remote DOMv job instead of launching a duplicate. See
`scratch/real_robot/finetuning-runs/fresh-dom-causal-full-real-20261114/remote_domv.json`.

The initial disposable job 20503556 failed before production: inherited CPU affinity exposed one core, and BF16 master gradients were incompatible with the FP16 GradScaler path. Job 20504544 uses explicit CPU binding settings, validates exposed cores before loading, and uses FP32 masters. Initial logs and submission metadata are retained for audit.

The corrected GPU preflight passed on two Tesla V100-PCIE-16GB GPUs on node1349: 16 CPU cores exposed, three finite FP32-master/FP16 optimizer updates, exact native sampling parity, synchronized parameters, and 11.81 GiB peak allocated memory. Audit copies are saved locally under the original run’s `palmetto/` directory. Production reloads the original base independently; these disposable updates are not reused.

Production was verified beyond startup: the first accumulated optimizer update
completed with finite loss/gradient norm, loss scale 128 and zero skipped overflow
updates. Its initial physical-error baseline and production status are also saved
in the local `palmetto/` audit directory. Initial throughput includes cold worker
startup and must not be used as the steady-state epoch ETA.

## W&B live synchronization

The GPU producer writes a complete local W&B stream in offline mode. The user
subsequently requested a separate CPU allocation for cloud syncing after the
metrics/configuration/console/system-metadata payload and project destination
were disclosed. CPU-only Slurm job **20509119** requests one core and 2 GB RAM
for up to 72 hours. It runs the installed `wandb beta sync --live` command against
the active stream, using the existing Palmetto credentials. No dataset or model
checkpoint is uploaded, and GPU training is not restarted or modified.

Cloud run: `aparame-clemson-university/Kinematic-Residual-VLA/bs8kmkz1`.
Dashboard: https://wandb.ai/aparame-clemson-university/Kinematic-Residual-VLA/runs/bs8kmkz1

The dashboard becomes available after the CPU job is allocated and registers the
run. The sync record is `wandb_live_sync.json`; its logs are
`logs/wandb-sync-20509119.{out,err}`. The current running training fragment is
`wandb/offline-run-20261114_095417-bs8kmkz1`. This supported CLI option follows an
active producer; an unclean producer crash may leave it waiting, bounded by the
CPU job's wall time. A future training restart that creates a new offline run
fragment needs a corresponding sync update. Keep all control/status commands in
an allocated CPU job; do not monitor from login.
