# Portable VLA post-training

This branch contains **code and setup instructions** for SmolVLA, VLASH and pi0.5,
each with/without analytical KR. Dataset labels, evaluation outputs, private
protocol snapshots, credentials and weights are excluded from GitHub.

Default preparation starts **new velocity-conditioned runs** from pinned released
parents. Existing v6 family runs used velocity only for analytical KR; the neural
models did not receive it. Adding velocity prompts is a new input protocol, with
unchanged layers and tensor shapes. Existing workstation jobs are unaffected.

## Preserve the exact metrics and data

On the source computer, export the frozen dataset/metric settings locally using
this branch's exporter (the source campaign remains in your original checkout):

```bash
python3 post_training/export_local.py \
  --source-campaign /path/to/original/checkout/scratch/real_robot/family-finetuning-v6/campaign.json \
  --video-revision 5fd5ef3e4256d2fc02e653a0345ab3b71f6a71df \
  --output /path/to/private-handoff/frozen
```

Transfer that **private directory directly** to `post_training/frozen/` in the new
checkout. It is gitignored. Alternatively set `POSTTRAIN_FROZEN` to its absolute
location on the destination. The exported bundle contains the exact local
velocity/grasp derivative labels, quality masks, splits, train-only statistics,
sampling weights, original ten-episode evaluation list, validation anchors/seeds,
checkpoint provenance and source hashes. Published videos are downloaded from
the pinned dataset commit and checked against source SHA256 values. HF main
alone does not reproduce the local derivative. The current source videos were
verified against the pinned revision above before preparing this handoff.

The exporter does not copy weight files or videos. Current fine-tuned weights
and optimizer state require a separate transfer if you need to resume those runs.
The complete source dataset and fine-tuning outputs remain on the original machine.

## Destination setup

Use Linux with NVIDIA CUDA support. Install FFmpeg shared libraries for TorchCodec;
set `POSTTRAIN_FFMPEG_LIB` before sourcing the environment if they are in a
nonstandard location. The launcher runs one job at a time on physical GPU 0;
it does not implement distributed training across computers.

```bash
git clone --branch post-training-vla https://github.com/adindo428/Kinematic-Residual-VLA.git
cd Kinematic-Residual-VLA
python3.10 -m venv .venv-post-training
.venv-post-training/bin/python -m pip install --upgrade pip
.venv-post-training/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
.venv-post-training/bin/python -m pip install torchcodec==0.5 -r post_training/requirements.txt
source post_training/env.sh
"$POSTTRAIN_PYTHON" post_training/manage.py patch-transformers
```

For a GPU requiring CUDA 12.8, use the `cu128` wheel index with the **same**
Torch/TorchVision versions. Actual build versions are recorded. Native OpenPI
compatibility files are pinned and SHA-checked. Authenticate locally with
`huggingface-cli login` and `wandb login` if required. Never commit credentials.
Keep enough SSD space for the selected parent weights and full raw/EMA/optimizer
snapshots. The runtime and trainable masks retain the source campaign settings.

After transferring the private bundle, assign each run to only one computer:

```bash
# Omit --families for all six; another computer can own the pi05 pair.
"$POSTTRAIN_PYTHON" post_training/manage.py prepare --families vlash_baseline,vlash_kr
"$POSTTRAIN_PYTHON" post_training/manage.py download
"$POSTTRAIN_PYTHON" -m pytest -q real_robot/tests/test_post_training_handoff.py real_robot/tests/test_family_finetuning.py real_robot/tests/test_family_throughput.py real_robot/tests/test_family_benchmark_gate.py
"$POSTTRAIN_PYTHON" post_training/manage.py run
```

Set `POSTTRAIN_ROOT` before sourcing `env.sh` to change the default
`scratch/real_robot/family-finetuning-v7`; pass `--root "$POSTTRAIN_ROOT"` to each
manage command in that case. `prepare` refuses to overwrite a campaign. Downloads
and launch may be retried. Use tmux or your scheduler to survive disconnects.

The launcher checks the source/data/runtime and benchmarks **all assigned
checkpoints on the destination** before production training. It saves reports,
checks native sampler parity/gradients/EMA and then trains. Source benchmark
reports are provenance only and do not approve new hardware or velocity inputs.
Progress is in `ROOT/queue.json`, per-family logs/status files and W&B.

## Current object velocity, unchanged architecture

All six default runs pass current, quality-checked robot-base object velocity
through the existing language tokenizer and encoder:

```
Object base velocity mm/s: 125,-11,0; grasp the ball.
```

Source m/s values round to integer mm/s (at most 0.5 mm/s rounding per axis).
The existing quality policy supplies neutral zero after annotated holding,
matching the KR gate. No future trajectory enters this prefix. Training, raw/EMA
validation, final before/after evaluation and deployment must use this same
wrapper. Baseline/KR pairs receive the same velocity information. KR still
subtracts/restores analytical drift exactly once. Parent pretraining histories
remain those in the exported release manifest; they were not a matched ablation.

Backbones, action experts/projections, camera layout, state width, horizon,
normalization, parameter names/shapes and trainable masks remain unchanged.
The native token limits stay 48 for SmolVLA and 200 for pi0.5/VLASH. Prompts that
would truncate input fail explicitly. Every destination benchmark/readiness
check verifies that changing velocity changes neural input tokens. Whether the
policies learn to use velocity effectively needs measured action/robot evaluation.

Adding this input starts a new W&B recipe. Resume rejects a checkpoint with a
different conditioning contract, even if its tensor shapes match.

## Identical metric definitions

The physical metric implementation is unchanged from the source campaign and
has a synthetic numeric regression fixture. The private protocol locks the exact
real anchors, labels, masks, normalization, statistics and seeds across machines.

- `first/position_mse_m2`: XYZ squared error averaged across three axes, in m²,
  for the first predicted action. `full/` averages valid horizon actions.
- `position_distance_m`: mean Euclidean XYZ distance in meters; multiply by 1,000
  for mm. It is not RMSE or controller execution error.
- Rotation: chordal SO(3) error and geodesic degrees. Gripper: MSE and MAE.
- Selection: unchanged scaled physical score, with identical early-horizon
  weighting; raw and EMA use the same anchors and three fixed noise seeds.
- Final `before_after.json`: the same original ten validation episodes and all
  eligible anchors, seed 12345, and unchanged expert targets. Initialization
  and selected weights both use the selected input recipe.

Effective flow batch remains 16. Destination profiling may tune microbatch,
accumulation, workers and prefetch; LR, losses, augmentation, sampling weights,
epoch limit and early stopping remain locked. The protocol checks reject changes.
Identical metric definitions do not promise identical numerical training across
GPU models/worker schedules. These are offline target metrics, not task success.

## Resume an existing run

For the original v6 recipe, prepare a separate root with `--conditioning native`.
Transfer a stable raw `latest.pt` or `recovery.pt` and the run's JSON sidecars,
`initial_trainable.pt`, and best referenced inference checkpoint into
`ROOT/runs/FAMILY/`. Retain `baseline_selection.json`, all
`validation-epoch-*.json`, `best_checkpoint.json`, `status.json` (W&B identity),
and `run_config.json`. Stop the old writer before handing over ownership and
verify checkpoint hashes. EMA-selected `best.pt` does not contain the optimizer.

After destination profiling/readiness, the queue resumes incomplete assigned
runs automatically. To specify one raw snapshot explicitly:

```bash
bash workstation/family-python.sh workstation/train_family_finetune.py \
  --campaign "$POSTTRAIN_ROOT/campaign.json" --family smolvla_kr --mode train \
  --output "$POSTTRAIN_ROOT/runs/smolvla_kr" --resume \
  --resume-checkpoint "$POSTTRAIN_ROOT/runs/smolvla_kr/latest.pt"
```

This command still requires matching destination benchmark/readiness files.
Resume restores model, optimizer, scheduler, EMA, RNG and sampler cursor, and
rebases the best-checkpoint pointer. Never run the same output/W&B run on two
computers simultaneously. New velocity inputs require a new run.
