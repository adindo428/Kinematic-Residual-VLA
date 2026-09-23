# Scratch Full DOM Baseline vs Residual DynamicVLA MVP

This MVP is only about full-parameter DOM fine-tuning for two models:

- baseline DynamicVLA, with residual flow disabled
- residual DynamicVLA, with kinematic residual flow enabled

Both models train on DOM from scratch, not from a previous DynamicVLA policy
checkpoint. In practice this means the scratch configs leave
`POLICY.CHECKPOINT` empty, the training launchers default to `AUTO_RESUME=0`,
and no `--ckpt` is passed unless you explicitly override it.

The default strict speed split is:

```text
train/IID: mean object speed <= 0.10 m/s
test/OOD:  mean object speed >= 0.18 m/s
```

The selected split file is:

```text
configs/dom_speed_ood_0p10_0p18_split.json
```

The expected workflow is:

```text
CPU preprocess train split -> GPU full fine-tune baseline and residual
CPU prefetch eval split(s) -> GPU cached-only IID/OOD evaluation
```

## Environment

Use a DynamicVLA PyTorch env for this MVP.

```bash
module load cuda/12.3.0
module load gcc/12.3.0
conda create -n dynamicvla python=3.10 -y
conda activate dynamicvla
```

On Palmetto/Rocky 8 nodes, use this install path from the DynamicVLA tree:

```bash
module load cuda/12.3.0
module load gcc/12.3.0
conda activate dynamicvla
cd /scratch/aparame/Dynamic_VLA/third_party/DynamicVLA
python -m pip install --upgrade pip
conda install -c conda-forge "ffmpeg<8" -y
python -m pip uninstall -y torch torchvision torchcodec torchaudio
python -m pip install --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.7.1 torchvision==0.22.1 torchcodec==0.5.0
python -m pip install nvidia-npp-cu12
python -m pip install rerun-sdk==0.23.4
python -m pip install lerobot==0.3.3 --no-deps
python -m pip install -r requirements_rocky8.txt
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"
```

The `rerun-sdk` dependency warning is expected on these Rocky 8 nodes:

```text
lerobot 0.3.3 requires rerun-sdk<0.23.0,>=0.21.0, but you have rerun-sdk 0.23.4
```

Keep `rerun-sdk==0.23.4` there. Older `rerun-sdk==0.21/0.22` wheels usually
require a newer glibc than Rocky 8 provides.

If `python run.py` says `ModuleNotFoundError: No module named 'cv2'`, the
requirements install stopped early. Re-run the install commands above; `cv2`
comes from `opencv-python-headless<4.12`.

If `python run.py` fails with:

```text
ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
```

repair the HF Hub package with:

```bash
python -m pip install --upgrade --force-reinstall \
  "huggingface-hub[cli,hf-transfer]>=1.3.0,<2.0"
```

If `torchcodec` fails with:

```text
libnppicc.so.12: cannot open shared object file
```

install NPP and keep it on the loader path:

```bash
python -m pip install nvidia-npp-cu12
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"
```

## Full Fine-Tune Configs

The two default scratch DOM configs are:

```text
configs/dynamicvla_dom_scratch_speed_ood_0p10_0p18.yaml
configs/residual_dynamicvla_dom_scratch_speed_ood_0p10_0p18.yaml
```

Both configs:

- read `hzxie/DOM`
- use `configs/dom_speed_ood_0p10_0p18_split.json`
- use `observation.environment_state[..., 6:9]` for object speed splitting
- require both DOM camera streams, robot state, environment state, and action
- leave `POLICY.CHECKPOINT` empty
- set `USE_CHECKPOINT_NORMALIZATION: false`
- run for `30` epochs
- use batch size `16`, gradient accumulation `16`, and learning rate `5e-5`
- full-tune all VLA/VLM parameters with:

```yaml
FREEZE_VISION_MODEL: false
FREEZE_CONNECTOR: false
FREEZE_TEXT_MODEL: false
```

The baseline config sets:

```yaml
ENABLE_RESIDUAL_FLOW: false
```

The residual config sets:

```yaml
ENABLE_RESIDUAL_FLOW: true
RESIDUAL_DRIFT_MODE: trajectory
RESIDUAL_ACTION_DT: 0.04
RESIDUAL_NORMALIZE_DRIFT: true
RESIDUAL_CENTER_DRIFT: true
RESIDUAL_ESTIMATE_DRIFT_MEAN: true
```

## 1. CPU Preprocess Train Split

Submit the CPU-only training-split preprocessing job:

```bash
cd /scratch/aparame/Dynamic_VLA/third_party/DynamicVLA
sbatch slurm/preprocess_full_finetune_speed_ood_pair.sbatch
```

This sbatch does not request a GPU. It sets:

```text
RUN_PREFETCH=1
RUN_TRAIN=0
RUN_OFFLINE_COMPARE=0
```

and calls:

```bash
bash scripts/train_full_finetune_speed_ood_pair.sh
```

The script defaults to the scratch configs:

```text
baseline config: configs/dynamicvla_dom_scratch_speed_ood_0p10_0p18.yaml
residual config: configs/residual_dynamicvla_dom_scratch_speed_ood_0p10_0p18.yaml
```

The preprocessing pass resolves the DOM speed split, downloads the parquet data
needed for speed statistics, warms the selected train/IID episodes in the
LeRobot cache, and writes speed statistics under:

```text
/scratch/aparame/.cache/huggingface/lerobot/hzxie/DOM/.cache/dynamicvla_episode_speed_stats.json
```

This can take many hours on cluster nodes with slow Hugging Face throughput. It
is resumable: if it stops or is cancelled, submit the same sbatch again.

Useful prefetch controls already set by the sbatch:

```text
DYNAMICVLA_PREFETCH_STATUS_INTERVAL=300
DYNAMICVLA_PREFETCH_RETRIES=5
DYNAMICVLA_PREFETCH_CHUNK_PATTERN_THRESHOLD=1000
```

## 2. GPU Full Fine-Tune Baseline And Residual

After the CPU preprocessing job finishes, submit one GPU training job:

```bash
cd /scratch/aparame/Dynamic_VLA/third_party/DynamicVLA
sbatch slurm/train_full_finetune_speed_ood_pair_h100.sbatch
```

Use the launcher that matches the available GPU partition:

```bash
sbatch slurm/train_full_finetune_speed_ood_pair_h200.sbatch
sbatch slurm/train_full_finetune_speed_ood_pair_h100.sbatch
sbatch slurm/train_full_finetune_speed_ood_pair_a100.sbatch
```

The launchers call:

```bash
bash scripts/train_full_finetune_speed_ood_pair.sh
```

with training enabled and prefetch disabled by default. The script trains the
baseline first, then the residual model:

```text
baseline config: configs/dynamicvla_dom_scratch_speed_ood_0p10_0p18.yaml
residual config: configs/residual_dynamicvla_dom_scratch_speed_ood_0p10_0p18.yaml
```

Expected checkpoint directories:

```text
runs/checkpoints/dynamicvla_dom_scratch_speed_ood_0p10_0p18/
runs/checkpoints/residual_dynamicvla_dom_scratch_speed_ood_0p10_0p18/
```

The training launchers default to:

```text
AUTO_RESUME=0
```

so they do not resume from existing checkpoint directories unless you explicitly
override them. Keep these unset for a clean scratch run:

```text
BASELINE_CKPT
RESIDUAL_CKPT
```

Only set `AUTO_RESUME=1`, `BASELINE_CKPT`, or `RESIDUAL_CKPT` when you
intentionally want checkpoint-based training.

## 3. CPU Prefetch Evaluation Splits

After training, warm the held-out evaluation data on a CPU node before using a
GPU for evaluation:

```bash
cd /scratch/aparame/Dynamic_VLA/third_party/DynamicVLA
sbatch slurm/postprocess_full_finetune_eval_splits.sbatch
```

This sbatch calls:

```bash
python scripts/prefetch_eval_splits.py \
  --cfg configs/residual_dynamicvla_dom_scratch_speed_ood_0p10_0p18.yaml \
  --splits ood
```

By default it warms only the OOD/test split. To warm both IID/train and OOD/test:

```bash
EVAL_SPLITS="iid ood" sbatch slurm/postprocess_full_finetune_eval_splits.sbatch
```

Use this CPU prefetch job for slow DOM video downloads. The evaluation sbatch is
configured for cached-only evaluation by default, so it can score already cached
complete episodes without waiting for the entire DOM eval split to finish.

## 4. GPU Evaluate Baseline vs Residual

Once the requested checkpoint directories exist, submit:

```bash
cd /scratch/aparame/Dynamic_VLA/third_party/DynamicVLA
sbatch slurm/evaluate_full_finetune_iid_ood_h100.sbatch
```

This sbatch uses one H100 and calls:

```bash
python scripts/evaluate_baseline_residual_iid_ood.py \
  --cfg configs/residual_dynamicvla_dom_scratch_speed_ood_0p10_0p18.yaml \
  --baseline-checkpoint runs/checkpoints/dynamicvla_dom_scratch_speed_ood_0p10_0p18 \
  --residual-checkpoint runs/checkpoints/residual_dynamicvla_dom_scratch_speed_ood_0p10_0p18 \
  --output runs/proactive_vla/offline_compare/scratch_strict_0p10_0p18_iid_ood_report.json \
  --splits iid ood \
  --batch-size 1 \
  --max-batches 20000 \
  --cached-only \
  --max-cached-episodes 20000 \
  --shuffle-cached-episodes \
  --cached-episode-seed 1234
```

The output report is:

```text
runs/proactive_vla/offline_compare/scratch_strict_0p10_0p18_iid_ood_report.json
```

Set the episode/batch cap in `slurm/evaluate_full_finetune_iid_ood_h100.sbatch`:

```bash
MAX_BATCHES=${MAX_BATCHES:-20000}
MAX_CACHED_EPISODES=${MAX_CACHED_EPISODES:-20000}
```

You can also override it at submit time:

```bash
MAX_BATCHES=500 \
MAX_CACHED_EPISODES=500 \
sbatch slurm/evaluate_full_finetune_iid_ood_h100.sbatch
```

Useful evaluation overrides:

```bash
EVAL_SPLITS="ood" sbatch slurm/evaluate_full_finetune_iid_ood_h100.sbatch

EVAL_OUTPUT=runs/proactive_vla/offline_compare/ood_only_report.json \
EVAL_SPLITS="ood" \
sbatch slurm/evaluate_full_finetune_iid_ood_h100.sbatch

BASELINE_CKPT=/path/to/baseline/checkpoint \
RESIDUAL_CKPT=/path/to/residual/checkpoint \
sbatch slurm/evaluate_full_finetune_iid_ood_h100.sbatch
```

## Can Evaluation Run On A Halfway-Collected Dataset?

Yes. The evaluation sbatch now defaults to cached-only mode:

```text
EVAL_CACHED_ONLY=1
MAX_CACHED_EPISODES=20000
EVAL_SHUFFLE_CACHED_EPISODES=1
```

In cached-only mode, `scripts/evaluate_baseline_residual_iid_ood.py` filters
the configured IID/OOD split to episodes whose required parquet files and both
camera videos are already present in the local LeRobot cache. It shuffles that
cached episode list deterministically before applying `MAX_CACHED_EPISODES`, so
the eval does not always use the first cached files.

This means you can run:

```bash
sbatch slurm/evaluate_full_finetune_iid_ood_h100.sbatch
```

while the full eval-data prefetch is still incomplete, as long as there are
enough complete cached episodes for the requested split. Keep
`EVAL_CACHED_ONLY=1` if you want the job to avoid downloading missing files.

Set `EVAL_CACHED_ONLY=0` only when you want evaluation to restore the older
behavior and fetch missing files during dataset construction:

```bash
EVAL_CACHED_ONLY=0 sbatch slurm/evaluate_full_finetune_iid_ood_h100.sbatch
```

## Metrics To Watch

The IID/OOD report now includes both horizon-minimum and executed-final metrics.
For judging whether residual VLA helps, prefer these fields in
`cross_split_summary`:

```text
final_success_rate
final_object_distance
action_mse
ee_tracking_mse
mean_object_velocity_norm
```

The older `success_rate` is still present, but it uses the minimum object
distance across the predicted horizon. It can look optimistic if the trajectory
briefly passes near the object and then ends poorly.

## Mental Model

The baseline full fine-tune learns the complete action flow:

```text
pixels + language + robot state -> full action flow
```

The residual full fine-tune adds an analytical moving-object drift term:

```text
object velocity -> Phi(v_hat) -> action-space drift
```

The residual model learns the correction around that drift:

```text
full action flow = analytical drift + learned residual
```

For this MVP, the residual drift fills action XYZ channels. Rotation and gripper
remain learned by the VLA.
