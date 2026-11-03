# Training workflows

## Required checkpoint retention

For all future training, retain **latest** and **best-validation** checkpoints independently. Latest is the complete state for an exact resume; best is the complete saved state achieving the best finite held-out validation score. For DOM/VLA, use minimum physical horizon20 action MSE unless a study explicitly declares another metric. Keep best protected when latest advances, save the latest state at graceful stops and completion, and preserve both during cleanup. Immutable generation pointers may share a generation when both select the same state. Best is pending until the first successful validation; training logs or the final checkpoint cannot reconstruct an overwritten historical best.

Both states retain model, optimizer, scheduler, per-rank RNG, counters/data cursor, budget and provenance. See [standing training requirements](../AGENTS.md). Introduce this policy through fresh frozen source for future submissions; do not modify source under active or queued jobs.

Use separate output directories for every experiment. The tracking trainer and current DOM100 trainer are preserved separately because their normalization, labels, checkpoint formats and provenance differ. Local evaluation of the official baseline does **not** require either workflow.

Live jobs and historical submissions are tracked in the [KRVLA training run register](https://chatgpt.com/space/page_4ff536c9060081919225467494390460), refreshed hourly. Agent maintenance instructions and the local registry are in [maintenance/training_run_register](../maintenance/training_run_register/README.md).

## Tracking training: 75 epochs

Source: `experiments/tracking75/source`. Dataset: [adi2440/dom-franka-tracking](https://huggingface.co/datasets/adi2440/dom-franka-tracking), revision `5c46a9f26b5f85edafa53885187cc03fa8706219`. It contains 100 expert tracking episodes; `configs/dom_franka_tracking_split.json` preserves the 80/10/10 split. Use Hugging Face authentication if the dataset requires access; credentials stay outside Git.

Use the policy environment described in [the evaluation guide](../evaluation/official_dom/README.md). FFmpeg and the matching TorchCodec installation are required for decoding. Download the entire pinned dataset (including both camera video trees and metadata) to LeRobot's expected cache:

```bash
conda activate krvla-policy
export HF_HOME="$HOME/.cache/huggingface"
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
unset LEROBOT_HOME
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download('adi2440/dom-franka-tracking', repo_type='dataset',
    revision='5c46a9f26b5f85edafa53885187cc03fa8706219',
    local_dir=os.environ['HF_LEROBOT_HOME']+'/adi2440/dom-franka-tracking')
PY
```

| Method | Recipe |
|---|---|
| DOM-VLA | `tracking_dom75_full20_baseline.yaml` |
| DOM-VLA+Velocity | `dynamicvla_dom_tracking_velocity75.yaml` |
| KR-VLA-Coordinate | `tracking_dom75_full20_coordinate.yaml` |
| KR-VLA-Coordinate+Velocity | `tracking_dom75_full20_coordinate_velocity.yaml` |
| Historical KR-VLA-Field | `tracking_dom75_full20_residual.yaml` |

The matched recipes use 75 full epochs, two observations `[-2,0]`, twenty predicted/executed actions, effective batch 352 (8 × 44 accumulation), AdamW LR `1e-4`, cosine schedule with 1,000 optimizer-step warmup, seed 42, full fine-tuning, and validation action MSE checkpoint selection. Preserve these settings when reproducing the comparison. The Coordinate+Velocity neural residual is explicitly velocity-conditioned.

Convert only workstation paths/logging into a new recipe (do not edit historical recipes in place):

```bash
export KRVLA_REPO=/absolute/path/to/Kinematic-Residual-VLA
cd "$KRVLA_REPO/experiments/tracking75/source"
python scripts/local_training_config.py \
  --recipe configs/tracking_dom75_full20_coordinate_velocity.yaml \
  --output "$HOME/krvla-evaluation/configs/coordinate_velocity.yaml" \
  --run-root "$HOME/krvla-evaluation/training/coordinate_velocity" \
  --normalization-reference "$HOME/krvla-evaluation/checkpoints/baseline"
python scripts/train_tracking_velocity.py \
  -c "$HOME/krvla-evaluation/configs/coordinate_velocity.yaml" \
  -e tracking_dom75_full20_coordinate_velocity -g 0
```

Velocity recipes reuse the baseline's training-split state/action normalizers; the reference directory must contain its actual validation-selected weights. Do not replace these with official checkpoint statistics. For a fresh matched family, train the baseline first, then use its checkpoint as the reference for the velocity variants. Baseline/Coordinate recipes compute train-split normalization. The helper disables W&B by default; add `--wandb` to enable it and configure your account. `DVLA_OUTPUT_ROOT` overrides output paths in the trainer; unset it when using the helper's explicit run root.

For each other method, substitute the recipe, output/config paths and experiment name. Launch fresh training with no `-p`. The trainer's `-p DIRECTORY` resumes existing weights; read its checkpoint/epoch handling before restarting. `--init-ckpt DIRECTORY` loads an initialization and starts from epoch zero. `--initial-checkpoint` on the recipe helper can select a locally copied, pinned initial DOM checkpoint instead of the recipe's Hub reference. A resumed run must retain the same training/data settings.

The 3080 is primarily the evaluation target. Training batch 8 may not fit; changing microbatch/accumulation would require a newly documented recipe, so do not assume this matched recipe has been validated on that GPU. Historical Palmetto launchers remain in the source tree for the existing A100/H100/H200 training workflows.

## DOM training: 100 epochs

Source: `third_party/DynamicVLA`. The current trainer supports these six matched methods:

| Variant | Neural velocity | Coordinate correction |
|---|---|---|
| `dom100` | no | none |
| `kr100` | no | constant |
| `domv100` | yes | none |
| `krv100` | yes | constant |
| `krp100` | no | causal phase gate |
| `krvp100` | yes | causal phase gate |

Full details and deployment history are in [DOM100.md](../third_party/DynamicVLA/DOM100.md). The current recipe has prediction/execution horizon 20, a 100-epoch cap, effective batch 352, shared initialization and state/action normalization, training-only velocity/drift statistics, deterministic grouped splits, and exact checkpoint generations with optimizer/scheduler/RNG restoration. The distributed campaign selected fixed LR `3e-5` and validation-based early stopping after at least five epochs: three validations without a relative improvement of at least 0.5%. Small gains accumulate against the significant-best reference; best-checkpoint selection still accepts any lower MSE. The frozen recipe, not the name DOM100 alone, specifies whether early stopping is enabled. The historical single-GPU workflow includes LR/sampler pilots; do not confuse it with that active fixed-rate campaign.

### Prepare a new family on a machine with enough storage

```bash
cd "$KRVLA_REPO/third_party/DynamicVLA"
export FAMILY="$HOME/krvla-dom100"
export DOM_DATA="$HOME/.cache/huggingface/lerobot/hzxie/DOM"
python scripts/prepare_dom100.py --family-root "$FAMILY" \
  --dataset-root "$DOM_DATA" --prefetch
```

Preparation resolves immutable dataset/model revisions and records them, verifies files, splits duplicate trajectories together, and freezes training statistics and causal phase prefixes. `--dataset-revision` / `--model-revision` select explicit revisions when needed. Use `--available-only` instead of `--prefetch` only if intentionally training on complete cached episodes: it freezes and reports excluded episodes, producing a **subset**, not the complete published dataset. Preparation refuses to overwrite a frozen family. Any source/recipe/data change requires a newly prepared family rather than copying an old fingerprint.

### Single-GPU workflow

```bash
python scripts/dom100.py profile --family-root "$FAMILY" --output "$FAMILY/profiling/local"
python scripts/dom100.py pilots --family-root "$FAMILY"
python scripts/dom100.py choose --family-root "$FAMILY"
python scripts/dom100.py smoke --family-root "$FAMILY"
python scripts/dom100.py train --family-root "$FAMILY" --variant krv100
python scripts/dom100.py verify --family-root "$FAMILY" --variant krv100
```

Profiling must approve the actual GPU/runtime and all variants, with the required capacity headroom. Do not import a Palmetto GPU profile into a different workstation. Inspect reports after each step; rejected/failed candidates are not permission to bypass checks. The training commands require approved family/profile/recipe/smoke artifacts. Run the remaining variants with distinct run directories within the same frozen family. GPU/CPU memory and data decoding tests are prerequisites, not manipulation-success evidence.

### Four/eight-GPU workflow

The DDP path supports one node with four or eight A100/H100/H200 GPUs and requires a compatible frozen family/recipe. It runs decoded distributed smoke checks before training, keeps global batch 352, and only rank zero writes checkpoints/validation. Exact resume requires the same GPU count. On Palmetto, follow the active family setup and `scripts/submit_dom100_distributed.py` in [DOM100.md](../third_party/DynamicVLA/DOM100.md); do not apply cluster provenance to a new family. The runtime entry point is:

```bash
torchrun --standalone --nproc_per_node=8 scripts/dom100_distributed.py \
  --family-root "$FAMILY" --variant krv100 --command smoke-train --microbatch 44
```

Four GPUs use two accumulation steps per rank; eight use one. This command is unsuitable for a single RTX 3080. The source includes Slurm submission, orderly-stop/requeue, profiling and verification tools. Training checkpoints and manifests remain in the family root; keep them together when transferring experiments.
