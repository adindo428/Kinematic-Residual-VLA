# Tracking model training: 75 epochs

This guide documents the frozen training family in [`source/`](source/). Use its model and trainer code for these checkpoints; the repository's `third_party/DynamicVLA` tree serves later DOM training work and has a different training/checkpoint contract. Collection and export instructions are in the [collector guide](../../third_party/Data_Collection_Dynamic_Objects/README.md).

## Dataset and labels

Use [`adi2440/dom-franka-tracking`](https://huggingface.co/datasets/adi2440/dom-franka-tracking) at revision `5c46a9f26b5f85edafa53885187cc03fa8706219`. The 100 accepted ten-second episodes contain three synchronized RGB cameras at 25 Hz. Policy training uses `opst_cam` and `wrist_cam`, two observation times `[-2, 0]`, the measured end-effector pose, and language. The dataset also records robot-base object position, orientation, and velocity. The [fixed episode split](source/configs/dom_franka_tracking_split.json) is 80 train, 10 validation, and 10 test; select the episode IDs explicitly because the LeRobot metadata lists all episodes as `train` for format compatibility.

Each action label is the **measured** tool pose five recorded frames (+0.2 s) ahead plus open gripper. The stored action has seven values: XYZ, XYZ Euler angles, and gripper. Training uses delta actions relative to the current measured end-effector state. The collector records the issued controller commands separately, so they are not interchangeable with action labels. Keep normalization statistics restricted to the training episodes.

## Matched recipes

| Model | Recipe | Explicit neural velocity | Analytic drift |
|---|---|---|---|
| DOM-VLA | [`tracking_dom75_full20_baseline.yaml`](source/configs/tracking_dom75_full20_baseline.yaml) | No | No |
| DOM-VLA+Velocity | [`dynamicvla_dom_tracking_velocity75.yaml`](source/configs/dynamicvla_dom_tracking_velocity75.yaml) | Yes | No |
| KR-VLA-Coordinate | [`tracking_dom75_full20_coordinate.yaml`](source/configs/tracking_dom75_full20_coordinate.yaml) | No dedicated velocity input | Yes |
| KR-VLA-Coordinate+Velocity | [`tracking_dom75_full20_coordinate_velocity.yaml`](source/configs/tracking_dom75_full20_coordinate_velocity.yaml) | Yes | Yes |
| Historical KR-VLA-Field | [`tracking_dom75_full20_residual.yaml`](source/configs/tracking_dom75_full20_residual.yaml) | No dedicated velocity input | Yes, in the flow field |

The matched settings are 75 full epochs, 20 predicted/executed actions, batch 8 with 44 gradient-accumulation steps (effective batch 352), seed 42, full fine-tuning, AdamW learning rate `1e-4`, a cosine schedule with 1,000 **optimizer-step** warmup steps, and selection of the lowest validation action-MSE checkpoint. Validation covers the full validation split each epoch; the test split is reserved for evaluation. Images receive the same configured augmentation. The initial model weights come from `hzxie/dynamic-vla-DOM`.

The Coordinate correction projects **current robot-base XYZ object velocity** into the action chunk. The recipe uses a trajectory drift with `RESIDUAL_ACTION_DT=0.04`, drift scale 1, action-space normalization, and a mean correction estimated from training data. Its drift grid starts at zero; dataset action labels start at +0.2 s. Preserve these conventions when reproducing the historical models, and distinguish them when interpreting the physical approximation. The velocity-conditioned variants put standardized current velocity into state-projection slots 6:9. They reuse the baseline family's training-split state/action normalization; velocity statistics come from training episodes. No future object trajectory enters the policy.

## Prepare the environment and data

Use a policy environment with Python 3.10, Torch 2.7.1, LeRobot 0.3.3, and Transformers 5.2.0. Install a matching video decoder (FFmpeg and TorchCodec). Keep the Isaac Sim/Isaac Lab evaluation environment separate. If the Hugging Face dataset requires access, authenticate outside Git.

The frozen training source expects the complete pinned dataset in LeRobot's cache. For example:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
unset LEROBOT_HOME
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    "adi2440/dom-franka-tracking",
    repo_type="dataset",
    revision="5c46a9f26b5f85edafa53885187cc03fa8706219",
    local_dir=os.environ["HF_LEROBOT_HOME"] + "/adi2440/dom-franka-tracking",
)
PY
```

Include both camera video trees and metadata. Before training, inspect the dataset split file and verify that the selected training examples exclude the validation and test IDs.

## Launch or resume

The frozen YAML files retain historical Palmetto output and normalization-reference paths. Create a local copy for a new machine; keep the model and data settings unchanged. Train the baseline first when creating a fresh matched family, then use its selected checkpoint as the normalization reference for the velocity variants.

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

For another model, substitute its recipe, experiment name, and output paths. The helper disables W&B by default; pass `--wandb` to enable it after configuring the account. `DVLA_OUTPUT_ROOT` overrides trainer output paths, so unset it when using an explicit run root. Training batch 8 has not been validated on an RTX 3080. The [frozen Slurm launchers](source/slurm/) preserve historical A100/H100/H200 commands and cluster paths; check every path and resource request before reuse.

Starting fresh uses no `-p` flag. The trainer's `-p DIRECTORY` loads existing weights for resumption; inspect its optimizer/epoch handling before relying on it for an exact continuation. `--init-ckpt DIRECTORY` starts a new training run from specified weights at epoch zero. Keep the same dataset, split, normalization, and recipe across a genuine resume. Checkpoint readiness and hashes can be checked with [`verify_tracking_training.py`](source/scripts/verify_tracking_training.py).

## Evaluate the trained models

Copy each checkpoint with its matching `config.json`, validation-selected weights, and `training_ready.json` marker. The [tracking evaluation guide](README.md) documents the historical checkpoint locations, strict loader, one-shard pilot, nine stress dimensions, 90-shard full run, and the scoring convention. The current evaluation runs all four models on matched initial states. Velocity comes from the simulator; there is no camera-derived velocity estimate in this comparison.

The tracking score is each episode's fraction of requested time within 3 cm of a target 10 cm above the moving object's visual top. It measures hover tracking, not manipulation success. Keep the raw telemetry, per-episode scores, split, checkpoint hashes, and full protocol with any published result.
