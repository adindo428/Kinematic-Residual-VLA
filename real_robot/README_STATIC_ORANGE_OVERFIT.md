# Static-orange KRV memorization experiment

Run: `scratch/real_robot/finetuning-runs/static-orange-krv-overfit100-20261113`.

This isolated diagnostic initializes from the released full-data pretrained KRV
checkpoint `krv100-full-epoch-003-u000196707`, not a failed real-robot fine-tune.
Its config and weights are SHA256-pinned before strict loading. The original
checkpoint, prior runs, and source dataset are preserved.

All 100 static-orange episodes are used for training (14,987 eligible anchors).
The copied manifest preserves each episode's original split. The original source
80/10/10 split is unchanged, but this diagnostic deliberately includes its former
holdouts. Metrics called validation in the generic trainer are **training-set
memorization probes**, not independent generalization estimates.

Native KRV coordinate residual flow, constant phase gate with its trained quality
mask, velocity conditioning, [-2,0] observation history, two RGB cameras,
20-action horizon and 25Hz action clock are retained. State/action/velocity
normalization is rebuilt from this isolated training dataset and saved in each
checkpoint. Velocity innovation is not a training operation.

Speed settings: bf16, fused AdamW, batch 16, four persistent loader workers,
TF32-enabled CUDA libraries, and a lossless uint8 decoded-image cache on SSD.
The cache matched the regular adapter exactly on checked samples; it changes
neither images nor action labels. Batch 16 benchmarked at about 44.6 samples/s,
including physical auxiliary-loss updates; batch 20 exceeded GPU memory.
No image augmentation, proprioceptive perturbation, weighted resampling, or
weight decay is used. All model parameters remain trainable. The native flow
loss and the existing decoded-action auxiliary loss are retained.

Learning rate: 1e-5, 100-step warmup, plateau reduction to a 1e-6 floor.
Training caps at 100 epochs, with patience 20 on the 64-anchor memorization probe
(one fixed noise seed). This is an operational cap, not an assertion of convergence.
Each epoch saves a completed checkpoint; retention keeps the best and latest two.
Recovery checkpoints are written first at step 500 and every 2,000 steps thereafter.
Discarded checkpoint provenance is archived to preserve resume patience.

Monitor from the repository root:

```bash
tail -f scratch/real_robot/finetuning-runs/static-orange-krv-overfit100-20261113/training.log
cat scratch/real_robot/finetuning-runs/static-orange-krv-overfit100-20261113/krv100/status.json
```

Inspect `selection.json`, `launch_contract.json`, `benchmark.json` and
`launcher.json` for episode mapping, configuration, measured speed and process
identity. No robot or camera devices are opened for training.

## Replacement run: original 80/10/10 split and unchanged prompt

The active replacement is
`scratch/real_robot/finetuning-runs/static-orange-krv-overfit80-20261113`.
It follows the user's split selection: 80 training, 10 validation, and 10 test
episodes. It initializes fresh from the original pretrained full-data KRV;
weights from the superseded all-100 experiment are not reused. Test episodes
are not used for optimization or checkpoint selection. The lossless image cache
is shared only after checking identical episode-to-source mappings.

Both training and evaluation use the recorded prompt verbatim:
`Pick up the rolling orange and place it in the box.` No generated wording
variants are used in this replacement. The selected checkpoint records
`instruction_mode=recorded`, and the evaluator checks that mode before launch.

The earlier all-100 attempt stopped with a spawned data-loader compatibility
error after the prompt-code update. Its artifacts remain under its original
folder for provenance; it is not the active run. The replacement includes a
spawned-worker loader check for both train and validation before launch.
