# Fresh original-DOM post-training on all released real tasks

Run: `scratch/real_robot/finetuning-runs/fresh-dom-causal-full-real-20261114/`.
KRV (`krv100`, DOM+KR) runs on the workstation RTX 4090. DOMv (`domv`) has moved
to Palmetto job 20504544 on two V100s; see README_PALMETTO_DOMV.md. They initialize
independently with separate optimizer/EMA state. The preceding SmolVLA evaluation
was stopped through its controller STOP file before GPU training began. These
training processes do not control the robot.

## Initialization and data

Both methods strictly load the same original released `hzxie/dynamic-vla-DOM`
weights, revision `c9609d94c5940eb42436a31927d7e80ae454279e`.
Weight SHA256: `e3da8b1e0326d7a3ee6873f7fa17cdf7015201251901be5107377cb851ce3af8`.
No local DOM 20K, KRV, DOMv, SmolVLA or prior real-robot fine-tune initializes either
run. Original released weights are the user's clarified meaning of “from scratch.”
All 430,155,232 native model parameters are trainable. Architecture, two camera
views with observation offsets [-2, 0], 20-action chunks and 25 Hz action timing
are preserved. Fresh train-split normalization replaces simulation statistics only
after strict base loading and is serialized in each deployable snapshot.

The frozen derivative contains all 859 episodes in the latest released real
collection: static and moving orange, striped ball and green marker. Existing
splits are preserved: 706 train, 80 validation, 73 test. There are 101,781 eligible
training anchors and 12,020 validation anchors. Source episodes are unique across
splits; finite labels, monotonic clocks, episode lengths and source/weight hashes
were checked. Test demonstrations do not enter optimization or model selection.
This experiment uses the released real dataset, not a new synthetic DOM training
run. The retained dataset directory has a historical “continuation” name; its
use does not load continuation model weights.

Recorded instructions are returned verbatim for every episode. In particular,
the orange prompt remains `Pick up the rolling orange and place it in the box.`
Task/static-source/phase frequencies receive capped inverse-square-root sampling
weights, with unknown phases capped at weight one. One epoch draws exactly the
training-anchor count with replacement; it is not a claim to visit every anchor.

## What was learned from DynamicVLA

The [paper's training scheme](https://arxiv.org/html/2601.22153v1) states that
real-robot post-training uses the mid-training flow objective. Its
[released training code](https://github.com/hzxie/DynamicVLA/blob/master/core/train.py)
retains new-dataset normalizers when loading a checkpoint. The
[default configuration](https://github.com/hzxie/DynamicVLA/blob/master/configs/dynamicvla.yaml)
uses AdamW, cosine decay, warm-up and image photometric augmentation. These
sources do not establish an optimal learning rate for this robot or dataset.

This run restores native flow-only training and a uniform valid action horizon;
the previous fourfold early-step weighting and sampled reconstruction auxiliary
are disabled. The local native loss continues to mask padded timesteps and
supervises seven physical channels. This physical-channel mask is an existing
local adaptation rather than a verbatim copy of upstream padded-channel loss.
There is no added velocity-innovation training target.

Our explicit adaptations for this dataset/GPU are:

| Setting | New recipe |
| --- | --- |
| Effective batch | 64: local KRV uses 8 × 8; remote DOMv uses 2 × 16 × 2 GPUs; exact final-window weighting |
| Learning rates | Action/expert 3e-5; pretrained vision/language backbone 3e-6 |
| Optimizer | AdamW, betas (0.9, 0.95), epsilon 1e-8, weight decay 1e-10; bias/norm without decay |
| Schedule | 1,000 optimizer-update warm-up, cosine decay to 10% of peak |
| Duration | Maximum 60 epochs; earliest stopping at 15, patience 12 completed epochs |
| Gradient handling | Norm clipping at 1; finite checks; local KRV BF16 autocast, remote DOMv FP16 autocast/GradScaler with FP32 masters |
| EMA | Trainable-parameter EMA, target decay 0.999 with warm start |
| Images | Shared photometric draw across views/history: brightness/contrast/saturation [0.8, 1.2] |
| Geometric augmentation | None; calibrated pixel/action correspondence preserved |
| Input sensor noise | Disabled; expert labels unchanged |

Quality takes priority over speed. The runtime still uses bounded decoder caches,
persistent workers, batched frame decoding and vectorized photometric operations;
there is no artificial throttling.

## Causal KR gate and velocity innovation

The real-data adapter previously returned a constant gate whenever quality flags
were present, bypassing causal prefix replay. It now honors explicit
`phase_gate_mode='causal_rules'` for this run, while keeping older constant-gate
runs unchanged. The same causal rule implementation is used by offline replay
and live evaluation. Replay reads current/past object/EE inputs and only the
previous gripper command. Changing the current/future command or future
observations cannot change an anchor's gate; regression tests cover this.

Compact gate states are cached and hash-checked for 786 train/validation episodes.
Only an already initiated gate fade is projected into the future chunk. No future
grasp label is inserted into a chunk. Existing tracking-quality masking remains;
static and inferred-held quality annotations retain their original status and
are not promoted to measured observations. The replay audit shows attachment
confirmation is conservative on these noisy inputs (only 220 fade and 11 carry
frames); quality masks also suppress unavailable/neutral object velocity. This
is a data/estimator limitation, not evidence that a newly trained model succeeds.

Live evaluation now accepts the trained causal gate and thresholds. The controller
maintains gate history using the previous applied, calibrated gripper command and
publishes the gate alongside the observation; the worker validates it rather than
inventing one. New run contracts default velocity innovation **on** for KRV and
DOMv, while old run defaults remain unchanged. It applies the existing causal,
rate/magnitude-bounded physical XYZ correction once during execution. Training
weights and targets do not receive this correction. IK/governor limits are not
changed by this training recipe.

## Selection, checks and monitoring

An original-base physical-error baseline is measured before optimization. Every
completed epoch runs full held-out flow validation and evaluates raw and EMA
weights on the same 512 fixed task/source/phase-balanced validation anchors with
three fixed independent noise seeds. Position, SO(3), and gripper scores are
averaged across samples; there is no best-of-N selection. The better raw/EMA
snapshot is selected by the configured combined physical score. Best and last
completed checkpoints, epoch histories and the last two snapshots are retained.
Native selected snapshots are deployable; `training_state.pt` separately retains
raw weights, optimizer, EMA and RNG state for resuming at an epoch boundary.

A STOP file is checked at an optimizer boundary and prevents the next queued
method from starting. Partial epochs are excluded from evaluation; resume uses
the last completed epoch. No exact mid-epoch replay is claimed. Resume a method
with `train_dom_posttraining.py --resume` and the same contract/output/family,
after removing an intentionally created STOP file.

Both disposable GPU preflights passed strict initialization, exact native sampling
parity, finite gradients, optimizer and EMA updates, at about 11.9 GiB peak
allocated memory. Their changed test weights were discarded. Production training
loads the original weights again. Unit/regression checks cover gate causality,
quality-flag bypass, sample/gradient accumulation tails, seeded epoch sampling,
EMA and native deployment contracts. No physical success claim follows from them.

```bash
cat scratch/real_robot/finetuning-runs/fresh-dom-causal-full-real-20261114/queue.json
cat scratch/real_robot/finetuning-runs/fresh-dom-causal-full-real-20261114/krv100/status.json
tail -f scratch/real_robot/finetuning-runs/fresh-dom-causal-full-real-20261114/logs/krv100.log
```

After completed checkpoints exist, the evaluator reads this run's recorded
instruction, saved normalization, causal gate configuration and innovation
defaults. It starts paused:

```bash
.venv/bin/python -m real_robot.evaluate --run scratch/real_robot/finetuning-runs/fresh-dom-causal-full-real-20261114 --model krv100 --checkpoint best --task orange --execute --max-trials 3 --trial-seconds 30
.venv/bin/python -m real_robot.evaluate --run scratch/real_robot/finetuning-runs/fresh-dom-causal-full-real-20261114 --model domv --checkpoint best --task orange --execute --max-trials 3 --trial-seconds 30
```

The controller remains asynchronous with elapsed-prefix dropping. This recipe
uses native DynamicVLA continuous inference/LAAS; it does not silently substitute
the SmolVLA RTC adapter. Offline errors and physical closed-loop success must be
reported separately; training does not fix calibration or control-lag faults.

## DOMv moved to Palmetto, November 14

DOMv is now Slurm job **20504544**, requesting two V100s independently from the
same original DOM base and frozen real dataset. The local queue now contains only
the already-running KRV child; its training was not interrupted. See
[Palmetto setup, policy, monitoring and precision differences](README_PALMETTO_DOMV.md).
The remote job runs a disposable DDP/FP16 GPU check before production, retaining
global effective batch 64 and the same held-out validation protocol.
