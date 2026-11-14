# Next real-robot fine-tuning batch

Prepared recipes are in `scratch/real_robot/robust-finetuning-v4-grasp-common-dom/`.
These use the new grasp-annotated derivative and supersede v3's earlier data and
v2's mixed simulation/DOM initialization. No full training
or new W&B run was started during this preparation. Disposable optimizer checks
for DOM, DOMV and KRV100 passed; their updated weights were discarded.

## Grasp annotation update

All 564 episodes retain success supervision: 563 saved outcomes and the operator's
declaration for the first episode whose original outcome file is unavailable.
The new derivative adds supported held intervals in 444 episodes using wrist
retention, measured closure and >=20mm measured lift. Of the resulting held
frames, 3,012 are visually reviewed confirmations (including existing labels),
and 23,245 are explicitly marked automatic success-supported inference.
The automatic labels are not presented as manual ground truth. There were 47
new episode-level visual reviews; an observed unsuccessful lift before a later
successful grasp remains excluded from holding.

The 120 episodes with unresolved pickup timing remain successful demonstrations;
no held timing is invented and their existing usable frames remain available.
The derivative preserves all actions, robot states, object poses, timestamps,
videos, split assignments and earlier tracking-repair provenance. Held velocity
is a neutral model input with `velocity_required=false`, not physical zero motion.
No depth correction is applied again. Low marker grasps are retained, with a
task-level expected-table-contact flag rather than a force/contact timestamp.

This recovers 15,170 previously excluded frames. With the same two-frame history
mask, usable anchors change from **52,215 to 65,689 training**, **6,657 to 7,710
validation**, and **6,528 to 7,576 test**. These counts reflect improved phase
coverage; validation loss on this new population is not directly comparable to
loss on the old mask. Re-evaluate old checkpoints on the new mask for comparison.

The local source is
`scratch/real_robot/training_releases/KRVLA-5fd5ef3e4256-extrapolated-80ms-grasp-v1`.
See `grasp_annotation_report.json`, `annotation_validation.json` and
`dataset_validation.json`. Review decisions and sampled wrist strips are under
`scratch/real_robot/grasp-annotations-success-v2/`. This local derivative has not
been uploaded to Hugging Face. The numerical examples below describe the earlier
dataset/control audit; current phase/anchor/workspace counts are in the v4 audits.

## What the first trials tell us

The current failures mix policy errors and execution errors. The initial model
comparisons are diagnostic, not a controlled closed-loop benchmark: some starts
differed substantially, reset/telemetry failures interrupted trials, and the
Cartesian controller could not realize all requested motion. The improved
bounded IK solver passed numerical checks, but physical equivalence to the
leader/follower command path has not been established.

The numerical dataset audit found these differences at quality/history-eligible
training anchors (not all frames or all future chunk elements):

| Task | Anchors | Command TCP below evaluation's 20 mm floor | Command TCP outside evaluation XYZ workspace | Median command-FK minus measured-FK distance |
| --- | ---: | ---: | ---: | ---: |
| Striped ball | 18,020 | 4.2% | 23.4% | 21.7 mm |
| Green marker | 13,091 | 13.2% | 14.5% | 19.4 mm |
| Orange | 21,104 | 3.6% | 5.9% | 26.4 mm |

The workspace is `[0.20, -0.30, 0.02]` to `[0.58, 0.30, 0.35]` m. Targets outside
it cannot be followed unchanged during evaluation. This is not evidence that
the recorded demonstrations were physically unsafe. Check the intended task
envelope, TCP/contact model and a supervised replay of expert commands through
the inference controller. Do not silently clip the labels, discard grasp phases,
or disable clearance guards to make the numbers agree.

Training command-FK speed p95 is approximately 0.44–0.46 m/s across tasks.
The historical smooth evaluation profile capped TCP motion at 0.15 m/s; measured
moving-marker speeds also exceeded that cap. These are different quantities
(commanded motion, measured object motion and governed robot motion), but the
speed mismatch confounds interception results. Keep the controller profile fixed
within comparisons and report saturation, following error and action age.

The saved static-marker demonstration contains a waiting prefix followed by a
moving-object grasp. That prefix does not establish static-grasp competence.
The fine-tuned model produced motion on the recorded demonstration and smaller
motion in live scenes, so the evidence does not establish a globally constant
policy. Original DOM's larger predicted displacements also did not establish
better directed actions. A replay from a training episode is not held-out
performance.

## Preserve the collection/training/inference contract

| Item | Current contract and verification |
| --- | --- |
| Raw encoders and FK | Collection `follower_pose` and inference `TapeArm.telemetry` now share `vx300s_kinematics.joints_from_raw_ticks` and `forward`. All 1,213 sampled records from one marker session matched saved and current transforms exactly. This checks software parity, not physical calibration. |
| TCP and base frame | TCP offset `[0.160, 0, 0]` m; same existing camera/base transform. That transform remains operator-accepted, not independently validated. |
| State | FK of **measured** follower joints. |
| Action | FK of **commanded** follower joint goals, with continuous gripper target. It is not the measured future EE trajectory. The roughly 19–26 mm median command/measured difference must not be interpreted as sensor noise. |
| Rotation | XYZ Euler radians, export X/Z modulo 2π; `wrapped_xyz_v1` delta encoding and decoding in training/deployment. Compare physical rotations rather than raw Euler differences. |
| Images | Front and wrist, same full-frame RGB transform and padding to 384×384; prepared videos are not transformed twice. Side camera is used for tracking, not as a third policy view. |
| Time | Dataset/model rate 25 Hz, history offsets `[-2, 0]` (80 ms), 20-action horizon (0.8 s). Capture cameras run at 30 Hz and are resampled. Request metadata now includes front, wrist and state timestamps for checking actual age/skew. |
| Normalization | Unaugmented training-split statistics, serialized in each fine-tuned checkpoint and loaded at deployment. Do not mix simulation normalizers with real-data deployment. |
| Depth/quality | Existing software depth correction and tracking-quality policy are retained. No new depth fit was conducted here. Do not apply correction twice or treat missing velocity as a stationary object. |
| Actuation | Collection follows leader joint goals directly; inference sends Cartesian targets through bounded DLS and Ruckig. This difference is intentional but remains a physical validation item. FK agreement alone does not validate the IK/control loop. |

New capture/evaluation manifests record state/action semantics. Inference requests
record observation timestamps. No robot motion, gain changes or safety-envelope
changes were made during this preparation.

## Highest-value data changes

1. **Review grasp/lift coverage before increasing regularization.** Before history
   filtering, the training split has 25,962 `lift_candidate` frames, only 9,848
   tracking-usable; 8,116 `closing_candidate` frames, only 2,483 usable. There are
   just 82 `held_confirmed` frames. Candidate phases are not ground truth. Review
   wrist video together with gripper closure and sustained measured EE lift, as
   agreed. Mark velocity unnecessary only after confirmed grasp/lift. This may
   recover useful lift/place supervision without inventing tracking values.
2. **Add successful stationary grasps and real recovery demonstrations.** Cover
   small offsets around the trained starts, near misses, delayed approaches,
   regrasping and marker orientation changes. Save synchronized front/wrist
   images, measured state, expert command and intervention timing. Recovery labels
   should come from the expert correcting an actually perturbed situation.
   Synthetic EE noise leaves the images unchanged and cannot provide that evidence.
3. **Keep task and phase coverage visible.** Report anchors per task, phase and
   collection session, plus masks per action horizon. Evaluate task/phase-balanced
   sampling after reviewing phase annotations; it is not enabled in these recipes.
   Preserve session-separated train/validation/test assignments for new data.

This recovery-data recommendation follows the covariate-shift motivation of
[DAgger](https://proceedings.mlr.press/v15/ross11a.html) and
[DART](https://proceedings.mlr.press/v78/laskey17a.html). Neither justifies adding
arbitrary noise to expert action labels. No physical perturbation is enabled here.

## Implemented augmentation and experiment order

`real_robot/training_augmentation.py` perturbs inputs online; it does not rewrite
the dataset or statistics. Validation is unaugmented.

- **First candidate: photometric only.** Brightness/contrast factors 0.9–1.1 and
  saturation 0.95–1.05, shared across both history frames and camera views within
  a sample. Avoid horizontal flips, arbitrary rotations or crops unless the
  geometry/action labels are transformed consistently.
- **Second candidate: photometric plus small proprioceptive bias.** In 30% of
  samples, add per-axis translation σ=2 mm and rotation-vector σ=0.5°, clipped at
  ±3σ and shared across history. Compose rotations physically, then encode using
  the export convention. Keep the expert **absolute** target and gripper unchanged;
  recompute deltas relative to the perturbed current state. Position/rotation
  preservation and temporal consistency are tested. These magnitudes are starting
  hypotheses, not values inferred from the much larger command/measured offset.
- Keep quality masks and object-velocity supervision unchanged initially. Do not
  extrapolate long occlusions or synthesize stationary velocity. Later camera
  delay/dropout augmentation should replay measured timing patterns, respecting
  quality flags and deployment freshness rules; it is not implemented here.

The recipes use LR **3e-6** (previously 1e-5), AdamW weight decay **0.01** (unchanged),
200-step linear warmup, validation-plateau LR halving (patience 2, minimum 3e-7),
gradient clipping 1, maximum **30 epochs**, and early stopping after **7** unimproved
validation epochs. Do not simultaneously increase weight decay, freeze the
backbone and alter noise: change one factor at a time. A no-augmentation control
at the same new optimizer settings is needed to attribute improvement specifically
to augmentation; the old run also differed in LR/scheduler.

The queue order is:

| Method | Initialization | Object velocity | Kinematic residual |
| --- | --- | --- | --- |
| DOM | Original released `hzxie/dynamic-vla-DOM`, pinned revision/config/weights | Off; object/environment fields removed | Off |
| DOMV | Same original released DOM weights | On | Off |
| KRV100 | Same original released DOM weights | On | On, constant gate, residual coordinates |

These are three independent fine-tunes from identical original DOM parameters,
not sequential transfer from DOM to DOMV to KRV. No simulation-specific weights
or optimizer states are used. Velocity occupies existing padded state-projection
slots; KRV adds analytical drift and changes the flow coordinates. Neither adds
a new learned module requiring separate weight initialization. All retain FastViT. Continuous
inference and LAAS apply at deployment, not inside gradient training.
The DOM path passed strict loading, backward/optimizer execution, and a fixed-noise
prediction invariance check when object/velocity inputs were altered (error 0).
All three use the same frozen dataset, split, normalization and 52,215 usable
training anchors in the original pre-annotation check; v4 uses 65,689 equally for
all three methods. This removes the previous pretraining-history confound. An RGB-only
DOM run using additional tracking-invalid frames could be a later separate study.

`photometric.json` and `photometric_plus_proprio.json` are prepared, not running.
Their one-sample optimizer checks establish compatibility, not expected accuracy
or batch-16 throughput. W&B is configured for
`aparame-clemson-university/Kinematic-Residual-VLA`; credentials remain outside
the recipes. When starting the first queue:

```bash
cd ~/Desktop/Kinematic-Residual-VLA
workstation/finetune-python.sh workstation/train_real_finetune.py \
  --contract scratch/real_robot/robust-finetuning-v4-grasp-common-dom/photometric.json \
  --output scratch/real_robot/finetuning-runs/robust-v4-grasp-common-dom-photo
```

Use a fresh output directory per experiment. No command above runs the robot.

## Judge improvement with comparable measurements

Keep clean validation flow loss for selection. Augmented training loss and clean
validation loss have different input distributions; historical online training
loss and fixed-noise validation loss also use different sampling. Their gap alone
does not diagnose overfitting. Add a fixed clean training probe and a separate
perturbed validation probe under the same seeds/sampling convention.

For the next result comparison, report sampled absolute EE position error in mm,
SO(3) orientation error, gripper error, horizon-wise error, task/phase/session
breakdowns, and controller realizability. Log desired Cartesian target, IK result,
governed joint command, measured FK, action age and guard/saturation causes. The
new training runner still selects on flow loss; these richer validation probes
are recommendations, not yet implemented selection metrics.

Use identical recorded inputs and sampling seeds for offline model comparisons;
use a settled common measured start and fixed controller build/profile for physical
trials. Keep test episodes held out until final selection. Report reset failures
and guard stops separately from object-manipulation success. Neither augmentation
nor an offline loss improvement establishes that the current reset/control faults
are resolved.
