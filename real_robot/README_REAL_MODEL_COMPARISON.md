# Matched real experiments after SmolVLA+KR

The staged comparison is
`scratch/real_robot/real-experiments/post-smolvla-v6-20261112/plan.json`.
It includes the latest real-data DOM, DOM+Velocity, KRV, SmolVLA and SmolVLA+KR
fine-tunes. Select each completed run's **best sampled-action-score** checkpoint,
including the selected raw/EMA weights. VLASH and pi0.5 are not completed
real-data fine-tunes on this workstation and are excluded from this comparison.
Historical epochs are not separate model entries.

The training supervisor is paused after the user requested this comparison;
SmolVLA+KR itself finishes normally. `family-finetuning-v6/evaluation-hold.json`
records the supervisor identity and resume command. Keep that supervisor paused
while using the GPU for real experiments. After all experiments, verify its
identity before resuming the remaining training queue. Training was not killed.

The preparation watcher waits for completed SmolVLA+KR training **and final
before/after evaluation**, then for its process to release the GPU. It freezes
checkpoint hashes, image preprocessing and model input protocols, and launches
one isolated saved-observation inference preflight per model. This opens no
robot/camera devices and starts no physical motion. Inspect `status.json` and
`preflight/*.json`; a failure blocks experiment launch.

Current models keep their original inputs: DOM has no explicit velocity; DOMV
and KRV use their saved configuration; v6 SmolVLA+KR uses current velocity for
analytical drift only. Do **not** retrofit the migration branch's new neural
velocity prompt into these trained weights. SmolVLA retains its single-frame
neural input while DOM uses [-2, 0]. Both share the existing two-frame transport,
absolute 20-action Cartesian output, 25Hz action clock and elapsed-action expiry.
The live family adapter uses the same native prompt/normalization and absolute
pose conversion as training action validation; it rejects invalid required
current velocity and gates drift after operator-confirmed holding.

## Plain DOM without object tracking

Plain DOM needs the front/wrist RGB images, the robot's six-dimensional measured
EE pose history and the instruction. It does not consume externally tracked
object position or velocity. `--model dom` now selects a front camera feed with
no EfficientTAM, no object mask and no object-state gate (provided the selected
checkpoint has neither velocity conditioning nor residual flow enabled).
Camera/wrist freshness, robot health, fixed start, collision and motion checks
still apply. No dummy object position/velocity is sent to the model or archives.
`--new-mask` is ignored in this mode; there is no mask selection window.

For one static-marker DOM trial:

```bash
DISPLAY=:1 .venv/bin/python workstation/real_experiment_campaign.py launch \
  --root scratch/real_robot/real-experiments/post-smolvla-v6-20261112 \
  --model dom --task marker --phase pilot --condition static \
  --max-trials 1 --execute
```

Place the stationary marker in view, wait for READY, then press SPACE in the
controller window. The session prints `DOM uses RGB + robot state only; object
tracker and mask selection are disabled.` No changes are made to the model's
architecture, weights or task instruction. Object occlusion can still affect
what the network predicts from RGB, but it no longer causes a controller hold
solely because object tracking disappeared.

DOMV/KRV retain their required object tracking/velocity checks. Current native
SmolVLA evaluation remains on its existing tracking path. Requests/manifests
record object-input usage; summaries separate these plain-DOM runs from earlier
DOM sessions gated by the shared tracker. With tracking disabled, speed-bin
labels are requested conditions, not measured object speeds. Independent motion
measurement is needed before reporting a speed-stratified comparison.

## Supervised pilots

The trained task set is the orange, striped ball and green marker, derived from
quality/history-eligible anchors in the actual training split. The pilot has three trials
per model/object (45 trials across five models and three objects) and is reported
separately from scored trials. All models use
the same hardware motion profile, joint governor, bounded DLS IK, 80ms pose
blend, grasp geometry and start acceptance settings. The first pilot captures
the shared measured start and holding commands; later sessions reuse that file.
Every trial still needs the operator's SPACE at READY. Roll the object only
after execution begins. ENTER labels completed pick-and-place, X failure, P/Q
pause/stop; H confirms visible holding using the existing grasp/lift gate.

Use the workstation's normal robot Python, from the repository root:

DOM:

```bash
DISPLAY=:1 .venv/bin/python workstation/real_experiment_campaign.py launch \
  --root scratch/real_robot/real-experiments/post-smolvla-v6-20261112 \
  --model dom --task striped_ball --phase pilot --condition iid_slow \
  --new-mask --execute
```

DOMV:

```bash
DISPLAY=:1 .venv/bin/python workstation/real_experiment_campaign.py launch \
  --root scratch/real_robot/real-experiments/post-smolvla-v6-20261112 \
  --model domv --task striped_ball --phase pilot --condition iid_slow \
  --new-mask --execute
```

KRV:

```bash
DISPLAY=:1 .venv/bin/python workstation/real_experiment_campaign.py launch \
  --root scratch/real_robot/real-experiments/post-smolvla-v6-20261112 \
  --model krv100 --task striped_ball --phase pilot --condition iid_slow \
  --new-mask --execute
```

Run one session at a time. Each command requests three 30s striped-ball trials
using each model’s recorded input requirements. DOM bypasses object tracking;
DOMV/KRV use the common two-second contact-stop hypothesis configured in the plan.
Append `--max-trials 1` for one trial. For the static green marker, replace
`--task striped_ball --condition iid_slow` with `--task marker --condition static`.

The process starts paused. Follow the block order in `plan.json`, swapping
`--model` and `--task` (`orange`, `striped_ball`, `marker`) for each three-trial session. Each launch captures model
inputs, complete action chunks, controller feedback, reset diagnostics and video.
No success/failure labels or physical outcomes are generated by the preparation
watcher. CLI launch without `--execute` is preview and is excluded from scores.

## Scored comparison

After the pilots establish usable physical behavior, the existing collection
protocol provides 20 trials per model/object/condition with a 30s timeout:
`iid_slow` (0.01–0.04m/s), `iid_medium` (0.04–0.08m/s), and `speed_ood`
(0.12–0.18m/s). Across five models and three objects this would be 900 scored
trials; it is a prepared schedule, not an automatically launched commitment.
Model block order is seeded and shuffled across task/condition blocks.

Pass `--phase scored` and the selected `--condition` when ready for a scored
block. Requested condition is logged independently of measured object velocity:
manual rolling does not establish the speed bin. Verify actual current velocity
from the recordings before reporting speed-stratified performance. Preserve
out-of-bin and failed attempts rather than silently discarding them.

```bash
.venv/bin/python workstation/real_experiment_campaign.py summarize \
  --root scratch/real_robot/real-experiments/post-smolvla-v6-20261112
```

`rollout-summary.json` keeps pilot/scored groups separate and reports success
rates, successful completion time, confirmed holding, timeouts, takeovers and
raw failure reasons. Controller tracking plots remain in each session's `plots/`.
Autonomous rollout has no synchronized expert target, so its tracking error is
not the held-out action MSE. Camera/perception/controller settings and inference
latencies remain part of the result; do not attribute all execution failures to
the policy network.

## Front-view occlusions

Tracking-dependent models use the campaign’s **two-second contact-stop
assumption**. Plain DOM does not use object tracking or this extrapolator. On object loss, velocity decays toward zero with a 0.15s time constant;
position integrates that decaying velocity and settles. This is the explicitly
selected hypothesis that end-effector contact stopped the object, not evidence
of detected contact or grasp. Complete camera occlusion can also happen while
an object is moving. Predicted inputs and their model are logged as assumptions.

The model is `v(t)=v_last*exp(-dt/0.15)` and
`p(t)=p_last+v_last*0.15*(1-exp(-dt/0.15))`. Visible accepted measurements are
unchanged, and a genuine ready measurement immediately replaces the prediction.
The anchor must have trustworthy position/velocity covariance. No startup zero
or predicted observation can become a new measurement anchor. Fresh front/wrist
frames and unchanged camera/base calibration remain required.

Limits remain 5cm predicted travel, position sigma <=25mm and velocity sigma
<=0.15m/s under the logged hypothesis. Position uncertainty includes a 5mm/sqrt(s)
stationary diffusion term and velocity uncertainty has a 3mm/s floor. These are
assumed model uncertainties, not calibrated physical accuracy. If a limit or the
two-second expiry is reached, the controller holds. Confirmed holding with H
remains a separate explicit operator transition with its existing closure/lift
checks. Motor, collision, timing and start-pose checks remain active.

The campaign uses `settings.object_occlusion_model=contact_stop_assumption`,
`settings.object_prediction_max_gap_s=2.0`, and
`settings.object_prediction_max_displacement_m=0.05`. These settings are passed to
all model sessions and saved in requests/manifests/recordings. Results from
other tracking recipes are grouped separately in rollout summaries. Restart a
running controller to apply new settings.

Direct evaluation also supports `--object-occlusion-model constant_velocity`
with `p=p_last+v_last*dt`, `v=v_last`, and the original 0.5s default. Its
`--object-prediction-max-gap-s` accepts 0..5s (0 disables prediction). The campaign
commands above explicitly select the common two-second contact-stop settings.

Offline replay of the failed static-marker trial is saved under
`occlusion-replay-contact-stop-2s/summary.json`. The new hypothesis admitted 59
tracking frames up to about 1.97s into occlusion; the earlier CV recipe admitted
7 frames up to about 0.23s (`occlusion-replay/summary.json`). Most of that trial's
prolonged loss still exceeded the window. This replay is not physical validation
or proof that the assumed stop occurred. The VLA continues to receive the actual
camera image, including the occluder; its architecture/weights are unchanged.

## Constant-gate continuation: DOMV/KRV with velocity innovation

The newer pair is in
`scratch/real_robot/finetuning-runs/static-continuation-constant-20261112/dom-runs`.
Use those checkpoints explicitly; the older five-model campaign above still
selects the v5 runs. Check completion of **both** models and print their launch
commands with:

```bash
.venv/bin/python workstation/prepare_closed_loop_pair.py --wait --task striped_ball
```

This watcher only reads training records and prints commands. It does not stop
the training queue, reserve the GPU, open devices or start trials. Run the printed
commands one at a time when the GPU is available for evaluation. Each starts
paused and requires the existing operator SPACE at READY. Reuse the first
session's `controller/start_pose.json` with `--start-pose` for the paired trial.
Use `--task marker` or `--task orange` for the other objects.

The native checkpoint config, constant residual gate, normalization, neural
velocity inputs and KRV reconstruction are preserved. DOMV has no residual flow;
its saved constant gate is inactive. `--velocity-innovation on` adds the same
training-free correction to both models after physical action reconstruction
and before pose smoothing/IK/geometry checks. No training change is required.
Direct evaluation defaults to `off`, preserving previous experiment recipes;
use matched `on`/`off` runs to measure its effect.

The correction is the causal trapezoidal measured-velocity integral since the
active chunk's **front source observation timestamp**, minus its source velocity
multiplied by elapsed physical time. Stale-prefix removal keeps that anchor.
Accepted robot-frame measurements are integrated once; the latest accepted
velocity is held only from its timestamp to the current control tick (<=200ms).
Occlusion predictions are excluded and break the integral history; an unavailable
anchor fades the correction toward zero until a new measured chunk is selected.
The paper's gain 1, 5cm Euclidean magnitude limit, 5mm per-step change limit and
latched first nonpositive gripper-command fade apply. This latch affects only
the innovation offset, independently of the checkpoint's constant gate. A new
trial resets the integral, offset and latch. Orientation/gripper predictions
are unchanged; no full object displacement is added again. Session/controller
manifests and action logs record the enabled setting, source time, raw innovation,
applied offset and latch state. Hardware performance remains to be measured.

### Native training/evaluation parity audit

The launcher now checks the selected native checkpoint config against its source,
training dataset/quality identity, saved state/action normalization, velocity
statistics, constant gate, camera/base transform, TCP offset and observation/action
clock before starting devices. It saves `native_contract.json` in the session.
The live task instruction now comes from the actual training instruction generator
and dataset task metadata. For orange it is
`pick up the rolling orange and deposit in the box.` Earlier rollouts used the
collection instruction (`Pick up the rolling orange and place it in the box.`),
which was outside the generator's exact template set. The effect of this change
on grasping has not been physically measured.

Reproduce the device-free GPU parity audit:

```bash
HF_HUB_OFFLINE=1 workstation/finetune-python.sh \
  workstation/audit_native_eval_parity.py \
  --run scratch/real_robot/finetuning-runs/static-continuation-constant-20261112/dom-runs \
  --output scratch/real_robot/finetuning-runs/static-continuation-constant-20261112/native-eval-parity \
  --session scratch/real_robot/evaluations/orange-domv-innovation-20261113T220745Z \
  --session scratch/real_robot/evaluations/orange-krv100-20261113T221713Z
```

The audit compares training-validation sampling against deployed sampling with
identical noise on held-out examples of all three tasks, recorded rollout inputs,
and confirmed-holding quality gates. It also compares available raw training RGB
through live preprocessing against decoded training video. AV1 is lossy, so those
pixels differ slightly. Training-only photometric augmentation remains disabled
at evaluation, as during validation. Native decoding excludes velocity innovation;
`--velocity-innovation on` remains the separate requested execution ablation.
Constant residual gating retains the trained `velocity_required` quality mask;
it does not introduce a geometric grasp gate.

Model decoding parity does not make the physical collection and execution paths
identical. Demonstrations used leader/follower joint commands; deployment converts
Cartesian predictions through bounded IK, pose blending and Ruckig. Live
asynchronous inference skips expired chunk prefixes on the same 25Hz clock.
These differences are retained and recorded; physical equivalence still requires
expert replay. No checkpoint weights or training gates are changed by the audit.


## Fresh original-base restart and Palmetto DOMv

The November 14 experiment is independent of the continuation runs above. KRV
uses original released DOM initialization and causal gating on the workstation.
DOMv uses the same original DOM base and all 859 real episodes on two Palmetto
V100s, job 20504544. Both preserve recorded prompts and the 706/80/73 split;
selection uses the same 512 validation anchors and three seeds. The remote run
uses FP32 masters plus FP16 autocast/GradScaler because V100s do not support the
workstation BF16 path. The local DOMv queue stage was removed without stopping
KRV. See [the Palmetto run and CPU/DTN policy](README_PALMETTO_DOMV.md) and
[the fresh training recipe](README_FRESH_DOM_POSTTRAINING.md).
