# Learned-policy real-robot evaluation

> Plain `--model dom` now runs from RGB + robot state without an object tracker
> or mask-selection gate. DOMV/KRV retain their object-state requirements.


> The experiment campaign now uses a two-second contact-stop hypothesis for
> occluded objects. Direct evaluation retains the configurable constant-velocity
> option. See the comparison guide for assumptions, bounds and logging; restart
> the controller to apply a changed recipe.


> For the five-model post-training comparison and supervised pilot commands, see
> [README_REAL_MODEL_COMPARISON.md](README_REAL_MODEL_COMPARISON.md). The launcher
> now defaults to the v5 action fine-tunes and also supports native family checkpoints.


After the severe vibration reported for KRV100 trial 20261111T022039Z,
unfiltered learned-policy execution is blocked before cameras or robot devices
are opened. `--joint-governor off` is preview-only for learned policies. Smooth
expert-action replay did not validate unfiltered sampled VLA targets. The
default governor is retained as a prerequisite, not a claim of physical
stability or adequate tracking. The optional `--joint-acceleration-deg-s2` and
`--joint-jerk-deg-s3` parameters support offline motion-profile assessment;
candidate values must not be treated as physically validated from that audit.

The single marker expert replay with the corrected clock sent all 169 frames,
settled without a controller fault, and was reported smooth by the operator.
It used an 80ms pose blend, 600 degrees/s² acceleration and 6000 degrees/s³ jerk
bounds. Median target-to-measured error was 23.2mm, so tracking lag remains.
The policy now shares the replay's nearest-sample clock for observation indices,
LAAS prefix expiry, action selection and deadline checks. This avoids conflicting
rounding at sample boundaries; genuinely late actions still expire. The original
run and comparison are archived in
`scratch/real_robot/replays/marker-single-governed-clockfix-20261111T024148Z`.
The expert test validates that one trajectory, not VLA prediction quality.

The subsequent 10s KRV100 trial `marker-krv100-governed-20261111T024503Z` timed
out, returned to the fixed start without a controller fault, and was reported
mildly jerky. Its logs include 27 command-timestep holds. The controller now
samples one index per feedback tick and keeps it through chunk merge, selection,
and computation; the target expires at the next nominal action time rather than
the half-period nearest-rounding boundary. Decision times and deadlines are
logged explicitly. The original trace lacks those decision timestamps, so it
cannot determine how many of the 27 holds this change prevents. This timing fix
has unit coverage but has not yet been physically validated.

Run from the repository root, on the workstation with the operator at the robot:

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.evaluate \
  --model krv100 --task striped_ball --execute
```

Evaluation now defaults to `--motion-profile hardware`: no added TCP, angular,
joint or gripper velocity cap. Existing motor velocity/profile registers are
read back and respected, including the slower motor in coupled joints; no motor
limits are rewritten. Ruckig acceleration/jerk continuity remains at 120 degrees/s²
and 600 degrees/s³, along with the 80ms pose blend. These can still add lag and
are separate from velocity caps. Automatic reset is a separate, non-policy
procedure and retains its 15 degrees/s joint return limit. Perception retains
the same 1.2m/s object-velocity outlier filter used during collection; it is not
an arm-speed limit, and the policy does not clip velocity conditioning inputs.

The historical `--motion-profile smooth` is retained explicitly for comparison.
A measured-state damped
least-squares solver maps the VLA's absolute pose target for each 40ms timestep
into joint targets. The default `--ik-method bounded_dls` solves a weighted,
damped least-squares problem with individual joint-position, velocity and
following-error bounds included in the optimization. Translation and rotation
references are limited independently relative to the retained command. Rotation
residuals use a 0.1m characteristic length, also used by the offline pose IK,
instead of implicitly weighting one radian like one metre. This gives XYZ
tracking more weight without dropping orientation. A local Ruckig
trajectory generator then replans from the previously accepted joint reference's
position, velocity and acceleration. It bounds joint acceleration to 120 degrees/s²
and jerk to 600 degrees/s³, with joint speed capped at the lower of 60 degrees/s
and the existing hardware limit. TCP/angular ceilings are 0.15 m/s and 45 degrees/s.
The TCP ceiling is near the collected striped-ball demonstrations' median speed
(0.141 m/s); the acceleration/jerk settings are conservative troubleshooting
parameters, not values claimed by DynamicVLA. This deliberately adds lag to make
changing predictions gradual. It does not correct task accuracy.

Joint limits, measured following error, finite Cartesian command steps and swept
geometry are checked after trajectory generation. A violation stops the trial;
commands are not clipped after planning, which would invalidate the reference
derivatives. Tracking/health holds immediately retain the last goal and reset
trajectory derivatives. Those safety stops take precedence over smoothness.
Gripper commands use firmware read-back limits in hardware mode; the historical
250 ticks/s additional limit applies only in smooth/pilot/standard modes.
No motor gains or persistent hardware limits are written. Raw IK targets and
trajectory position/velocity/acceleration/limits are logged separately.

Install the extra dependency when setting up another workstation:

```bash
.venv/bin/python -m pip install -r real_robot/requirements-policy-eval.txt
```

`--motion-profile auto` omits the joint trajectory stage. Its configured TCP ceiling is
0.5 m/s and angular ceiling is 90 degrees/s.
`--ik-method uniform_dls` explicitly selects the historical unbounded DLS solve
followed by common joint scaling. Both methods retain the finite Cartesian-step,
following-error and collision guards. Motion profile and IK method are separate
recorded choices; changing `auto` alone does not select the historical solver.

`--motion-profile pilot` and `--motion-profile standard` remain optional lower
ceilings for supervised troubleshooting. They use the direct IK solver without
the acceleration/jerk trajectory stage.

The bounded implementation uses SciPy's
[bounded-variable least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html)
on the augmented weighted Jacobian/damping system. Its `command_scale` records
only the final finite-Cartesian backtracking factor; the historical method's
value includes global joint scaling. Use `command_scale_scope` and XYZ errors
when comparing logs, rather than interpreting the two values as the same gain.

`scratch/real_robot/bounded-dls-audit-20261110/report.json` compares both solvers
on 866 identical recorded states/targets. Overall median one-step XYZ error
changes from 10.8mm to 7.8mm; moving-phase tail errors remain near 53–61mm. A
simple lagged-reference simulation is mixed, improving one trial and slightly
worsening the other. These offline results do not establish a physical fix.
The new implementation retains the existing speed/acceleration/jerk ceilings;
a supervised comparison with the same saved start pose is still required.
Append `--trial-seconds 60` for a longer trial. Keep controller settings matched
when comparing models.

Use `--checkpoint last` to select the last **completed** fine-tuning epoch
(currently epoch 10), instead of the default best-validation epoch 3.
`--max-trials 1` ends the session after a single success, failure or timeout.
The session JSON records the selection and epoch; recovery snapshots are excluded.

The controller captures both the measured arm joints/TCP/gripper and the motor
commands that were holding that pose at startup in `controller/start_pose.json`.
Return trajectories restore `command_joints_rad` / `command_gripper_raw`; the
settling gate compares feedback against `joints_rad` / `gripper_raw`. These are
deliberately separate: commanding measured angles directly can reproduce a
steady load-dependent following offset a second time. No motor gains or integral
error correction are added. Every trial is gated on measured agreement with
that fixed reference: <=2 degrees per joint, <=2 degrees rotation, <=12 gripper
ticks, and 0.4s quiet feedback. TCP position uses the window's coordinate median:
READY requires <=10mm initial entry and remains latched up to 11mm to avoid
rearming on encoder quantization. Every raw sample must still be within 12mm,
with <=2mm TCP excursion across the window and the existing joint-speed check.
A real movement, stale sample, or other failed check clears READY immediately.
The target home itself is unchanged. Normal trial completion returns
to it automatically, including after the last requested trial. Return motion
uses a prechecked, jerk-limited joint path at <=15 degrees/s, retimed to the
configured Cartesian/angular ceilings. The same live health, following-error,
joint and swept-geometry checks remain active; failed validation or
telemetry stops the return and prevents a new trial. P/Q and mask reselection
cancel a return. SPACE from a different pose first requests return, then waits
for another SPACE at READY. No motion is sent before the operator's first SPACE.

Start matching is approximate; the measured start and accepted tolerances are
recorded for each session/trial. Adjust evaluation's acceptance gate using
`--start-position-tolerance-mm 10 --start-joint-tolerance-deg 2`. The previous
strict gate is available with values `5` and `1`. These options do not change
the target home, IK solver, reset trajectory, motor gains or motion/geometry
guards. They apply even when reusing an older `--start-pose` file; that file
supplies the fixed reference and holding commands, not the new run's tolerance.
Reset follows the original joint holding commands through Ruckig, bypassing
Cartesian DLS. A quiet 8mm / 1.14-degree residual previously failed the stricter
gate despite reaching its joint-command endpoint.

To reuse exactly the same target for another model/session, add:

```bash
--start-pose scratch/real_robot/evaluations/PREVIOUS_SESSION/controller/start_pose.json
```

Trials record measured starting joints/TCP/gripper, the reference file and the
settling check. Return motion is logged outside the scored episode. The two
November 9 moving-marker trials recorded before this feature had different arm
starting poses and must be treated as diagnostics, not a matched-start comparison.

Older start-pose files without holding-command fields are rejected when explicitly
reused; take a new paired snapshot or recover the initial holding command from
the same-cycle feedback plus logged following error before any reset. The failed
`marker-domv-20261110T003004Z` session has an audited
`controller/start_pose.recovered.json`; its original reference and logs are
preserved. `reset_bias_diagnosis.json` documents the observed 12mm residual and
the recovered arm command. The original gripper command was not logged, so that
recovered file retains the measured initial gripper target; its observed residual
was already within one tick of the required reference.

Use `--task marker --new-mask` to evaluate the green marker. The launcher uses
`object_profiles/marker/on_ramp` and the exact collection instruction,
"Pick up the moving green marker and place it in the box." Select the marker in
the front-camera prompt, press ENTER, then SPACE in the evaluation window. Task
selection changes the tracking profile, language instruction and session label;
the chosen checkpoint and smooth controller settings are retained.

For three short moving-marker trials with exact input/chunk recording:

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.evaluate \
  --model domv --checkpoint best --task marker --new-mask \
  --motion-profile smooth --pose-smoothing-ms 80 --record-inputs \
  --execute --trial-seconds 20 --max-trials 3
```

Each trial starts with SPACE; roll the marker along its normal collection path
after EXECUTING appears. Subsequent trials remain paused until SPACE. There is
an automatic guarded return to the saved session-start pose after timeout or
ENTER/X. Wait for READY before placing/rolling the marker for the next trial.
P pauses a trial/return; Q ends the session without starting return motion.
Full absolute predicted chunks are saved in `exchange/inference.jsonl`.
`--record-inputs` additionally saves lossless observations and corresponding
chunks under `exchange/inference_inputs`, using a bounded background disk writer.
Control outputs are published before recording. A disk failure/backlog ends the
worker and causes the controller to stop rather than silently lose recordings.
After shutdown, per-trial XYZ plots and metrics are generated under `plots/`,
including measured EE feedback, raw LAAS-selected targets, full predicted chunks,
FK of governed joint commands and tracked object surface positions.

Repeat with `--model domv` for the matched comparison. The launcher uses each
model's best completed fine-tuning epoch, currently epoch 3. It starts EfficientTAM
with the saved striped-ball prompt, an isolated GPU policy worker, and a paused
follower controller. It requires the robot's existing healthy holding torque.
It does not enable torque or modify motor modes, gains, or current/velocity limits.

SPACE starts a trial. ENTER labels success; X labels failure; P marks a takeover
failure and pauses. R pauses (failing an active trial) and opens fresh front-mask
selection. Draw a tight box around the visible ball and press ENTER; failed mask
seeds can be redrawn without restarting the session. Q ends the session while retaining the existing holding goals.
A takeover, timeout or independent joint/geometry violation is a failed trial. Judge success
by completing the instructed placement without drop or assistance. Confirm the
physical workspace before pressing SPACE. All requested and resolved motion ceilings are recorded in the session manifest.
The learned evaluator has no configured Cartesian XYZ workspace box. Geometry checks evaluate
intermediate joint configurations along the command step. Closed-loop behavior
with a changed controller must be measured again; offline solver convergence is
not physical success evidence.

H confirms an observed grasp/lift only when the operator sees the object held in
the wrist view and measured closure/lift conditions also pass. This annotation
allows neutral external-object velocity during holding. Tracking loss alone never
confirms holding and never creates a stationary-object measurement. Without fresh
required tracking, the controller holds. Brief gaps preserve the trial token and
future buffer; elapsed actions are rejected using their intended execution index.

Closure support uses the same aperture bound as offline grasp annotation:
measured and commanded aperture <=67.5% of calibrated travel and measured lift
>=20mm, with fresh wrist imagery and operator visual confirmation. Objects can
keep the jaws apart; the previous <=20% empty-gripper criterion rejected valid
ball/orange grasps. Telemetry thresholds alone never establish attachment.

The full valid suffix of each 20-step chunk is retained; there is no extra
six-step source-age truncation. A genuine trial reset clears the stream. There is
no scripted grasp/place fallback. Return-to-start motion is separate from the
scored policy rollout.

The policy uses the same full-image 384px resize/pad as the training export, RGB
channel order, two camera views at offsets [-2,0], saved train-only normalization,
wrapped XYZ Euler deltas, and +1=open/-1=closed gripper convention. Finite
gripper predictions are saturated to [-1,1] before conversion to calibrated raw
endpoints; non-finite commands are rejected and the gripper rate limit still applies. Predicted deltas
are converted to absolute commands relative to the observation used for inference.
The continuous inference worker reads the newest available observation immediately
after each completed chunk. The controller indexes commands on the elapsed 25 Hz
clock, drops latency-invalid prefixes, and replaces overlapping actions with newer
predictions. The worker uses chunk inference with `enable_streaming=False` because
the outer worker/controller pair implements asynchronous inference and LAAS.

Each session writes controller feedback/actions, chunk source indices and discarded
prefixes, raw wrist captures during trials, combined front/wrist MP4s, and explicit
operator trial outcomes under `scratch/real_robot/evaluations/`. `latest.json` points
to the latest session after worker and camera startup. A session that stays paused
has zero evaluated trials. Missing tracker inputs are not counted as successes.

Offline integration checks (no robot access):

```bash
HF_HUB_OFFLINE=1 workstation/finetune-python.sh \
  workstation/check_real_eval_transport.py \
  --run scratch/real_robot/finetuning-runs/20261109T014917Z
workstation/finetune-python.sh -m pytest -q real_robot/tests/test_policy_stream.py
```

The saved-data transport check exercises both selected policies, image processing,
normalization, delta-to-absolute conversion, and action-prefix removal. Its saved
images are reconstructed to raw transport size; this check measures integration,
not closed-loop performance or physical position accuracy.

## Mask selection at startup

Append `--new-mask` to force a new live front-camera prompt. The evaluator now
enables the manual-selection preview, backs up the previous saved profile in the
session, and waits for fresh valid position/velocity tracking before starting the
robot controller. If a reused prompt cannot establish usable geometry for four
seconds during startup, it asks for a fresh selection automatically. Time spent
drawing a mask does not consume the normal startup timeout.

During evaluation, press R to pause and request reselection. Motion stays paused
while `MASK_PENDING` is present; SPACE is required again after fresh geometry
returns. A visible segmentation mask and valid depth are separate requirements.
If a fresh mask still gives insufficient depth, the robot remains paused.

## Small EE-target smoothing and grasp workspace

The evaluator defaults to an 80ms causal quintic pose-reference blend before
the same measured-state DLS solver. Replanning carries reference position,
velocity and acceleration across updates; rotations use continuous rotation
vectors rather than Euler-component interpolation. Tracking holds reset the
reference to the retained joint goal with zero reference velocity/acceleration.
`--pose-smoothing-ms 0` disables it for a matched ablation. Raw and smoothed
targets must be finite; IK, joint/rate/following and geometry checks still apply.

The default evaluator configuration is `real_robot/config_policy_eval.yaml`.
Its legacy servo XYZ bounds remain for other controller/schema consumers, but
are not enforced on learned-policy targets, smoothed references, start poses or
reset paths. A DOM target at z=12.3mm was previously rejected by the 20mm TCP
floor; 4,421 of 33,657 marker action frames also lie below that floor.

Evaluation defaults to `--geometry-profile grasp`. The two named fingertip
meshes may contact the table, including modeled fingertip/table overlap. Their
table pairs are not classified as self-collision. Other moving links and the
wrist-camera bound must stay above the modeled table plane (0mm clearance,
1mm numerical allowance), and self-collision checks remain. The profile permits
contact; it does not add force regulation or establish physical penetration
depth from the mesh model. The legacy `--geometry-profile hover` retains the
old 20mm geometry margin. Hover/tracking controllers keep their own XYZ bounds.

All model comparisons must use the same profile. The first DOM and DOMV trials
before this change remain historical diagnostics; repeat them before comparing
against KRV100 with the new profile. See `marker-contact-profile-audit.json` in
`scratch/real_robot/evaluations/` for the offline dataset compatibility check.
Human striped-ball action profiles were extracted from 151 training episodes with
quality-usable consecutive frames. On three contiguous example segments, the 80ms
blend reduced p95 positional jerk by about 50% with 7.1mm p95 positional lag. These
are offline reference statistics, not measured closed-loop improvement.

## Continuity and clearance diagnostics

Depth-only coasting gaps of at most 80ms may use a causal position prediction
from the last accepted position/velocity. This requires fresh camera data, the
same rig/frame, bounded tracker covariance and at most 200ms total measurement
age. Predicted rows never renew the measurement anchor. Startup, uncertain
tracking and longer gaps still hold. Live status and inference metadata label
accepted measurements versus short-gap predictions.

The controller logs exact measured joints/TCP when a geometry guard trips, and
the error names the geometry and its clearance. Proposed paths are checked both
at nominal commanded joints and with the current measured following offset.
The grasp profile permits the expected fingertip contact. Other geometry stops
identify the affected link/camera; retaining more action suffixes does not
resolve those independent checks.

## USB transport at the required visual rate

VLA evaluation requires 30FPS capture; 15FPS is not offered by the evaluator.
The trained 25Hz observation/action clock is retained. A faster command loop
cannot restore missing visual samples from a slower camera.

The evaluator automatically selects packed YUYV on a front USB 2 connection and
BGR8 on USB 3, both at 30FPS. No USB override flag is needed:

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.evaluate \
  --model krv100 --checkpoint last --task striped_ball \
  --new-mask \
  --pose-smoothing-ms 80 --motion-profile smooth --execute \
  --trial-seconds 60 --max-trials 1
```

YUYV transports color in two bytes/pixel and is decoded into the same BGR image
layout before EfficientTAM and policy preprocessing. It preserves 640x480 geometry
and the existing depth-correction profile. The side stream remains disabled and
the wrist remains on USB 3. No calibration or persistent camera settings are
changed. The stream format, frame numbers and source FPS are logged.

A front-only 120-frame probe of this mode started in 1.26s, measured 29.99 unique
RGB and depth frames/s, and had no steady-state timeouts or skipped mismatched
pairs. The visible ball diagnostic ROI passed 90/90 steady frames with 89.3%
median valid depth. The ROI check is not a tracking-identity or robot-success test.

Inspect camera identities and USB links without opening streams with:

```bash
.venv/bin/python -m real_robot.camera_transport
```

## Constant-target control isolation

Before further gain or IK changes, add `--hold-target-check --trial-seconds 8
--max-trials 1` to the same evaluation command. Return the arm to a clear held
starting pose first. The session starts paused; SPACE captures the existing
commanded TCP pose and gripper goal as a fixed reference. The full VLA/perception
workload still runs, but its pose/gripper predictions are logged and ignored for
actuation. This isolates DIK/motor behavior from changing policy targets without
commanding a new task pose. The selected geometry and joint/health checks remain
active. After eight seconds it records `diagnostic_complete`, not task success.

The 2026-11-09 changing-target audit found exact DIK replay and no software
integral term. Motor position gains read P=800, I=0, D=0; Profile Acceleration and
Profile Velocity both read zero. No gains/profiles were changed. The raw VLA targets
jumped up to 38mm and 25 degrees between control samples, so jitter already exists
before IK. Raw statistics and measured joint responses are saved in each audited
session's `control_stability_audit.json`.

## DynamicVLA comparison and smooth-motion replay

DynamicVLA's [paper](https://arxiv.org/html/2601.22153v1) describes continuous
inference plus LAAS: discard elapsed actions and prioritize newer overlapping
chunks. This evaluator retains that timing behavior. Their released
[simulation controller](https://github.com/hzxie/DynamicVLA/blob/master/simulations/configs/robot_cfg.py)
also uses absolute EE poses and DLS IK, with damped simulated joint actuators.
The released examples do not supply a VX300s real-robot controller or establish
their real-robot motor gains. The four-stage geometric state machine generates
demonstrations; it is not an extra controller replacing learned VLA actions at
deployment. The new Ruckig stage is our hardware adaptation, not a claim about
their unpublished real-robot implementation.

`scratch/real_robot/control-smoothing-audit-20261109/audit.json` replays the saved
KRV100 and DOMV joint targets through the new stage. On consecutive 40ms samples,
95th-percentile joint acceleration changed from 2156/2114 to 117/96 degrees/s²,
and jerk from 89861/81113 to 600/600 degrees/s³. Reference generation took under
0.4ms at the 99th percentile. These are numerical reference tests, not physical
smoothness or task-success results; replay does not update the model's
observations using the changed motion. Hardware feedback, encoder quantization,
tracking holds and motor response still require a supervised trial. Use matched
motion profiles and smoothing settings when comparing KRV100 and DOMV.

## Offline marker model/controller comparison

`workstation/compare_marker_models.py` compares the original
`hzxie/dynamic-vla-DOM` checkpoint, the untouched DOMV initialization, and
real-robot DOMV completed epoch 3 on identical recorded inputs. It opens no
robot/camera devices and updates no weights. Outputs under
`scratch/real_robot/static-marker-eval/comparison` include XYZ plots, all 20-step
predictions, cached inputs, measured/expert references and `report.json`.
Episode 551 is a training demonstration, including a static waiting prefix and
subsequent rolling/grasping. The separate stationary live segment is from frames
400 onward of the November 9 marker trial; its measured EE trace is the original
policy rollout, not an expert reference. Do not interpret these plots as a
closed-loop success benchmark. The live plot also includes raw LAAS-selected VLA
targets and FK of the smoothed joint commands.

Re-render the saved predictions without GPU inference:

```bash
MPLCONFIGDIR=/tmp/krvla-mpl workstation/finetune-python.sh \
  workstation/compare_marker_models.py \
  --output scratch/real_robot/static-marker-eval/comparison --plot-only
```

## Shared occlusion estimation for subsequent evaluations

All tracking-dependent learned-policy entrypoints now use the same profile
resolver in `real_robot/object_estimation.py` and the existing causal
`LiveObjectEstimator`. This includes direct DOMV/KRV evaluation, native SmolVLA
and KR+SmolVLA, the best/last SmolVLA wrapper, and newly prepared campaigns.
Plain DOM still uses its trained RGB/state-only path without requiring a tracker.
Prediction occurs once in the controller; each worker receives that same object
state and its logged source. No policy is given extra neural channels.

The common maximum age is now two seconds, with the existing 5cm displacement,
25mm position-sigma, and 0.15m/s velocity-sigma limits. These can expire a prediction
earlier than the time limit, especially for moving objects. No prediction renews
its measured anchor. Camera freshness/calibration and robot checks still apply.
A ready genuine measurement replaces the prediction immediately. A prediction
is logged as an assumption and never becomes measured velocity for innovation
integration. Startup requires a real ready track; no missing startup state is
filled with zero.

`--object-motion auto` selects the prior `contact_stop_assumption` only when all
checkpoint dataset sources for that task are declared static. Otherwise it uses
`constant_velocity`, preserving the measured moving-object velocity and integrating
its position. The static profile decays residual velocity with the prior 0.15s
time constant; it is a contact-stop hypothesis, not a detected grasp. Direct
controller invocation without dataset metadata defaults to the moving profile.

Override the task's motion explicitly when appropriate:

```bash
.venv/bin/python workstation/smolvla_orange_eval.py best --execute --object-motion moving
```

The existing best/last commands need no extra flags for the static-orange model.
Direct `real_robot.evaluate` also accepts `--object-motion static|moving` and an
explicit `--object-occlusion-model constant_velocity|contact_stop_assumption`.
An explicit model override takes precedence. New campaigns resolve `auto` from
the requested static/moving condition; historical plan settings remain explicit
and are not rewritten.

Restart a running evaluator to pick up this change. Visible RGB may still be
occluded even while state prediction remains available; extrapolation does not
reconstruct the object in the image or establish physical grasp success.

Unconditioned native family baselines now share the RGB/state-only path with
plain DOM. In particular, the static-orange `smolvla_baseline` needs no object
position/velocity stream, mask selection, or two-second occlusion timer.
`smolvla_kr`, DOMV, KRV and explicit velocity-conditioned variants retain their
trained object-state requirements. This is a controller/transport dependency
change, not a change to any neural input dimensions or learned weights.

## Native SmolVLA guided RTC

`workstation/smolvla_orange_eval.py best|last` now defaults to `--rtc guided`;
append `--rtc off` to compare the original native sampler. The separate inference
and controller processes were already asynchronous. RTC now guides each new chunk
against the controller-acknowledged pending physical plan using a measured delay
estimate. Direct `real_robot.evaluate` keeps its explicit `--rtc off` default and
accepts `--rtc guided` for `smolvla_baseline` and `smolvla_kr` only. Unsupported
families reject the flag rather than silently substituting a different sampler.

No IK, pose-blend, acceleration/jerk or firmware limit changes accompany RTC.
See [the static-orange RTC audit and commands](README_SMOLVLA_STATIC_ORANGE.md#guided-rtc-and-the-existing-action-controls)
for the complete control table, measured saved-input latency and comparison commands.
