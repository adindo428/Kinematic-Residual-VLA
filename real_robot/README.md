# Real-robot dynamic-object collection: CPU prototype and implementation plan

This module starts the SO101-to-ALOHA collection effort without using the GPUs
running DOM training. It implements **stationary-camera RGB-D object tracking,
causal velocity estimation, calibration utilities, and a model-based hover
controller evaluated in shadow mode**. It has no robot-command backend.
Camera-only captures and synthetic demos are explicitly marked `expert_episode:
false`; they cannot be counted as grasp demonstrations.

## What DynamicVLA actually releases

Release audit: **October 16, 2026**, upstream tree
`434dc7105cf6d559018d15af5fecbcdad44f036e`. The [official repository](https://github.com/hzxie/DynamicVLA)
contains generic training, inference, simulation expert state machines, and a
simulation HDF5-to-LeRobot conversion script. The public
[DOM dataset](https://huggingface.co/datasets/hzxie/DOM) is synthetic; the
[released checkpoint](https://huggingface.co/hzxie/dynamic-vla-DOM) is trained
on DOM. The author's current public dataset listing contains only `hzxie/DOM`.
I did not find a released real-world dataset or robot-specific post-trained checkpoint.

In [issue 16](https://github.com/hzxie/DynamicVLA/issues/16), the author says the
real collection code awaits refactoring and that `dynamicvla_real_aloha.yaml`
is not included in their repository. They distinguish the front/side collection
views from the wrist/front policy views and mention about 200 episodes per
initial object configuration. [Issue 12](https://github.com/hzxie/DynamicVLA/issues/12)
also says real collection, training, and evaluation code is not yet released;
an earlier reply describes REALTIME_SAM2 and pre-registering objects.
Do not present the nonexistent ALOHA config as a released recipe.

The [paper, v2 §4.4 and Appendix B](https://arxiv.org/html/2601.22153v2)
describes mask tracking with EfficientTAM, calibrated front/side triangulation,
short-window least-squares linear velocity, and curved-path extrapolation.
It explicitly does **not** estimate real-world object orientation/angular
velocity: grasp orientation uses motion direction and object priors. A
state-machine expert approaches, grasps/lifts, places, and resets. Post-training
fine-tunes on robot-specific demonstrations using the same action objective.
These published descriptions and the older issue reply differ in tracking
implementation; neither provides a complete released hardware collector.

Relevant local code to inspect/reuse conceptually:

- `third_party/DynamicVLA/simulations/state_machines/{pick_sm,place_sm,sm_utils}.py`:
  task stages and predictive grasp targets. `get_grasp_position` uses a 0.23 s
  lookahead; this is an upstream choice, not a calibrated latency for our arms.
- `third_party/DynamicVLA/core/train.py`, `run.py`, `configs/dynamicvla.yaml`:
  generic training path. The YAML's old dataset default must be overridden.
- `third_party/DynamicVLA/scripts/create_lerobot_dataset.py`: camera, Cartesian
  state/action and environment-state schema. It expects simulation-specific
  HDF5 and scene metadata; real recordings need a dedicated exporter.
- `third_party/Data_Collection_Dynamic_Objects/dom_tracking/controller.py`:
  our existing **simulation hover** expert; not a real-robot driver.

## Hardware observed on this PC

Read-only device queries and short camera captures found:

| Item | Observation | Consequence |
|---|---|---|
| GPUs | Two RTX A6000, 96–99% utilization, about 35 GiB used each at inspection | No CUDA model, Isaac Sim, or new training job in this prototype |
| RealSense `252122300709` | D455, USB 3.2; low view across the table | Previous orange-ball front view; not selected for the blue-tape pilot |
| RealSense `234222300865` | D455, USB 2.1; user repositioned the overhead camera to a front/downward view | Selected stationary blue-tape view; requires fresh camera-to-right-base calibration; use USB 3 for later synchronized collection |
| Robot USB | Four CH34x serial adapters under `/dev/serial/by-id` | Serial numbers alone do not identify leader/follower or left/right |
| Calibration files | Existing SO101 follower and leader files, including left/right IDs | Reuse only after matching physical arm, port, units and motor zeroes |

The observed local cameras are **D455**, not D405. The user's D405 designation
may refer to the remote ALOHA PC. Inspect that PC separately; “D405-113” is not
used as a model identifier in code. The [D405 product documentation](https://www.realsenseai.com/products/stereo-depth-camera-d405/)
lists an ideal 7–50 cm range, so a distant front-facing D405 needs a placement
check. Do not copy camera intrinsics/extrinsics or depth limits between models.

Raw local evidence is under `scratch/real_robot/research/` and
`scratch/real_robot/camera_probe/`; images, device identifiers, calibration files,
and recordings are not added to version control. GPU training was not stopped,
reconfigured, or imported by the prototype.

## First runnable test: perception only

The selected pilot uses a **blue tape roll, stationary front/downward camera, and the right
SO101 follower** for eventual manipulation. Current live capture only observes
the tape; it does not connect to any arm. A saved right-follower calibration named
`my_awesome_follower_arm_so101_right.json` exists. Its six motor IDs and numerical
ranges pass a file-level check, but neither its correspondence to the physical
arm nor the right follower's USB port has been verified. The audit is saved in
`scratch/real_robot/right_calibration_audit.json`. Do not recalibrate or enable
torque based solely on that file's presence.

From the repository root, use the existing **separate** RealSense environment:

```bash
ROBOT_PY=/home/ailab-students/miniconda3/envs/realsense/bin/python
"$ROBOT_PY" -m real_robot devices
"$ROBOT_PY" -m real_robot --help
"$ROBOT_PY" -m unittest discover -s real_robot/tests -v
nice -n 10 "$ROBOT_PY" -m real_robot demo --output scratch/real_robot/demo-new
```

`demo` generates noisy synthetic motion, a dropout, and an outlier, then runs the
tracker and hover controller. It reports synthetic velocity error and candidate
commands; it is not a real-robot result. Output directories must be new.
The CLI disables CUDA and limits NumPy/BLAS threads; the camera path also limits
OpenCV to one thread. Nothing is installed into the policy/train environments.
`requirements.txt` is for a separate future perception environment only.

For the selected blue-tape test, place one roll fully inside the tabletop ROI,
then run from this PC's graphical terminal:

```bash
ROBOT_PY=/home/ailab-students/miniconda3/envs/realsense/bin/python
nice -n 10 "$ROBOT_PY" -m real_robot track \
  --serial 234222300865 --fixed-camera --detector blue_tape \
  --fps 15 --seconds 30 --preview \
  --output "scratch/real_robot/blue-oblique-$(date +%Y%m%d-%H%M%S)"
```

Keep the tape still for the first 10 seconds, translate it slowly across the
table for 10 seconds, then stop and reverse direction for the remaining time.
Keep the camera fixed. Press `q` to end early. This sequence checks tracking
behavior; an independent reference is still needed for moving-speed accuracy.
Add `--save-rgbd` if replay images/depth are needed (extra CPU and disk cost).

The live preview shows the target outline, a cyan 1.5-second measurement trail,
and a magenta arrow representing an approximate image projection of 0.25 seconds
of estimated 3D motion. The side panel displays position, signed XYZ velocity,
speed, readiness, image FPS, and processing time (excluding rendering and camera
waits). Motion along the viewing ray can produce little arrow displacement;
check the Z velocity as well. Without either frame option, XYZ uses camera
axes: X right, Y down, Z away from the camera. Static noise can produce small
arrows. Add `--roi X Y W H` only after checking the current camera view.

For an approximate display in the operator's tabletop axes near the right base,
add `--arm right --provisional-landmarks PATH/landmarks.json` with at least three
non-collinear correspondences from the unchanged camera mounting. This rotates
the metric velocity into `right_base_estimate`; the preview and manifest mark it
unvalidated, and `frame_estimate.json` records fit errors and the source hash.
These axes have not been verified against the SO101 kinematic base frame. The
shadow controller accepts only `robot_base`, so this display estimate cannot
enable control. Use `--calibration PATH/calibration.json` for a validated
camera-to-base transform instead; the two options are mutually exclusive.

Upright rolls are supported without a circular-silhouette requirement. Clipped
blue cables/objects at image or ROI boundaries are excluded before the ambiguity
check, so they do not suppress a fully visible upright roll. Two competing
fully visible blue objects still require a tighter ROI or unique marker.

The default tracker also infers stillness from a one-second window of accepted
3D positions, requiring at least eight samples and 0.8 seconds of coverage.
A fitted speed below 3 mm/s with residual spread below 6 mm enables a soft
zero-velocity Kalman update (3 mm/s measurement standard deviation). It changes
the filter state and covariance, rather than hiding velocity in the display.
`stationary_assumed` records this inference, and the preview labels it. Raw
position measurements remain available in the log.

The constraint releases when the latest three-sample mean moves over 4 mm from
the fixed anchor, window speed exceeds 6 mm/s, or spread exceeds 6 mm. Missing
or rejected measurements clear it. Slow starts can take up to roughly a second
to release in synthetic tests; motion below the 3 mm/s entry threshold can be
mistaken for stillness. This is a heuristic motion prior derived from the same
positions, so its conditional covariance and reduced speed jitter do not prove
physical accuracy. Use `--disable-stationary-update` for an unmodified-filter
comparison. Parameters are recorded in the run manifest.

`blue_tape` uses OpenCV HSV bounds `[90,60,45]`--`[135,255,255]`. It preserves
the tape roll's hole in the mask so table/background depths are excluded; it
also supports a strip of tape without imposing a circular shape. It rejects
comparable multiple candidates and silhouettes clipped by the image or ROI. Lighting can
require explicit `--hsv-low` / `--hsv-high` overrides or a tighter
`--roi X Y W H`. These geometric/color checks do not establish semantic identity;
another blue object can still pass. The ROI above covers the central tabletop
in the inspected view; adjust it if you change framing or test elsewhere.
The 3D measurement remains a **visible-surface proxy**, not a calibrated grasp pose.
No marker is required for this pilot. The marker and generic HSV options below
are alternative diagnostics.

The previous `orange_ball` mode remains available with its original color and
roundness checks. Old captures retain their original camera and object labels.

For a uniquely identified object, attach an ArUco `DICT_4X4_50` marker ID 0 to a
rigid, graspable object. A marker image was generated at
`scratch/real_robot/marker0.png`; print with the white margin intact. Metric
position here comes from depth, so marker size is not an input. The tracked
point is a visible marker/surface proxy, not an estimated center of mass.

```bash
# Generate a new marker file if needed.
"$ROBOT_PY" -m real_robot marker --id 0 --output scratch/real_robot/marker-new.png

# Confirm the selected camera is stationary; move the object by hand initially.
# This opens only the named camera. It never opens a serial port or moves an arm.
nice -n 10 "$ROBOT_PY" -m real_robot track \
  --serial 234222300865 --fixed-camera --detector aruco --marker-id 0 \
  --seconds 30 --preview --save-rgbd \
  --output scratch/real_robot/moving-marker-001
```

Without a printer, select a uniquely colored object and a tight tabletop ROI:

```bash
nice -n 10 "$ROBOT_PY" -m real_robot track \
  --serial 234222300865 --fixed-camera --detector hsv \
  --hsv-low 0 120 80 --hsv-high 10 255 255 \
  --roi 175 395 60 80 --seconds 30 --preview \
  --output scratch/real_robot/color-test-001
```

The example ROI is a small red object in the inspected overhead image; **expand
or change it for the intended moving object**. HSV examples cover only one hue
interval (red also wraps near hue 179); tune bounds for actual illumination.
Comparable competing blobs are rejected. RGB/depth outliers and invalid depth
are rejected; this is not a semantic general-purpose tracker. A marker may
occlude/disappear during grasping, so it is a first-test aid rather than the
final object-tracking solution.

Files per capture:

- `manifest.json`: camera, detector, limits, coordinate frame and calibration.
- `tracking.jsonl`: sensor and host arrival times, intrinsics/distortion, RGB/depth
  skew, measurement/quality, filtered XYZ/velocity/covariance, status, and null action.
- `last_frame.jpg`: annotated last view. `--save-rgbd` additionally retains aligned
  BGR/depth arrays per frame for debugging; this adds disk/compression cost.
- `summary.json`: frame counts and CPU processing timings. These exclude capture
  waiting and recording overhead, and are **not end-to-end sensor latency**.

Filtering uses monotonically increasing relative **sensor** timestamps; host
arrival is logged on a separate clock. Frame clock changes/duplicates abort the
capture. No cross-camera synchronization or sensor-to-host latency calibration
is claimed. A dropped/rejected measurement prevents a controller candidate;
long gaps reset velocity and require reacquisition. No detected object is an
expected `not visible` state, not a successful tracking test.

## Coordinates, calibration, and controller design

Camera coordinates are the RealSense optical frame (right, down, forward), in
meters. Uncalibrated captures remain in `camera_optical`, never `robot_base`.
For the static front/overhead camera, measure at least four non-collinear paired
landmarks distributed across the intended workspace, plus at least three
**independent held-out landmarks**. Use robot-base XYZ and synchronized camera
XYZ of the same physical point; account for tool-tip and marker offsets.

The overhead camera was repositioned for this pilot. **Do not reuse any transform
from its previous mounting**, even though its serial is unchanged. Moving the
camera does not, by itself, require repeating the arm's motor calibration.

To begin a fresh interactive camera calibration, use a desktop terminal:

```bash
ROBOT_PY=/home/ailab-students/miniconda3/envs/realsense/bin/python
CALIB_RUN="scratch/real_robot/right-oblique-calibration-$(date +%Y%m%d-%H%M%S)"
nice -n 10 "$ROBOT_PY" -m real_robot collect-landmarks \
  --serial 234222300865 --arm right --fixed-camera --output "$CALIB_RUN"
```

Press **F** to freeze a view, click a known surface point, then enter its
independently measured **right-arm-base X Y Z in meters** in the terminal.
The same physical point must be measured in both frames. Use a surveyed fixture
or a verified robot TCP/FK with its offsets accounted for; arbitrary tabletop
coordinates are not automatically robot-base coordinates. If base coordinates
are not yet available, end with **Q** and establish the base frame first.

Collect at least four non-collinear fit points (six or more distributed points
are preferable); press **V** for at least three distinct held-out validation
points. Press **U** to undo, **Q** to end. Keep the camera, base and physical
landmarks fixed during each observation. Click a continuous surface away from
the tape's hole and depth edges. The tool records RGB/depth, timestamps and point
membership after each accepted entry, and never opens a motor connection.
It does not estimate base coordinates or certify calibration automatically.

To continue the saved three-point session while preserving its original files:

```bash
nice -n 10 "$ROBOT_PY" -m real_robot collect-landmarks \
  --serial 234222300865 --arm right --fixed-camera \
  --resume-from scratch/real_robot/right-calibration-20261016-164733/landmarks.json \
  --output "$CALIB_RUN"
```

The output directory must be new. The tool copies the saved points, partition
membership, original document and RGB-D files, checking file checksums. Only
resume if camera mounting and reference axes are unchanged. Aim for 6–10 fit
and 3–4 independent held-out points, including measured height variation where
possible. Remeasure inaccurate existing coordinates; more points cannot repair
incorrect origin/axis definitions. Approximate image-aligned tabletop axes do
not establish the SO101 kinematic base frame.

Every metadata save also goes to a separate `/tmp/dynamicvla-landmark-recovery-*`
directory, printed at startup. A pending point's exact coordinates are written
there before its RGB-D file is saved on the SSD, so an SSD failure does not lose
the terminal entry. The pending record does not establish that its RGB-D file
was saved. This temporary copy is metadata only and does not survive reboot.

After collecting actual point pairs, fit and validate in the same terminal:

```bash
"$ROBOT_PY" -m real_robot calibrate \
  --landmarks "$CALIB_RUN/landmarks.json" \
  --max-error 0.005 --output "$CALIB_RUN/camera-to-right-base.json"
```

Use the resulting transform only if `calibration_validated` is true and the
physical frame checks below pass. Camera calibration and right motor calibration
are separate steps; the physical right follower's USB mapping is still unverified.

```bash
# Fill measured arrays in a copy of configs/landmarks.example.json.
"$ROBOT_PY" -m real_robot calibrate \
  --landmarks scratch/real_robot/landmarks-measured.json \
  --max-error 0.005 --output scratch/real_robot/front-calibration.json
# Add --calibration scratch/real_robot/front-calibration.json to a later track run.
```

The fit is rigid SE(3), with no arbitrary scale, and rejects degenerate geometry.
The example 5 mm threshold is a proposed pilot target, not a measured camera
accuracy. Fit and held-out residuals must both pass; static mounting and camera
serial must match. Revalidate after moving the camera or robot base. A passing
landmark fit alone does not validate robot TCP offsets, joint calibration, or
collision clearance.

For the wrist camera the required transform is
`T_base_camera(t) = FK(q(t)) @ T_tool_camera`. Synchronize/interpolate joint
feedback to image exposure time, transform each 3D measurement, **then** fit
velocity. Differentiating wrist-frame XYZ directly introduces arm motion into
object velocity. Wrist hand-eye calibration, timestamped FK, and two-view fusion
are planned; the current CLI deliberately requires a stationary camera.

The implemented constant-velocity filter maintains `[p, v]` and covariance with
variable time steps, innovation gating, speed/uncertainty limits, and warm-up.
The first predictor is `p_future = p + v * (age + horizon)`. Curved-motion models
and CPU optical flow/learned masks can be compared later using the same interface.
The target point's surface/marker offset to the intended grasp must be calibrated;
unknown object orientation is not silently filled with zeros.

`ShadowHoverController` computes velocity feedforward plus proportional position
feedback toward the predicted point with a vertical hover offset. It bounds
speed/acceleration and rejects out-of-workspace, stale, uncalibrated, or uncertain
inputs. Its workspace, grasp priors and horizon must be measured/tuned for the
actual robot. The pure DLS helper is a kinematics reference, not a hardware
adapter or collision checker. The live camera CLI has no measured end-effector
state, so controller candidates are currently exercised in the synthetic demo.
There is no `--enable-motion` switch.

## Staged plan for SO101 here, then Mobile ALOHA

| Stage | Work | Evidence before advancing |
|---|---|---|
| 1. Moving-object perception | Fixed camera, tabletop object moving laterally, pauses, reversals, brief occlusion; save RGB-D and telemetry | Inspect XYZ units/sign, stationary speed jitter, ready/dropout rate, reacquisition, and velocity against an independent known-motion reference |
| 2. Robot/base geometry | Confirm follower/leader ports, reuse verified joint calibration, obtain correct URDF/TCP, camera-to-base and wrist hand-eye transforms | Held-out landmark errors, FK checks against measured tool poses, joint/image timing, workspace/keep-out verification |
| 3. Shadow following | Feed read-only joint feedback into FK and the hover planner; log IK solutions without dispatch | Reachability and joint limits throughout the path, predicted lag, bounded command differences, no stale-target outputs |
| 4. Physical hover | One follower, empty gripper, low speed, object well inside workspace; parked second arm and stopped mobile base | Operator-supervised start, working stop/watchdog, bounded tracking error and clearance under target loss |
| 5. Dynamic pick/place | Add descend/close/lift/place/reset stages and grasp/drop detection; first slow motion, then speed/direction sweeps | Repeated grasp/placement success from observed state, explicit failure labels, no timer-only success assumption |
| 6. Expert dataset + adaptation | Record synchronized wrist/front views and robot/command/target state; export and fine-tune matched baselines | Validated schema/action reconstruction, held-out episodes/configurations, offline checks, then separate robot evaluation |

For stage 1 start with a single rigid, textured or marked object moved by hand
at modest speed; a guided slider/known trajectory is better for quantitative
velocity accuracy. Proposed checks include stationary velocity p95 below
0.02 m/s and position jitter below 5 mm, plus a moving-path velocity reference;
measure these before accepting them as achievable. Do not call a hand-moved
sequence ground-truth velocity. Measure stop/reversal response and report the
filter's lag. Training continues during CPU tests; do not change its GPU limits.

For SO101, use the [LeRobot SO101 setup](https://huggingface.co/docs/lerobot/en/so101)
and the actual calibrated follower. Its limited task-space freedom means full
6D arbitrary pose commands are inappropriate: solve position plus feasible
orientation constraints with calibrated kinematics, joint limits and collision
checks. Do not equate LeRobot normalized motor positions with radians or meters.
Opening a robot SDK can change torque state, so read-only feedback requires a
reviewed backend rather than merely calling a default `connect()`.

For Mobile ALOHA, use the same tracker/planner/recorder with a separate robot
adapter for the confirmed Interbotix model and installed ROS stack. Verify the
[ALOHA hardware/bringup documentation](https://docs.trossenrobotics.com/aloha_docs/2.0/operation/bringup_shutdown.html)
against the actual machine; do not assume model, ROS version, or joint names.
Start with one follower, the other parked, and the mobile base stationary. Arm
kinematics and camera extrinsics are independent of the SO101 configuration.
Bimanual control or a moving base is a separate extension with a common frame,
base localization and collision handling, not part of the initial single-arm test.

The grasp expert should use observable transitions:

| State | Target/action | Transition/abort evidence |
|---|---|---|
| Acquire | No descent; establish a stable target | Valid target history and reachable hover pose |
| Approach/hover | Continuously predict interception, open gripper | Measured position/orientation alignment and acceptable uncertainty |
| Descend/intercept | Object-specific grasp offset and motion-aware approach | Verified clearance and contact-region alignment; loss => hold/retract policy |
| Close | Calibrated gripper closure | Gripper feedback plus visual/contact evidence, not elapsed time alone |
| Lift/verify | Bounded lift | Object follows tool and clears the table; otherwise failed grasp |
| Transport/place | Known destination with clearance | Measured arrival, release, and observed object at destination |
| Reset | Validated return path | Confirmed home pose before another episode |

This full state machine, robot-specific IK/collision/dispatch, stop handling and
expert-episode recorder/exporter remain to be implemented after hardware and
calibration checks. Existing simulation code supplies task logic, not validated
real-robot limits. Current code is explicitly a first perception/control prototype.

## Dataset and post-training contract

Record native timestamps; do not pretend 15 FPS capture is 25 FPS. Later profile
both cameras at the selected rate and define causal synchronization/resampling.
Keep pre-processing delays and command issue/feedback times. A proposed raw
expert record includes:

| Field group | Required contents |
|---|---|
| Images/calibration | Wrist and front RGB, depth sidecars, native timestamps, intrinsics, camera-to-base/hand-eye calibration hashes |
| Robot feedback | Joint positions/velocities in documented units, gripper state, measured TCP pose and timestamps, robot/model/TCP versions |
| Commands | Requested Cartesian target, constrained/issued joint command, issue time, saturation/rejection flags, controller state |
| Object estimate | Causal position/velocity/covariance, source view, target identity, age/visibility/quality, frame, grasp-prior offset |
| Episode | Language instruction, task/reset/speed condition, success/abort reason, manual intervention, source/config hashes |

For the first single-arm DynamicVLA adaptation, preserve `wrist_cam` and
`opst_cam` names, a verified Cartesian state/action convention, and a gripper
mapping, then apply normalization consistently. Retain joint-space commands
and measured poses separately. Decide and test whether the training action is
the issued Cartesian target or a future measured pose; our old hover dataset
used future measured poses, whereas the official converter reads saved `action`.
They are not interchangeable. Apply absolute-to-delta conversion exactly once.
Never substitute SO101 joint arrays for pretrained Cartesian state just because
the vector lengths happen to match. Bimanual ALOHA requires a deliberate
state/action-head/schema change and a new adaptation experiment.

Depth and an optional collection-only tag/tracker may assist the expert, but
baseline policies use RGB, instruction and robot state. KRFM variants require
an explicitly specified **causal velocity source at deployment too**. Preserve
velocity estimates/quality as side information, never simulator ground truth.
Do not require a fake 6D object pose to export the observed 3D proxy; update the
adapter/schema to mark unavailable orientation rather than invent labels.

Collect a small pilot first, validate all fields/video alignment/action replay,
then scale across initial positions, speeds, directions and object instances.
The author's roughly 200 episodes/configuration is a reference, not a guarantee
for our hardware. Split by whole episodes/configurations; keep evaluation trials
out of checkpoint selection. Retain failures for diagnosis and select accepted
expert episodes with a declared rule.

Post-train matched DynamicVLA, velocity-conditioned DynamicVLA and KRFM variants
on the same real-robot split and initialization. Start with short offline runs
on a separate allocation **after** data validation and the current training job;
check normalization, causal inputs, forward loss and action reconstruction
before a training campaign. The generic released trainer is reusable after a
real-data adapter is verified; there is no ready-to-run official ALOHA recipe.

## Verification completed on October 16, 2026

- 26 unit tests pass in the existing RealSense environment: variable-rate motion,
  outliers, dropouts, reacquisition, ambiguous masks, missing depth, calibration,
  wrist-frame transformation, control limits, bounded camera startup pairing,
  orange-ball shape/color checks, blue-tape hole preservation/ambiguity/clipping,
  and clicked-landmark depth rejection and held-out partitioning.
- The synthetic 240-frame demo exercises motion, an occlusion and an outlier;
  it produced 211 shadow candidate steps. This is algorithmic testing only.
- Both D455s produced aligned RGB/depth in short capture probes.
- With the repositioned camera, `scratch/real_robot/blue-oblique-smoke-02`
  recorded 56 frames, 49 accepted blue-tape measurements and 42 filter-ready
  frames. Processing median/p95 was 3.14/8.07 ms. The inspected final image
  outlines the tape roll, excluding its central hole. This is a short perception
  check, not measured motion accuracy or completed camera-to-arm calibration.
  The first full-view attempt is retained as `blue-oblique-smoke-01` (9 accepted
  measurements in 58 frames); the second uses the tabletop ROI and a lower
  saturation threshold to include the roll's lighter blue surface.
- `scratch/real_robot/orange-front-smoke-01` captured 61 front-camera frames
  with the orange-ball detector (processing median/p95 4.91/6.77 ms). The saved
  view contained no visible orange ball, and there were zero accepted target
  measurements. This verifies capture operation only; ball tracking and motion
  accuracy remain to be tested with the ball present.
- `scratch/real_robot/live-static-smoke-03` completed with 58 accepted measurements,
  51 filter-ready frames, and approximately 14.98 sensor FPS on the overhead
  camera. Median/p95 perception processing time was 3.34/6.35 ms (excluding
  acquisition/recording). Four startup pairs were discarded and one frame wait
  timed out before valid pairing. Accepted pair skew was at most 0.067 ms.
- The nominally stationary red object's estimated speed p95 was **0.044 m/s**,
  above the proposed 0.02 m/s pilot target; XYZ standard deviation was roughly
  0.76/1.67/1.44 mm. This test establishes pipeline operation, **not readiness
  for dynamic grasping** or accuracy against an independent motion reference.
- Offline lower-process-noise trials on that same short recording reduced
  speed p95 to 0.028–0.038 m/s but still missed the proposed target. Defaults
  were retained pending longer stationary and moving-object recordings so we
  do not optimize only for static scenes at the expense of response latency.
- Two earlier failed startup attempts remain preserved beside that run. The
  camera reader now waits at most 10 s initially / 3 s subsequently for a pair
  within 40 ms, without loosening the skew criterion or resetting the camera.
- No serial ports were opened, no torque/position commands were sent, and no
  existing environment was modified. Both training GPUs were still at 99–100%
  utilization with their previous memory usage at the final check.

The user changed the target to blue tape, repositioned camera `234222300865`,
and selected the right follower. Earlier left-arm audit records are historical.
Next physical steps are the tape pause/translation/reversal sequence above and
matching the saved right calibration to the physical follower and USB port.
Fresh measured camera-to-right-base landmarks are required before robot control.
The collector and tracker default to `--arm right`; use `--arm left` explicitly
for a separate left-arm session. Fitted transforms retain the arm label, and
tracking rejects transforms for another arm or with no arm label.
The remote ALOHA PC and its cameras have not been inspected in this session.
