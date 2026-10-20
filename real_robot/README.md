# Real-robot dynamic-object collection: CPU prototype and implementation plan

For the current workstation startup, commands, tuning and shutdown procedure,
use [the right-arm tracking runbook](README_TRACKING.md).

The original CPU path starts the SO101-to-ALOHA collection effort without using
the GPUs running DOM training. The optional EfficientTAM path explicitly uses
GPU inference on the current workstation. The module implements **stationary-camera RGB-D object tracking,
causal velocity estimation, calibration utilities, and a model-based hover
controller evaluated in shadow mode**. The perception CLI has no robot-command
backend. Separate right-ALOHA calibration capture and single-hover verification
modules can send explicitly enabled motor goals. The separate continuous
right-arm servo follows object positions using Cartesian PID and analytic IK.
Camera-only captures and synthetic demos are explicitly marked `expert_episode:
false`; they cannot be counted as grasp demonstrations.

## Continuous tracking: current workflow

Use two terminals from `/home/ailab/Desktop/Kinematic-Residual-VLA`:

```bash
# Terminal 1: EfficientTAM, front/side 3D fusion and object velocity.
DISPLAY=:1 .venv/bin/python -m real_robot.tracking_session perception --config real_robot/config.yaml

# Terminal 2: right-arm continuous position following.
DISPLAY=:1 .venv/bin/python -m real_robot.tracking_session follow --config real_robot/config.yaml --execute
```

No motion-plan JSON is needed for `follow`. The launcher selects the latest
perception session automatically. `--observation /path/to/observation.json`
overrides that choice. Omit `--execute` for read-only arm feedback and preview.
With the default `seed_session: null`, draw a box around the tape and press
Enter in each camera window at startup. To deliberately reuse prompts, set
`seed_session` to a prior successful session directory. Explicit paired
`--front-seed-box X Y W H --side-seed-box X Y W H` flags also work.
The follower must already be holding with healthy torque; only one process
may own its serial port. The leader-to-held-follower startup is described below.

Edit the **`servo:`** section of [`config.yaml`](config.yaml): `kp`, `ki`, `kd`,
`max_tcp_speed_mps`, `max_joint_speed_deg_s`, `target_smoothing_s`, workspace
bounds and run duration. The current YAML uses P control (`kp=2`, I/D zero), a
**0.5 m/s TCP command ceiling** and a ten-minute session. The pilot joint-rate,
acceleration and command-lead caps are disabled (`null`); joint rates use the
existing motor velocity/profile limits read at startup. Workspace bounds apply
to the TCP in right-base coordinates; analytic IK also checks reach and joint
limits, so the entire bounding box is not necessarily reachable. The desired
TCP is 5 cm above the estimated visible object top. Masked depth provides the
top height; a configurable proxy-to-top offset is the fallback. Base calibration
and the 160 mm fingertip offset remain provisional.

Each cycle uses `v = kp*error + ki*integral - kd*measured_TCP_velocity`, bounds
velocity, converts the next position to analytic IK,
and bounds joint increments and FK displacement. Startup preserves the existing
hardware goals, including gravity sag. Position control uses fresh associated
3D observations independently of velocity-filter readiness, with target smoothing
across contributing-view changes. A faster object can leave the arm behind;
it does not itself trigger the old pilot speed/3 cm displacement stop.

Stale/unreachable/out-of-workspace targets hold and automatically resume when
valid again. Joint following errors, hardware faults, temperature/current guards
and measured geometry violations stop the session. `Q` in the arm window stops
new goals and retains torque; the camera window is independent. The controller
checks robot self/table geometry and requires a clear physical workspace; it
does not implement arbitrary obstacle avoidance or automatic powered parking.

Motor gains, modes, torque and hardware current/shutdown limits are unchanged.
`experimental_current_guard: false` removes the pilot's software current
thresholds; existing motor current limits, hardware faults and the 55°C
temperature check remain active. A 12° following-error guard detects severe
tracking divergence. The tool points 60° downward to reduce static shoulder
load while following the fingertip midpoint; this is position tracking.

DynamicVLA's real-world object release range is 0.2–0.8 m/s; it does not provide
a drop-in VX300 controller/PID preset. These are our controller settings, with
the requested 0.5 m/s command ceiling. Physical speed and tracking accuracy at
that ceiling have not been validated. Base calibration and fingertip geometry
remain provisional; the controller uses their FK estimate for speed limiting.
An encoder/FK speed watchdog stops new goals if measured speed exceeds the
configured ceiling. It reacts after a measured excursion and cannot guarantee
an instantaneous physical speed bound. In the first uncapped trial, estimated
physical speed briefly reached 0.63 m/s despite the 0.5 m/s command ceiling;
this prompted the additional readback watchdog. That trial ended at the 55°C
shoulder temperature guard, with holding torque retained.

## Legacy small-workspace pilot with an explicit plan

The earlier `arm` mode below is retained for explicit trajectory experiments.
It is separate from the continuous `follow` mode above. Its `control` settings
use internal conservative defaults unless a YAML `control:` section is supplied.
Edit [`config.yaml`](config.yaml) for the camera rig, initial target prompts,
run duration and software Cartesian PID. The PID computes
`v = kp*position_error + ki*integral - kd*TCP_velocity`; the default is P control
(`kp=1.5`, `ki=kd=0`). These gains do not change Dynamixel motor registers.
Speeds use m/s, acceleration m/s², joint speed degrees/s. The current pilot
rejects settings above 0.015 m/s TCP speed, 3°/s joint speed or 0.05 m/s object
speed. Its desired hover is 50 mm above the estimated visible object top.
Absolute base calibration and fingertip offset remain provisional.

From the project directory, terminal 1 starts **camera-only** EfficientTAM,
fusion and velocity reporting, using the previously selected blue-tape prompts:

```bash
cd /home/ailab/Desktop/Kinematic-Residual-VLA
DISPLAY=:1 .venv/bin/python -m real_robot.tracking_session perception --config real_robot/config.yaml
```

Each launch prints its unique output directory and `observation.json` path.
A fresh scene-reviewed arm plan must reference that exact live observation
file, matching rig hash, current arm pose, collision model and workspace.
Plans expire after five minutes and cannot survive a computer reboot.
Editing a plan's approval flags alone does not constitute reviewing a fresh
scene or checking the path.

Terminal 2 first validates a fresh plan offline, with no motor access:

```bash
.venv/bin/python -m real_robot.tracking_session arm --config real_robot/config.yaml --plan /absolute/path/to/reviewed-plan.json
```

Then, with the right follower already holding its position and teleoperation
stopped, the explicit execution command is:

```bash
DISPLAY=:1 .venv/bin/python -m real_robot.tracking_session arm --config real_robot/config.yaml --plan /absolute/path/to/reviewed-plan.json --execute-held-start
```

Only one process may own the right follower serial port. For routine startup,
use the existing LeRobot leader teleoperation to lift the resting follower,
then stop it with Ctrl+C. The inspected LeRobot disconnect closes serial ports
without disabling follower torque. An October 20 hardware check verified all
nine motors stayed enabled and fault-free after this handoff. LeRobot's
**connect** does briefly disable torque, so do not restart that command with
an unsupported elevated follower. The tracking executor's held-start path
preserves existing holding goals and does not change torque or motor modes.
Normal tracking stop retains the last goals and torque. A powered parking
trajectory still needs a fresh scene check and physical validation; fault or
power-loss recovery may require support.

The preview shows estimated XYZ velocity, total speed in m/s and cm/s, and
[DynamicVLA v2 Appendix D](https://arxiv.org/html/2601.22153v2)'s real-world
release range of **0.2–0.8 m/s** (evaluation configurations 0.3/0.5/0.7 m/s).
That paper range is a perception reference; the arm pilot is much slower.
Velocity is a causal Kalman estimate of a visible-surface proxy, rather than
independent centroid ground truth. Its covariance excludes shape bias and
unmeasured latency. Unsynchronized camera span is capped at 35 ms for the
slow pilot (up to 1.75 mm displacement at 0.05 m/s); this does not validate
fast DynamicVLA-speed collection. Stale feed, workspace exit, following error,
current or hardware faults stop new tracking progression.

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

The table records the earlier machine inventory. The October 20 `/home/ailab`
workstation instead reports one 24 GB RTX 4090; its free memory was checked
before running the optional EfficientTAM model. Existing GPU jobs were not stopped.

| Item | Observation | Consequence |
|---|---|---|
| GPUs | Two RTX A6000, 96–99% utilization, about 35 GiB used each at inspection | No CUDA model, Isaac Sim, or new training job in this prototype |
| RealSense `252122300709` | D455, USB 3.2; low view across the table | Previous orange-ball front view; not selected for the blue-tape pilot |
| RealSense `234222300865` | D455, USB 2.1; user repositioned the overhead camera to a front/downward view | Selected stationary blue-tape view; requires fresh camera-to-right-base calibration; use USB 3 for later synchronized collection |
| Robot USB | Four CH34x serial adapters under `/dev/serial/by-id` | Serial numbers alone do not identify leader/follower or left/right |
| Calibration files | Existing SO101 follower and leader files, including left/right IDs | Reuse only after matching physical arm, port, units and motor zeroes |

The original survey above predates the current workstation setup. The October
19–20 workstation has a front D455 and three D405 cameras, identified below by
serial number. “D405-113” is not used as a model identifier in code.
The [D405 product documentation](https://www.realsenseai.com/products/stereo-depth-camera-d405/)
lists an ideal 7–50 cm range, so a distant front-facing D405 needs a placement
check. Do not copy camera intrinsics/extrinsics or depth limits between models.

Raw local evidence is under `scratch/real_robot/research/` and
`scratch/real_robot/camera_probe/`; images, device identifiers, calibration files,
and recordings are not added to version control. GPU training was not stopped,
reconfigured, or imported by the prototype.

## First runnable test: perception only

### Blue-tape hover target preview on the current workstation

The tracking CLI can display a target **5 cm along base +Z above the detected
visible tape surface**, with a 0.10 s constant-velocity prediction. XYZ depth
changes are tracked, rather than assuming the object remains on the table.
Missing, ambiguous, rejected or unready measurements produce no target. This is
a camera-only preview: it does not open motor ports or send goals, perform IK,
or establish that a target is reachable or collision-free. Camera/USB transit
latency is unmeasured. The ChArUco fit remains explicitly provisional.

```bash
DISPLAY=:1 PYTHONPATH=. .venv/bin/python -m real_robot track \
  --serial 234222300865 --fixed-camera --detector blue_tape \
  --candidate-calibration scratch/real_robot/right-handeye/session-20261019-03/candidate-calibration.json \
  --hover-height 0.05 --hover-lookahead 0.10 --disable-stationary-update \
  --fps 15 --seconds 300 --preview --live-frame \
  --output scratch/real_robot/blue-hover-next
```

Use a new output directory for each run. Q closes the preview. The full image
is retained; the operator can supply their own `--roi X Y W H`. JSONL records
separate `hover_preview.target_m` from `action: null` and `motion_sent: false`.
The yellow cross uses an approximate pinhole projection. `live_frame.jpg` is
updated once a second for external inspection.

Before the supported restart on October 19, both shoulder motors (IDs 2 and 3) reported hardware
error 32 with torque disabled after the grid-check overload. The read-only
snapshot is in `scratch/real_robot/blue-hover-20261019-01/motor-health-readonly.json`.
No faults were cleared or motors re-enabled. Live robot following remains
blocked pending physical support, overload diagnosis and recovery, and a fresh
motion plan. A displayed target is not evidence of robot-following execution.

### Supervised single tape hover after fault recovery

`python -m real_robot.aloha_tape_hover --plan PLAN.json --output NEW_DIRECTORY`
validates a separately checked right-arm path without opening hardware. Adding
`--execute-supported-start` enables holding torque only after verifying a healthy,
fully torque-off, user-supported arm, fresh matching arm/tape positions, operating
modes, paired encoders and local holding-goal readbacks. The observed torque-off
goal register follows each motor's own encoder, including one-count fluctuations;
the check allows two counts, rather than requiring a shadow motor's register to
equal its master's encoder. The hardware probe and first rejected preflight are
preserved under `scratch/real_robot/blue-hover-20261019-04`.

This executor approaches one static tape target at at most 3 degrees/s nominal,
with bounded feedback compensation (1.5 degrees) and a 3.5 degrees/s goal slew
limit. It checks all nine motor currents, hardware faults, torque state and
temperature; joint error; the mesh model including camera housing; camera
freshness; and the tape remaining within 15 mm of the checked target. Both
shoulder motors are monitored independently. Software current thresholds are
conservative abort choices, not measured safe operating envelopes or changes to
the motors' protection settings. No reboot, gain, mode or protection-limit writes
are performed. The separate perception CLI still has no motor backend.

The original 2-degree parked-pose allowance is retained. The first post-restart
pose exceeded it (model elbow 94.13 degrees versus a 92-degree upper limit), so
no torque activation was attempted from that pose; the operator slightly
unfolded and supported the arm before replanning. Scene checks document robot
self observations separately from a nearby fixed mount which the initial path
moves away from. Omitted self pixels do not certify free space. The calibration
and 160 mm fingertip offset remain approximate, with the prior R3C3 visual check
recorded as supporting evidence rather than a declaration of exact calibration.

Q stops generating new goals and retains the last bounded goal. A fault can
independently disable torque. The controller does not automatically release a
loaded arm or replay a trajectory after a fault. It records `status.json`, joint
and current telemetry, `reached.json` only after settling, and live/final images.
After reaching, monitoring continues for up to five minutes. This is a single
hover verification, **not moving-object following** or an expert grasp episode.

For a separately checked continuation from a healthy loaded hold,
`--execute-held-start` preserves torque and initializes the servo from the
existing hardware goal and its load compensation. It never replaces that goal
with measured position on restart. The two execution flags are mutually
exclusive. A plan may select the calibrated right wrist camera; the observation
transform is then `FK(current_joints) @ T_gripper_wrist_camera`, updated from live
feedback, rather than treating a wrist camera as fixed. Joint state is recent
host feedback, not hardware-synchronized to camera exposure; latency remains
unmeasured.

The front approach was stopped when its partial view shifted the tape's proxy.
The wrist camera saw the complete roll, but its default saturation threshold
also selected a blue-tinted tabletop reflection after auto-exposure settled.
The wrist-specific plan uses HSV `(90,150,45)` through `(135,255,255)`; a stationary
180-frame check accepted all 180 depth measurements. This does not alter the
front detector or apply a crop. `perception.jsonl` retains camera metadata,
quality, joint state used for compensation and filtered estimates. A resumed
loaded hold is tested specifically to retain its existing goal; a synthetic
moving-wrist test checks that a stationary object stays stationary in base
coordinates after compensation.

### Current workstation: SDK and ChArUco capture

#### Additional side view for collection (surveyed October 19)

The connected table-level **D405 `218722271409`** sees the tape from the side.
A camera-only survey obtained a unique mask with 100% valid depth within that
mask, median range 0.38285 m and depth MAD 0.00495 m. This is one stationary
snapshot, not a measured accuracy improvement. D405 `218622274583` provides a
higher view and did not yield a unique tape detection in its snapshot. The
right wrist remains `218622278247`; the front D455 remains `234222300865`.
Images, aligned arrays, metadata and the survey report are preserved under
`scratch/real_robot/side-camera-survey-20261019-01`. These sequential snapshots
are explicitly not synchronized pairs. No robot settings or goals were changed
by the survey.

The October 20 side-only test (`side-test-20261020-01`) accepted tape depth in
1339/1339 frames at approximately 15 FPS. Stationary XYZ standard deviations
were 0.65, 1.02 and 1.05 mm, at median range 0.383 m. These measure stationary
repeatability, not absolute accuracy or moving-target performance.

`multiview_calibrate` captures nine distinct, steady ChArUco board poses in both
fixed cameras. A joint ray fit uses the first six poses, reserving three for
held-out relative-transform and reprojection checks. Board corners are converted
to rays with the native RealSense deprojection model. The board is 6×6 squares,
50 mm per square, with inverted DICT_4X4_50 markers; the inferred 37.5 mm marker
size remains provisional. Keep camera mounts and the right base fixed:

```bash
DISPLAY=:1 PYTHONPATH=. .venv/bin/python -m real_robot.multiview_calibrate \
  --output scratch/real_robot/front-side-calibration-next --auto --poses 9 \
  --seconds 900 \
  --base-candidate scratch/real_robot/right-handeye/session-20261019-03/candidate-calibration.json
```

Only supply `--base-candidate` after confirming that the front camera and right
base have not moved since that reference. Move the board 5–10 cm or tilt it
10–20 degrees after each save, then hold still. Both panels need at least 12
detected corners. Q ends capture; `--fit-only` can refit its saved pose files.
`rig.json` separately records relative validation and the still-provisional
base reference. Its convention is `T_front_side`, mapping side optical XYZ into
front optical XYZ; `T_base_side = T_base_front @ T_front_side`. Passing relative
checks does not validate absolute base XYZ or a physical robot contact point.

The October 20 capture `front-side-calibration-20261020-01` contains nine saved
poses. Static background matching identified a side-camera rotation between
poses 2 and 3: median side-image shift was 17.28 px, with zero median shift in
the front image. The full-set fit failed and is preserved as
`rig-all-poses-rejected.json`. The documented stable segment uses poses 3–7 for
fitting and 8–9 for checking: fit per-axis residual 0.225 px; held-out side ray
RMSE 0.478 and 1.369 px; relative translation differences 2.36 and 3.71 mm.
Its `rig.json` passes the relative checks while retaining a provisional base.
Capture exclusions require explicit zero-based `--exclude-poses` and an
`--exclusion-reason`; they are never selected automatically to improve a fit.

The separate camera-only fusion preview requires a rig that passed its held-out
checks:

```bash
DISPLAY=:1 PYTHONPATH=. .venv/bin/python -m real_robot.multiview_track \
  --rig scratch/real_robot/front-side-calibration-next/rig.json \
  --output scratch/real_robot/front-side-fusion-next --side-hsv-low 90 60 30 \
  --seconds 90 --preview
```

No crop is applied unless the operator supplies `--front-roi X Y W H` and/or
`--side-roi X Y W H`. Each camera has independent HSV settings. The side depth
range is 7–50 cm. Accepted pairs span at most 25 ms across all four RGB/depth
timestamps by default. Each sample is consumed once. The preview rejects
cross-view proxy disagreement above 6 cm, explicitly labels single-view
fallback, and removes displayed/published coordinates after 0.5 seconds without
a fresh pair. The side-specific minimum brightness of 30 captures the dark
ring body; the default 45 selected mainly its lit upper rim in this setup.
Blue regions are associated using a unique cross-view 3D match, rather than
selecting the largest blue region independently in each image. Multiple valid
matches remain ambiguous. After acquisition, a single view can continue only
within 60 mm plus 0.5 m/s times the elapsed capture time of the previous accepted
position; after a 0.5 s observation gap, both views must confirm reacquisition.
This rejects a far-away blue background feature during tape occlusion without
changing the operator's image crop. It records per-view qualities, timestamps, raw measurements,
fusion and filter outputs; sampled paired RGB-D arrays are saved by default.
The causal velocity filter resets when the contributing camera set changes.
These modules do not open motor ports or send robot commands.
Live SDK, resolution, distortion and principal point must match the calibration
profile. Small focal variations up to 0.5% are permitted using native per-frame
rays and recorded in every observation; this bounds the lateral effect at the
configured front depth limit below the heuristic 15 mm uncertainty floor.

#### EfficientTAM visual tracking

The HSV movement trials still lost the narrow dark tape and sometimes selected
blue background regions. The optional `--mask-tracker efficienttam` path uses
the [official EfficientTAM](https://github.com/yformer/EfficientTAM) Tiny 512 model,
at source revision `abcd061ebd3cc6e7527d152d75b890126aaa53f6`. One GPU model serves
separate front/side memory states. Select a tight object box once in each view;
these are prompts, not image crops. Subsequent masks use learned temporal
propagation without HSV thresholds. No future frames are read by the adapter.
It retains the original prompt plus 32 recent non-conditioning outputs and one
current image per view, preventing history growth during long captures.

The optional inference installation in the workspace `.venv` reuses its existing
PyTorch 2.6.0/CUDA 12.4 and torchvision 0.21.0; it does not modify the LeRobot
environment. Reproduce with the pinned official source and minimal inference
dependencies (the upstream demo-app requirements are not needed):

```bash
git clone https://github.com/yformer/EfficientTAM.git third_party/EfficientTAM
git -C third_party/EfficientTAM checkout abcd061ebd3cc6e7527d152d75b890126aaa53f6
.venv/bin/python -m pip install -r real_robot/requirements-efficienttam.txt
Efficient_Track_Anything_BUILD_CUDA=0 .venv/bin/python -m pip install --no-deps --no-build-isolation -e third_party/EfficientTAM
curl -L --fail -o third_party/EfficientTAM/checkpoints/efficienttam_ti_512x512.pt https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/efficienttam_ti_512x512.pt
DISPLAY=:1 PYTHONPATH=. .venv/bin/python -m real_robot.multiview_track \
  --rig scratch/real_robot/front-side-calibration-20261020-01/rig.json \
  --output scratch/real_robot/efficienttam-next --mask-tracker efficienttam \
  --seconds 300 --preview
```

Draw a rectangle around the tape in the front selection window and press Enter,
then repeat in the side window, keeping the object still during both prompts.
Explicit `--front-seed-box X Y W H` / `--side-seed-box X Y W H` support repeatable
offline tests. Model boxes and seed images are saved, with checkpoint SHA-256,
configuration and native camera timestamps. Model object-presence scores are
logged; these are not calibrated probabilities of correct target identity.
Low object score or tiny masks produce no depth observation. Green model masks
can remain visible while a rejected depth/cross-view estimate produces no XYZ.

If one view loses the model mask or valid depth while the other retains a strong
object score and usable depth, recovery projects that observation into the lost
view using the rig and native lens deprojection. It proposes a new box prompt at
most once per second, scaled from the initial object box. The proposal must have
at least 30% blue pixels and agree with the other view's 3D surface proxy within
60 mm. This appearance check only validates recovery; it does not threshold or
propagate the model mask. Rejected trials leave the original memory intact.
Recovery preserves the original manual prompt plus one latest recovery prompt
and bounded recent history. It cannot recover from both cameras being occluded.

For a restart with the same target and unchanged camera mounts,
`--seed-session scratch/real_robot/efficienttam-live-20261020-01` reuses the saved
manual target images/boxes and records their source, without selecting new boxes.
Keep the tape visible in both views initially so metric association can initialize.

A 20-pair sampled replay on this workstation's RTX 4090 produced median two-view
model time 26.6 ms (95th percentile 31.0 ms), with the tube hole preserved in the
inspected masks. This is a saved-frame smoke test, not live latency or accuracy
validation. Evidence is under `scratch/real_robot/efficienttam-offline-20261020-01`.
The metric fusion, timestamp limits and provisional-base status still apply.

The two-minute stationary live test `efficienttam-live-20261020-02` fused all
1258 accepted pairs, with 38.1 ms processing at the 95th percentile. Median view
disagreement was 19.6 mm; stationary XYZ standard deviations were 0.22, 0.30 and
0.37 mm. Pair span remained below 25 ms. These are repeatability/consistency
measurements, not absolute accuracy. The preceding movement test exposed the
occlusion recovery issue described above. A real-model saved-RGB-D recovery
probe, with one observation deliberately withheld, accepted the projected box
at 19.1 mm cross-view disagreement and 77% blue mask pixels. Its evidence is under
`scratch/real_robot/efficienttam-recovery-probe-20261020-01`.

The operator then performed a physical side-camera hand-cover test in
`efficienttam-live-20261020-03`. The front model retained the stationary tape
throughout; side absence spanned frames 1048–1065 and front-only tracking
continued for those 18 accepted pairs. Two-view fusion resumed at frame 1066.
The missing interval was 1.867 seconds including the physical cover; it is not
a latency measurement from hand removal. Model reacquisition handled this
first cover, while geometry-guided proposals during the cover were rejected.
Later movement produced accepted side recovery proposals at frames 1803, 2021
and 2086, with respective cross-view differences of 19.2, 39.3 and 18.0 mm.
Wider movement still produced periods without usable shared visibility/depth;
this is not a demonstration of uninterrupted dynamic manipulation. See
`occlusion-verification.json` and `occlusion-verification.jpg` in that session
for the source indices, hand-cover images and recovery checks. No motor goals
were sent in any of these tests.

For collection, use native capture times and measured cross-view skew; common
`global_time` timestamps do not establish synchronized exposures. Reject pairs
whose skew is too large for the allowed motion error, and evaluate moving-object
latency and dropouts before enabling fused control. Retain per-view observations,
quality/occlusion flags, source timestamps, frame/calibration identifiers and
fused position/velocity/covariance. Any online estimate must use only samples
that have already arrived. The fusion path is a separate CLI from the existing
single-camera tracker and has no robot-control backend.

Front and side visible-surface proxies can refer to different physical points.
For a shared geometric centroid, define a consistent object center, using a
multi-view shape fit for the tape roll or corresponding features where valid;
evaluate systematic view bias before claiming centroid accuracy. The current
fusion is explicitly a visible-surface position proxy, with a heuristic 15 mm
uncertainty floor plus view spread, excluding absolute base uncertainty. The side camera
can support the collection expert separately from the policy observation design.
If KR-VLA consumes fused velocity at inference, its deployment observation
source and latency must also be specified and matched during training.

On the current `/home/ailab` workstation, use the repository `.venv` for
perception. The historical `/home/ailab-students` paths below belong to an earlier
machine. Install `real_robot/requirements.txt` in the isolated environment.
The LeRobot environment's RealSense SDK 2.55 reports a different D405 color focal
length at 640×480 than SDK 2.59 (435.62 versus 393.08 pixels on camera
`218622278247`); the recorded ChArUco images independently favor the latter.
Do not fit the current wrist-camera recordings using the older SDK intrinsics.
Camera metadata now includes SDK version; raw historical recordings are retained.

`aloha_charuco_capture` owns the right leader/follower serial ports and wrist
camera in one process. Do not run a second teleoperation process beside it.
It requires SDK >=2.59 plus the existing LeRobot dependencies; the current local
`.venv` can read dependencies from the LeRobot environment without modifying it.

```bash
PYTHONPATH=/home/ailab/lerobot:. .venv/bin/python -m real_robot.aloha_charuco_capture \
  --output scratch/real_robot/right-handeye/NEW-SESSION
```

The window starts paused. `T` toggles leader following using a separate 30 Hz
loop and the usual five-degree relative-target limit. `S` saves a stationary
observation without pausing following; `Q` closes the collector. Initial leader
and follower poses should be close, as for normal teleoperation. The collector
does not change torque settings and requires follower torque already enabled
to follow. Exit retains torque settings and the last commanded position.
It rejects saves with fewer than 12 ChArUco corners or recent arm motion.
Raw joint ticks, software calibration files/hashes, RGB-D, detected corners,
timestamps and measured control rate are preserved. Saved observations do not
establish validated kinematics or camera-to-base transforms. The marker length
defaults to a provisional 37.5 mm and must be physically checked before fitting.

The first current-table dataset has 15 poses in
`scratch/real_robot/right-handeye/session-20261019-03`. Its unvalidated candidate
transforms and diagnostics are saved separately from original pose metadata.
Correcting the wrist RGB intrinsics reduces reprojection errors, but reserved
pose consistency still has a worst position residual of about 21 mm. Physical
grid measurements and independent base-frame checks remain required. Depth
aligned by the older SDK is excluded from the camera-to-base fit. Do not feed
the candidate transforms to a robot controller.

### Supervised right-ALOHA hover verification

`aloha_hover_check` is a separate actuator adapter for an explicitly supervised
calibration test. It does not override `ShadowHoverController`'s calibration
checks or make the candidate transform validated. The current executor is
restricted to the reviewed R3C3 hover with a provisional 160 mm gripper-frame
fingertip offset, as measured by the operator. It requires an offline sampled
path, working camera preview, matching initial feedback, correct motor models,
torque already enabled, and verified paired-motor secondary IDs. It writes only
six right-arm master-joint goal positions, leaving gripper, torque, modes, and
gains unchanged. Other arm fixtures and the mobile base are not actuated.

The sampled move limits commanded joint speed to 3 degrees/s, checks feedback
and motor faults, and supports Q to stop generating setpoints. A nearby fixed
hover can use bounded outer feedback (`--settle-only`): at most three degrees of
goal correction, two degrees/s slew, and 15 seconds. This compensates load-related
position error without changing motor gains. Settling requires less than 0.7
degrees error at every joint. On exit, retain the last bounded motor goal and
torque settings; replacing a compensated goal with the measured position would
remove its holding effort. The arm remains powered and holding after exit.

The first test's initial settling failure and correction are retained in
`scratch/real_robot/right-handeye/hover-first-*`. The final R3C3 check reached
0.61 degrees maximum joint error, with a model-estimated fingertip standoff of
73.8 mm against an 80 mm target. This is not an independent measurement of
absolute accuracy. The operator accepted R3C3 X-Y alignment as negligible for
current purposes and reported about 85 mm physical height. This single
approximate observation is recorded separately from model estimates and does
not justify a global vertical correction.

`aloha_grid_check` extends the supervised check to a reviewed 36-square route.
The route starts at the accepted R3C3 pose and visits each remaining square
through neighboring centers. Offline checks include bounded IK, table
clearance, and modeled self contacts. The control loop retains bounded load
compensation across transitions, caps nominal path speed at two degrees/s and
total command speed at three degrees/s, checks joint following error and model
standoff, and stops on motor faults or a stale preview. Each target must settle
before an inspection dwell. P pauses progression while retaining the hold
controller; Q stops new setpoints, leaving the last bounded goal and torque
settings in place. Images and joint/goal logs are saved per square. These
records do not constitute quantitative absolute calibration or autonomous
dynamic-object manipulation.

**Current hardware incident:** the first grid sweep reached 12 centers including
R3C3, then stopped near R1C5 with overload shutdown (hardware error 0x20) on both
shoulder motors, IDs 2 and 3. Their torque was automatically disabled; reported
temperatures were 42/41°C and voltage about 12.2 V. The root cause has not been
established. Logs, completed-point images and the read-only fault audit are in
`scratch/real_robot/right-handeye/grid-check-01/`. The existing hover/grid plan
files are marked execution-blocked pending mechanical support and recovery/load
assessment. Do not rerun the same sweep or clear/re-enable the shoulder motors
without addressing this incident. Software ceasing setpoints cannot preserve
active holding when actuator protection disables torque.

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
