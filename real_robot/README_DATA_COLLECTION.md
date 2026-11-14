# Five-task right-arm real data protocol

Protocol `right-arm-dom-real-v1`, revised 2026-10-22. This is the collection design
and schedule. Use the [DOM collection runbook](README_DOM_COLLECTION.md) for the
separate dynamic recorder and DOM v2.1 exporter now implemented.
The current recommended command is **`.venv/bin/python -m real_robot.collect`**:
it reuses saved masks, starts both processes and clutches leader motion without
manual five-degree matching. The runbook supersedes earlier manual startup steps.
Use the right leader/follower, fixed front/side cameras and wrist RGB-D camera.
Keep the current EfficientTAM perception pipeline and model-based controller.

## Recommended experiment

Collect **1,000 accepted successful demonstrations: 200 per object task**.
Split each task into **160 train / 20 validation / 20 offline test**, making
800 / 100 / 100 overall. Separately run **300 autonomous trials per model**
(five tasks × three motion conditions × 20 trials). For a baseline/KR pair this
is 600 evaluation trials. Keep every collection failure and every rollout;
success-only filtering applies only to the imitation-learning demonstration set.

First collect ten pilot attempts per object (50 total), outside the dataset.
Use pilots to validate graspability, tracking, timing and reachable motion.
Then freeze the protocol. Start with 500 accepted demonstrations as an interim
milestone, using only the preassigned training sessions; finish the target before
opening the final test. Reserve time for roughly 1,250–1,700 collection attempts
if acceptance is 60–80%; this is a planning estimate, not a measured yield.
At 30–60 seconds per attempt/reset that is roughly 10–28 hours, plus calibration,
pilot and review time. Manual collection should be spread across sessions.

### Why this amount

The latest `origin/main` campaign at `098e01e` uses 20,000 simulation training
episodes: 9,000 pick, 9,000 place and 2,000 long horizon, with 2,406,773 training
frames, 2,015 validation episodes and 2,011 test episodes. See
`experiments/main8_20k_10epochs/{README.md,subset-report.json}` on **main**;
those files are not necessarily present on the active experiments branch.
These runs continue previously trained checkpoints; local KRV100 has separate
full-data provenance. Do not describe every model as trained exclusively on 20k.

The real target is 5% of that simulation episode count, with 4% used for training.
It is a practical recommendation for five narrow tasks on a new embodiment,
not a statistical consequence of a simulation/real ratio. DynamicVLA reports
200K synthetic and 2K real episodes across a broader object collection; scaling
its approximately 1% ratio to 20k would give only 200 real episodes total,
which leaves little coverage per task after holding out sessions. Its paper
does not establish a universal ratio. Use validation and an extra dedicated
development rollout set to decide whether more training data is useful; extend
training in batches of 50/task while leaving the frozen test untouched.

## Five tasks and physical setup

All five tasks are **intercept, grasp, lift and place into a registered box**.
They vary object geometry/material/scale; they do not constitute five different
manipulation skills or a complete DOM benchmark.

| Task | Object and starting pose | Purpose |
|---|---|---|
| 1 | Blue tape upright on its curved edge | Existing setup; rolling ring and axle alignment |
| 2 | Green tennis ball, rolled along table | Textured sphere and compliant contact |
| 3 | Small orange, rolled along table | Irregular near-sphere and food appearance |
| 4 | White 40 mm ping pong ball, rolled along table | Small light target; depth and grasp difficulty |
| 5 | Small opaque cylindrical container on its side | Cylinder-axis alignment and a DOM-like container |

Choose a light closed container whose diameter fits the measured gripper opening;
test the actual tennis ball and orange too. Use a matte contrasting table surface.
If an object cannot be grasped reliably while static in the pilot, resolve that
before collecting dynamic demonstrations; log any object substitution and freeze
a revised protocol. Do not carry tape-specific grasps, geometry or box clearances
over to balls or containers. Teach and validate a separate grasp prior per object
for model-based collection. Use one box with enough clearance for all five objects;
register it again if moved. Keep the scene fixed initially, then vary launch
location, direction and ordinary lighting between dedicated sessions.

The tennis ball and container match object categories explicitly used in DOM's
real tasks. The tape, orange and ping pong ball are our chosen analogues; this
does not assert exact mesh/object identity with the released DOM assets.

## Collection paradigms

**A. Human demonstrations, primary dataset.** A person launches the object,
and the right leader commands the right follower. Perception records alongside
teleoperation; it does not also command the arm. Use a helper to launch while the
operator holds the leader. For solo work, use a repeatable passive release ramp
so the operator is ready before release. Record the launch method. Do not roll
with one hand and rush back to an unmatched leader. Keep hands clear before
interception. Begin at slow speeds and increase only after the pilot passes.

**B. Model-based demonstrations, supplementary.** Teach object-specific static
grasp/placement priors with the leader, then have the existing model-based state
machine pursue, grasp/lift, place and reset using EfficientTAM-derived feedback.
Manually launch each object. This is closer to DynamicVLA's automatic collection.
The current automatic path supports upright tape and still needs physical moving
validation; extension to the other objects is implementation work. Tag every
episode `collector=model_based` or `human_leader`; do not silently mix sources.
Keep the 1,000-example primary study human-led. Report any supplementary automatic
data, its split and count separately; give compared models identical data.

**C. Autonomous policy evaluation.** Freeze weights and normalizers, launch the
object, and let each policy run without leader assistance. Any takeover after
motion/response starts is a failed rollout. Save the takeover as a separate
correction episode if useful, without adding final-test data to training.

DynamicVLA uses automated collection because human reaction becomes limiting at
high object speeds. Our slow-to-moderate manual procedure adapts its perception,
temporal recording and phase structure; it does not reproduce its automatic
collection or its reported maximum simulation speeds.

## Demonstration balance and sessions

Measure speed from causal tracked **horizontal base-frame velocity**, using a
fixed 200 ms window immediately after the launch hand clears and before first
robot response/contact. Save the samples, window and estimator quality. Static
controls use the same window after the start cue. If insufficient valid samples
exist, mark speed unclassified and retain the raw attempt outside the quota.
Bins are lower-inclusive and upper-exclusive; 0.12 m/s is outside demonstration
coverage. Intended launch strength is not a measured speed label.

| Speed | Range (m/s) | Train/task | Validation/task | Offline test/task |
|---|---:|---:|---:|---:|
| Static control | [0, 0.01) | 16 | 2 | 2 |
| Slow | [0.01, 0.04) | 64 | 8 | 8 |
| Medium | [0.04, 0.08) | 64 | 8 | 8 |
| Brisk manual | [0.08, 0.12) | 16 | 2 | 2 |
| Total | | 160 | 20 | 20 |

Each task has ten dedicated 20-accepted-example session blocks: eight train,
one validation, one offline test. Session IDs and split assignments are fixed
before recording. Alternate feasible launch directions; vary start position
within the calibrated overlap. Create a fresh recording for every block, retaining
all retries with unique attempt IDs. A target slot is filled only by an accepted
example of the specified speed. A wrong-speed attempt can fill another unfilled
slot in the **same session/split**; otherwise retain it outside the quota. Never
move an attempt across splits to achieve balance. A shared continuous recording,
repeated launch sequence or duplicate clip must stay in one split.

Generate the deterministic schedule, without accessing robot hardware:

```bash
.venv/bin/python -m real_robot.collection_schedule \
  --output scratch/real_robot/data-collection-v1/targets.csv
```

The output refuses overwrites. `collection_plan.json` is the machine-readable
source. CSV rows are accepted-example targets, not fabricated recordings.
Maintain a separate append-only attempt ledger mapping `attempt_id` to
`target_id`, actual speed, raw paths, outcome and rejection reasons. Reviewers
must be able to account for every attempt, including incomplete ones.

## One episode

1. Start the one-command collector. It starts front/side perception and wrist
   recording, reuses the object profile and anchors leader motion at the current
   follower pose on N. Check three live views and target identity.
   Use one follower controller. Follow the existing tracking runbook for torque
   handling; collection must not unexpectedly reconnect or disable torque.
2. Start a new episode **before launch**, saving at least one second of pre-roll.
   Mark `episode_start`, `launch`, `hand_clear` and `first_robot_response`.
   For a new object, save its EfficientTAM reference/masks once while visible;
   later runs reuse them and may start before the object enters view. Do not
   reprompt on future frames to improve past observations.
3. Roll the object in the shared visible/reachable region, then intercept with
   the leader. Record uninterrupted motion through approach, contact, closure,
   lift, transport, open release and placement confirmation. Do not stop at pose
   markers. Save phase events with timestamps; annotate retrospectively if keys
   would interrupt control.
4. Confirm lift clear of the table and object released wholly inside the box,
   remaining there for two seconds. Save one second of post-roll, outcome and
   any failure reason. Reset motion is recorded with a separate phase and kept
   outside the imitation segment by default.
5. Review video and motion. A successful intercept must show at least 20 mm of
   object travel between first response and contact, with valid tracking there.
   Static controls are explicitly exempt. A roll that stops before approach is
   not accepted as a dynamic success. Retain it with the proper label.

## State estimation: what is actually measured

EfficientTAM supplies 2D masks. Our RGB-D pipeline deprojects masked depth and
fuses views, then causally estimates translation and linear velocity. In
`perception.py` this point is explicitly a **visible-surface proxy**, not an
object center. View switching, occlusion and changing visible surfaces can
produce apparent velocity. Save per-view points, fusion sources, innovations,
uncertainties and validity, not only a smoothed final trajectory.

For spheres/near-spheres, absolute rotation is generally ambiguous; cylinder and
tape spin about a symmetry axis is also ambiguous. Save `orientation_valid=false`
and `angular_velocity_valid=false` unless a separately validated geometric or
texture-based estimator supports them. A quaternion identity can be a storage
placeholder only with an explicit validity mask and no loss on that label.
Do not treat the manually supplied tape heading/reference frame as measured 6D
pose. A six-component filter of position plus velocity is not a 6D pose estimator.

For this first study use validated **translation and linear velocity** for KR.
Fit a known sphere/cylinder model to multiview depth if center/axis are needed;
validate the offset against stationary measurements before replacing the proxy.
Full rotation/angular velocity requires an additional pose estimator.
Estimated position/velocity are accepted without independent ground-truth
validation for this study. Preserve raw depth so future improvements are possible.
Optional rolling-without-slip orientation is an assumption, not ground truth.

The simulation campaign used simulator state; real perception adds estimator
noise and latency. Report this difference and an estimated-state robustness
ablation instead of claiming identical state quality. All estimates supplied to
the policy at time t must use only measurements available by t. Save smoothed
offline trajectories separately and exclude them from real-time policy inputs.

## Recording and LeRobot contract

Preserve native 30 Hz robot feedback/commands and camera rates with timestamps.
Build a **25 FPS** training export, matching the DOM training cadence; retain
native RGB-D and calibration as sidecars. Use wrist and front RGB for policy
inputs, front+side for estimation. Map front to the checkpoint's fixed-view key
explicitly; check preprocessing/crops against the real camera geometry.

| Stream | Required content |
|---|---|
| RGB video | Front, side, wrist; original intrinsics, distortion, serial, frame IDs and decode counts |
| Depth/masks | Aligned depth, scale, per-view target masks/identity and quality |
| Robot | Measured follower joints, velocities if measured, leader joints, **issued/clipped follower command**, gripper, health, read intervals and command issue times |
| TCP | Measured FK pose and commanded FK pose, TCP definition, kinematic/calibration hashes |
| Object | Proxy position, linear velocity, estimate timestamp and availability time, covariance, view sources, validity and symmetry metadata |
| Episode | Task/instruction, object instance and dimensions, box transform, session/split, collector, phase events, speed window, outcome/reason, operator/launch method |
| Provenance | Git revision, Python/package versions, config/calibration/checkpoint hashes, clock/boot ID, raw path hashes and export version |

Canonical native orientations use quaternion **xyzw**, translations meters,
joint model angles radians, velocities m/s, all poses in the robot base frame.
Preserve raw calibrated motor units (arm degrees, gripper percent) separately.
Save camera hardware timestamps/domain and host monotonic capture/arrival times;
fit and record each device's clock mapping. Never subtract unrelated hardware
clock domains directly. Log robot sample intervals and timestamp uncertainty.

At each 40 ms training anchor select the latest causal frame/state whose
**availability time** is no later than the anchor. Initial acceptance targets:
camera age ≤80 ms, mapped front/side capture skew ≤40 ms, robot sample age ≤50 ms,
object estimate age ≤80 ms. Reject/mask anchors outside these bounds; do not
silently copy old frames or interpolate using future observations. Validate
these limits on the pilot and freeze revisions before production. The existing
150 ms stationary event-snapshot bound is insufficient for dynamic alignment.
Save measurement time and availability time so inference latency is reproducible.

Use LeRobot video features and parquet state/action rows, plus standard task and
episode metadata. The DOM loader uses LeRobot v2.1 conventions; the local leader
checkout has an older API, so pin a separate exporter environment to the training
loader's compatible version. Do not upgrade either working robot environment
in place. Modern LeRobot documentation may describe a newer on-disk format.

Keep a joint-space export for native LeRobot policies and a separate Cartesian
DOM adapter. For the latter, `observation.state` is measured TCP xyz+Euler
(six values); `action` is commanded TCP xyz+Euler+gripper (seven values), obtained
from the actual clipped command and a validated FK/TCP model. Specify the Euler
convention from the checkpoint helper and unwrap consistently. Do not relabel
next measured joint positions as issued actions. Verify the physical mapping
with replay **without motor execution** and a slow supervised pilot. Let the
existing loader perform its configured action-minus-current-state conversion;
do not delta actions twice. Preserve gripper calibration and endpoint convention.
Do not apply Franka joint statistics to the VX300.

The environment-state adapter must match the exact policy's existing key/shape
and position semantics, adding separate validity features or sidecars as needed.
Do not silently fill missing object rotation and call the result measured pose.
Select valid causal training anchors without sampling future object state into
KR inputs. Fit real-domain normalization on the 800 training episodes only;
keep frozen parent normalizers with each checkpoint and make any adapter/refit
an explicit shared choice for a baseline/KR pair. Never use held-out statistics.

Validate episode indexes, monotonic times, matching video/parquet frame counts,
units, finite values, no cross-episode windows and padded final action chunks.
Use temporal observations t−2 and t and a 20-step action horizon where required
by the DynamicVLA recipe; other architectures retain their declared recipe.
Check all raw dependencies and hashes before export. Upload to Hugging Face
later, after local validation, as a private versioned dataset with a dataset card
and frozen split manifest. Keep heavy RGB-D sidecars separately if necessary,
but publish their manifest and clearly identify which data the training uses.

## Current code limitations and implementation work

`grasp_capture.py` already records continuous robot feedback/issued commands,
wrist RGB-D, perception snapshots and sparse phase events. However, its episode
start requires a stationary tape, freezes `T_base_tape_reference`, and declares
that the tape stays stationary until closure. It is for teaching grasp priors.
The open `episode_0000.json` is an event/reference document; by itself it is not
a dynamic VLA training episode. Keep its parent session and referenced RGB-D.

The separate `dynamic_capture.py` mode now uses live object state, optional
timestamped launch/response events, task/instance metadata and an attempt ledger,
without stationary pose-marker holds. It reuses serial ownership, heartbeat,
feedback logging and torque behavior. `dom_dataset.py` provides causal alignment,
a DOM v2.1 export and quality reports. Follow the runbook's actual acceptance
limits (native 30 Hz camera capture resampled to 25 FPS), which supersede the initial
timing targets above. Estimated state is accepted; missing orientation is assumed
heading and occluded position/velocity are filled with explicit quality sidecars.
The loader does not automatically mask these fills. Pilot the physical workflow
before collecting 1,000 examples. Object-general automatic control remains separate
work; this study uses human teleoperation. Existing grasp-teaching commands remain
appropriate for static per-object priors and are unchanged.

## Autonomous comparison after post-training

Freeze the chosen real post-trained baseline and KR checkpoints using validation
only. Keep their demonstration IDs, compute budget, action space, camera inputs,
execution rate and latency/streaming mode matched; report extra pose/velocity
inputs to KR and include a baseline+velocity comparison where feasible.
For the main claim prefer a matched baseline/KR family from the 20k campaign;
local full-data KRV100 versus another family has a training-provenance confound.

Run 20 trials per task in each condition: slow [0.01,0.04), medium [0.04,0.08),
and speed-OOD [0.12,0.18) m/s, subject to a successful physical pilot. Do not
extend to unsafe/unreachable speeds to fill bins; freeze a reduced achievable
range before any model comparisons, label it honestly and revise OOD claims.
Use a fixed passive release ramp/marked release heights and starts for
evaluation if available; hand launches require measured-speed stratification.
Randomize model order in blocks and balance directions/start positions.
The paper uses a secondary robot to standardize launches; manual rolling is
less repeatable and must be described as such.

Assign launch condition before running. Keep an intent-to-test record of every
scheduled trial; speed misses are retained and reported separately by actual
speed, with predeclared replacements to fill valid-condition quotas. Never
discard a difficult launch after seeing policy failure. Sensor/controller
failures after response starts count in primary success rate, with reasons;
pre-response infrastructure failures remain in the audit and are retried.

Timeout is 30 seconds from hand-clear (static cue for static tests). Success
requires tracking the requested object, dynamic interception, lift, release
inside the box and two-second dwell without a drop/takeover. Report interception
success separately from full placement. Save success/failure video, launch speed,
tracking validity, policy/action latency, task time and TCP path length. Report
success n/N and Wilson 95% intervals by task/condition, then the equal-weight
macro-average. Twenty trials is an initial benchmark unit with wide uncertainty;
it does not guarantee significance for small baseline/KR differences. If more
precision is needed, add predeclared balanced blocks for every compared model.

## Sources

- [DynamicVLA paper](https://arxiv.org/html/2601.22153v1), IV-C/IV-D:
  multiview EfficientTAM/state estimation and automatic four-stage collection;
  V-A: 20 repetitions and standardized launches; Appendix B: real post-training.
- [Official DOM dataset](https://huggingface.co/datasets/hzxie/DOM): released
  simulation data and camera/rate metadata.
- [LeRobot real-robot guide](https://huggingface.co/docs/lerobot/il_robots):
  demonstration, training and autonomous evaluation workflow. Use local pinned
  API/version details rather than copying current guide commands onto this rig.
- Local implementation: `grasp_capture.py`, `perception.py`, `tracking.py`,
  `README_MANIPULATION.md`, `README_TRACKING.md`; DOM adapter:
  `third_party/DynamicVLA/scripts/create_lerobot_dataset.py` and `utils/datasets.py`.
