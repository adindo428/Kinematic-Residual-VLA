# Upright blue-tape pick-and-place

Use the existing `.venv` from `/home/ailab/Desktop/Kinematic-Residual-VLA`.
Prepare the arm with the leader as shown in [Tracking](README_TRACKING.md#1-prepare-the-arm-with-the-leader),
then stop teleoperation. Start with an empty gripper and TCP at least **100 mm**
above the table. Keep the tape upright on its curved edge and the box fixed.

## Run pick-and-place — one command

Stop separate perception, collection and `tracking_session follow` processes
first so this command can own the cameras and follower.

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.automatic_pick_place run \
  --template scratch/real_robot/pick-place/blue-tape-auto-template.json \
  --box-registration scratch/real_robot/pick-place/box2-rim-registration.json \
  --speed-scale 1.5 --execute
```

The command starts EfficientTAM perception and the wrist camera, reuses saved
front/side tape masks, and stops its perception processes when the attempt ends.
The shared mask library is `scratch/real_robot/object_profiles/blue_tape`, the same
layout used by data collection. If no saved prompts exist, select the tape once
in each fixed view; the prompts are saved automatically. Existing teaching
reference prompts are imported when available. Add **`--new-mask`** to redraw
and replace the front/side prompts. Raw camera frames are recorded during the
managed session, including startup, so finish/quit when done.

Wait for the wrist preview, then press **SPACE** once to start pursuit and pick.
The tape may already be moving, or you can start rolling it during approach.
**Q** stops new
goals and retains the current hold. Omit `--execute` for preview only.
A stopped attempt does not automatically release, retreat or retry.
If the wrist mask selects the wrong object, press **R**, keep the tape still,
draw a box around it and press **Enter**. Goals hold during selection and resume
after a fresh mask arrives. Green shows the accepted mask; the cyan cross is
the fixed-camera target projected into the wrist view.
With the gripper open, press **G**, click the left then right inner finger tip,
and press **Enter**. Do this once before the first pick. The fixed pixel reference
is saved in `scratch/real_robot/pick-place/wrist-gripper-reference.json` and reused
across runs. Repeat G only if the wrist camera mount or image framing changes.
The magenta line shows the saved open-jaw reference. Finger opening moves
horizontally; alignment uses the fixed horizontal midpoint so background shadows
cannot move the grasp center. Alignment centers the tape across this reference; descent waits for centering while tracking
continues. Clipped wrist masks do not supply new object motion, and pickup height
stays anchored to the initial tabletop observation.

Approach and wrist alignment run together; descent and closure keep following
the tape. Lift and transport follow one curved path, with holding verification
running alongside motion. Then release/verify, retreat and return.
Placement uses **five distinct front-camera frames after opening**: the tracked
tape must be inside the fixed box footprint and below its rim. The arm begins
its upward retreat while this check runs; wrist/side visibility is not required
after release. Managed startup publishes this feedback automatically.

| Current setting | Value |
|---|---|
| Travel/reset at `--speed-scale 1.5` | `18°/s` joints, `120 mm/s` TCP |
| Precision baseline/contact/first 50 mm of lift | `4.5°/s` joints, `30 mm/s` TCP |
| Final box approach | `12°/s` joints, `60 mm/s` TCP |
| Grasp orientation during approach/descent | `1×` rotation rates; wrist roll `9°/s` |
| Moving-target pursuit | Target velocity feedforward plus relative contact motion |
| Current box | Right-side `box_2`, measured `90 × 165 mm`, rim `65 mm` |
| Release TCP height | Approximately `185 mm` above the tabletop |

Following and manipulation share one pose/velocity servo. Edit `manipulation:`
in [`config.yaml`](config.yaml) for phase rates and gains. The default speed
scale is **1.5**; `--speed-scale 1` uses the baseline rates in that section.
`approach_rotation_scale: 1.0` uses baseline orientation rates before closure;
translation, closure and transport rates keep their existing settings.
Motor velocity/profile readbacks also apply. The precision rate bounds motion
relative to the tape; pursuit uses the tracking speed setting and motor limits.
Wrist depth is registered to the fixed-camera proxy while both are visible;
the demonstration supplies an initial offset if overlap is unavailable.
Capture-time encoder poses compensate camera motion during occlusion.
If both target estimates disappear, goals hold until fresh feedback returns.
The gripper keeps its grasp orientation relative to
the arm's radial heading during transport, avoiding the taught drop wrist twist.
Release requires 35 mm interior clearance, height within 10 mm of the target,
and filtered lateral speed below 15 mm/s for 0.2 s; it keeps converging until ready.
There are no total wrist-angle/translation budgets, stationary-start checks,
or software torque/current abort caps. Table/robot/box geometry checks remain.
The moving-target implementation has not yet been validated on the arm.
The camera/base, wrist and fingertip transforms remain provisional.

## External perception / teaching

For teaching or box registration, start camera-only perception separately:

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.tracking_session perception \
  --config real_robot/config.yaml --record-rgbd
```

To use that process for picking, add `--external-perception` to the run command.
Optionally supply `--observation /path/to/observation.json`. The controller leaves
external perception running when it exits; `--new-mask` uses managed startup.

### Teach additional grasps

Keep this external perception running with `--record-rgbd`. Match the leader to the follower,
then start capture with a **new output directory**:

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.grasp_capture \
  --config real_robot/config.yaml --heading-deg 180 --execute \
  --output scratch/real_robot/grasps/blue-tape-box2-03
```

Heading `180` declares rolling along base -X, with the tape axle along base Y.
Focus the wrist window for keyboard controls. Use the leader for all movement
and gripper actions; pose-marker keys only record what you demonstrated.

| Key | Action |
|---|---|
| **T** | Toggle leader following |
| **1 / 2** | Select `box_1` / `box_2` before the episode |
| **N** | Begin with tape and arms still, tape visible in both fixed views |
| **P** | Mark open pregrasp |
| **G** | Mark aligned pose immediately before closure |
| **C** | Mark closed contact |
| **L** | Mark lifted tape |
| **A** | Mark waypoint above the box |
| **R** | Mark open-gripper release after dropping the tape |
| **V** | Confirm successful placement; required to save a successful episode |
| **X / Q** | Mark failure / quit |

Use **P → G → C → L → A → R → V** in order. Hold still before each marker.
Keep both the grasp directory and its referenced front/side perception directory.

## Build assets for new demonstrations or a new box

Existing assets above can be reused while the scene stays unchanged.
Use new output names when rebuilding:

```bash
.venv/bin/python -m real_robot.automatic_pick_place prepare \
  --session scratch/real_robot/grasps/blue-tape-box2-02 --episode 0 --box-id box_2 \
  --output scratch/real_robot/pick-place/blue-tape-auto-template-new.json
```

Register a fixed box with perception running; select its visible interior floor:

```bash
DISPLAY=:1 .venv/bin/python -m real_robot.automatic_pick_place register-box \
  --template scratch/real_robot/pick-place/blue-tape-auto-template-new.json \
  --rim-z-base-m 0.065 \
  --output scratch/real_robot/pick-place/box2-registration-new.json
```

Use the new template/registration paths in `run`. If the floor is too occluded,
`register-box --help` lists `--rim-corners` (four inner-rim image corners) and
`--box-size-mm`; measure/select these for the actual scene.

## Common issues

- **Camera busy:** stop the other perception/collection process before managed
  startup. For external perception, use `--record-rgbd` and `--external-perception`.
- **Blank wrist:** wait for its preview before SPACE. A camera detected over
  USB but delivering no frames may need its wrist USB connection restarted.
- **Not saved before N:** hold the leader, follower and tape still; check valid
  depth/fusion in both fixed views, not just visible mask outlines.
- **Stopped attempt:** inspect the printed reason, reset manually to the empty
  held starting pose, then start a new run.

Managed sessions are under `scratch/real_robot/pick-place/sessions/<timestamp>/`:
`perception/`, `perception.log`, and `pick/`. External-perception runs use
`scratch/real_robot/picks/automatic-*`. Demonstrations,
calibrations, templates and box registrations remain local; GitHub contains code.
