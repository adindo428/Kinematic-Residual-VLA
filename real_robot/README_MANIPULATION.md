# Upright blue-tape pick-and-place

Use the existing `.venv` from `/home/ailab/Desktop/Kinematic-Residual-VLA`.
Prepare the arm with the leader as shown in [Tracking](README_TRACKING.md#1-prepare-the-arm-with-the-leader),
then stop teleoperation. Start with an empty gripper and TCP at least **100 mm**
above the table. Keep the tape stationary on its curved edge and the box fixed.

## 1. Start perception — terminal 1

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.tracking_session perception \
  --config real_robot/config.yaml --record-rgbd
```

Select the tape in both fixed views and wait for stable fusion. Leave this
terminal running. Continuous RGB-D recording is required for manipulation;
stop perception when finished because it uses substantial disk space.

## 2. Run one pick-and-place — terminal 2

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.automatic_pick_place run \
  --template scratch/real_robot/pick-place/blue-tape-auto-template.json \
  --box-registration scratch/real_robot/pick-place/box2-rim-registration.json \
  --speed-scale 1.5 --execute
```

Wait for the wrist preview, then press **SPACE** once to start. **Q** stops new
goals and retains the current hold. Omit `--execute` for preview only.
A stopped attempt does not automatically release, retreat or retry.

Sequence: approach → wrist alignment → descend → grasp → lift/verify →
above box → release/verify → retreat → return to the starting pose.

| Current setting | Value |
|---|---|
| Travel/reset at `--speed-scale 1.5` | `18°/s` joints, `120 mm/s` TCP |
| Alignment/contact/first 50 mm of lift | `4.5°/s` joints, `30 mm/s` TCP |
| Final box approach | `12°/s` joints, `60 mm/s` TCP |
| Wrist roll during alignment | `9°/s`; corrections up to `3°` |
| Pregrasp handoff | `5 mm / 3°` plus reliable wrist tracking |
| Current box | Right-side `box_2`, measured `90 × 165 mm`, rim `65 mm` |
| Release TCP height | Approximately `185 mm` above the tabletop |

Following and manipulation share one pose/velocity servo. Edit `manipulation:`
in [`config.yaml`](config.yaml) for phase rates and gains. The default speed
scale is **1.5**; `--speed-scale 1` uses the baseline rates in that section.
Motor velocity/profile readbacks also apply. Fresh wrist tracking carries alignment and descent
through fixed-camera occlusion after pregrasp. Optional expired fixed estimates
do not stop lift/transport. The gripper keeps its grasp orientation relative to
the arm's radial heading during transport, avoiding the taught drop wrist twist.
Release requires 35 mm interior clearance, height within 10 mm of the target,
and filtered lateral speed below 15 mm/s for 0.2 s; it keeps converging until ready.
There are no software torque/current abort caps. This entry point remains for
stationary tape; moving-object grasping is the next integration step.
The camera/base, wrist and fingertip transforms remain provisional.

## Teach additional grasps

Keep perception running with `--record-rgbd`. Match the leader to the follower,
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

- **Requires perception `--record-rgbd`:** stop perception and restart with the
  flag from step 1; leave it running before starting manipulation.
- **Blank wrist:** wait for its preview before SPACE. A camera detected over
  USB but delivering no frames may need its wrist USB connection restarted.
- **Not saved before N:** hold the leader, follower and tape still; check valid
  depth/fusion in both fixed views, not just visible mask outlines.
- **Stopped attempt:** inspect the printed reason, reset manually to the empty
  held starting pose, then start a new run.

Run logs are under `scratch/real_robot/picks/automatic-*`. Demonstrations,
calibrations, templates and box registrations remain local; GitHub contains code.
