# Right-arm blue-tape tracking: workstation runbook

Run these commands on this workstation from the `experiments` checkout. Use
the existing project `.venv` for perception/control and the existing `lerobot`
Conda environment for leader teleoperation. No motion-plan JSON is required.

## Hardware and saved calibration

| Component | Identifier |
|---|---|
| Front RealSense D455 | `234222300865` |
| Fixed side RealSense D405 | `218722271409` |
| Right VX300 ALOHA follower | `/dev/ttyDXL_follower_right`, 1 Mbps |
| Front/side rig | `scratch/real_robot/front-side-calibration-20261020-01/rig.json` |
| Controller configuration | `real_robot/config.yaml` |

Keep both camera mounts and the right-arm base fixed. The relative camera
calibration passed its checks; absolute right-base calibration and the
160 mm fingertip-midpoint offset remain provisional. Moving a mount or the
base requires checking/calibrating the transforms again. Camera boxes below
select the tracked object; they do not crop the tabletop image.

The tested setup uses the two fixed cameras. Disconnect unused wrist cameras
if USB bandwidth/control errors recur. The side camera worked after connecting
it directly to a PC USB 3 port. Keep the target visible in both views initially.

## 1. Prepare the arm using the leader

Start with the follower securely resting and cool. Match the leader to its
pose and clear the movement path. Only one process may own the follower port.

```bash
cd /home/ailab/lerobot
/home/ailab/miniconda3/envs/lerobot/bin/python lerobot/scripts/control_robot.py \
  --robot.type=aloha_solo \
  --robot.cameras='{}' \
  --robot.max_relative_target=5 \
  --control.type=teleoperate \
  --control.fps=30
```

Use the leader to lift the follower into a comfortable, unfolded pose above
the table. Hold the leader still, then press **Ctrl+C** in that terminal.
This checkout's LeRobot disconnect retains follower torque and holding goals;
the tracking process takes over that held state. LeRobot's connection briefly
disables torque, so do not reconnect it while the elevated arm is unsupported.
Do not force the follower by hand while torque is on.

## 2. Start EfficientTAM and front/side fusion

In terminal 1:

```bash
cd /home/ailab/Desktop/Kinematic-Residual-VLA
DISPLAY=:1 .venv/bin/python -u -m real_robot.tracking_session perception \
  --config real_robot/config.yaml
```

Keep the tape still. Draw a tight rectangle around it in the **front** selection
window and press **Enter**. Repeat in the **side** window. Wait for the combined
preview and check that both masks follow the tape rather than the background.
The default `seed_session: null` asks for new selections on each launch.

The preview reports fused 3D position and estimated XYZ velocity/total speed
in m/s and cm/s. This is a filtered visible-surface proxy, not independently
measured object-centroid ground truth. Source changes, rolling shape and
occlusion can affect velocity. Avoid interpreting stationary noise as motion.

## 3. Start right-arm following

Keep the tape still for the initial approach and clear the entire arm sweep.
In terminal 2:

```bash
cd /home/ailab/Desktop/Kinematic-Residual-VLA
DISPLAY=:1 .venv/bin/python -u -m real_robot.tracking_session follow \
  --config real_robot/config.yaml --execute
```

The launcher selects the latest perception session from this boot. The arm
must already have healthy holding torque; this command does not enable it.
Omit `--execute` for feedback/preview without sending motor goals.

Once it reaches the object, roll the tape **toward the arm**, beginning slowly
inside the shared camera view. Keep it within the reachable workspace and away
from the base and robot links. The current controller follows measured position
using Cartesian PID and inverse kinematics. It does not predict an intercept,
close the gripper, or perform a grasp. It does not enforce the rolling direction.

The fingertip midpoint targets **5 cm above the estimated visible object top**.
Depth-mask top estimates are used when available, with a configured height
offset as fallback. The tool's fixed `tool_pitch_deg: 60` is a downward angle
chosen to reduce shoulder load. It is separate from angular velocity; there is
no object angular-velocity tracking. Grasp orientation/control can be added later.

## 4. Change speed, PID and workspace

Edit `real_robot/config.yaml` while the controller is stopped, then restart
`follow`; settings are loaded at startup, not live-reloaded.

| `servo` setting | Current value | Meaning |
|---|---:|---|
| `kp`, `ki`, `kd` | `2.0`, `0.0`, `0.0` | Cartesian feedback gains |
| `max_tcp_speed_mps` | `0.5` | Commanded fingertip speed ceiling |
| `max_joint_speed_deg_s` | `null` | Use existing motor velocity/profile limits |
| `max_tcp_acceleration_mps2` | `null` | No additional pilot acceleration cap |
| `max_command_lead_m` | `null` | No additional pilot command-lead cap |
| `target_smoothing_s` | `0.12` | Smooth fused target changes |
| `hover_height_m` | `0.05` | Standoff above estimated object top |
| `tool_pitch_deg` | `60.0` | Fixed downward tool pitch |
| `tool_pitch_rate_deg_s` | `4.0` | Rate of initial pitch alignment |
| `max_following_error_deg` | `12.0` | Stop on excessive joint tracking error |
| `experimental_current_guard` | `false` | Disable pilot software current caps |
| `max_observation_age_s` | `0.20` | Hold if observations are too old |
| `control_hz` | `30` | Requested controller rate |
| `seconds` | `600` | Session duration |

The PID uses `v = kp*position_error + ki*integral - kd*measured_TCP_velocity`.
Workspace bounds are right-base TCP coordinates in metres:
`[0.20, -0.30, 0.09]` to `[0.58, 0.30, 0.35]`. IK also checks reach and joint
limits; not every point in this box is reachable. Stale/unreachable targets
hold and resume automatically when valid. Motor faults, temperature, excessive
following error, overspeed and modeled geometry violations stop the session.

[DynamicVLA v2 Appendix D](https://arxiv.org/html/2601.22153v2) reports object
release speeds of **0.2–0.8 m/s**. These are object speeds, not a VX300 motor/PID
preset. Our arm's requested ceiling remains **0.5 m/s**; an object can outrun it.
Current tests do not validate interception or grasping at those release speeds.

The first uncapped trial briefly measured **0.63 m/s by encoder/FK**, despite
the 0.5 m/s command ceiling, and ended at the **55°C shoulder guard**. A readback
speed watchdog was added afterward: it stops new goals on a measured excursion.
It cannot guarantee an instantaneous physical speed bound. Do not treat the
0.5 m/s setting as a certified physical limit. Existing firmware current,
velocity and fault limits remain unchanged. Cool the arm before another trial.

## 5. Stop and switch off torque

Press **Q** in the arm window or **Ctrl+C** in terminal 2 to stop following.
This retains the last goals and holding torque. Then press **Q** in the camera
preview or **Ctrl+C** in terminal 1 to close perception. Ensure both have exited.

Read the current torque state without changing it:

```bash
cd /home/ailab/Desktop/Kinematic-Residual-VLA
.venv/bin/python -m real_robot.arm_power
```

To release the arm, **physically support its full weight first**, stop any
teleoperation/controller using the port, then run:

```bash
.venv/bin/python -m real_robot.arm_power --supported --execute
```

Verify that every entry in `torque_after` is `0`. This disables all nine
right-follower motors, including the gripper; it does not cut electrical power
or move the arm to a sleep pose. Leave it securely resting before switching
off its normal power supply. Automatic powered parking remains unvalidated.

## Logs, installation and checks

Runs are saved under `scratch/real_robot/tracking/perception-*` and `follow-*`.
The follow directory contains the config snapshot, `status.json`,
`motion.jsonl` and `summary.json`. Logs include fused target/object velocity,
motor currents, actual/goal joints and measured FK TCP speed. The launcher uses
`latest-perception.json` to find the camera session. These are diagnostics,
not labeled expert grasp demonstrations.

The existing `.venv` already contains the working dependencies. Its Python
paths also reference the local LeRobot environment and source tree. For the
pinned EfficientTAM installation commands and calibration procedures, see
[the main README](README.md#efficienttam-visual-tracking). Do not install perception
dependencies into the training or LeRobot environment. This runbook assumes
the saved rig, collision-model XML and mesh assets remain on this workstation.

```bash
.venv/bin/python -m unittest discover -s real_robot/tests -q
git diff --check
```

If a camera reports USB errors, stop its preview before reconnecting it while
keeping its mount fixed. If the serial port is busy, stop its existing owner;
do not run teleoperation and the follower simultaneously. If a motor faults or
overheats, stop and support the arm; do not bypass the guard or blindly reboot
and resume. Robot/table geometry checks do not detect arbitrary obstacles.
