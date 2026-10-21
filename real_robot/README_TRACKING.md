# Right-arm tape tracking

For grasp recording and pick-and-place, use [Manipulation](README_MANIPULATION.md).
Keep the fixed camera mounts and arm base unchanged. Use only one arm controller.

## 1. Prepare the arm with the leader

```bash
cd /home/ailab/lerobot
/home/ailab/miniconda3/envs/lerobot/bin/python lerobot/scripts/control_robot.py \
  --robot.type=aloha_solo --robot.cameras='{}' \
  --robot.max_relative_target=5 --control.type=teleoperate --control.fps=30
```

Match the leader to the supported follower before starting. Move to an unfolded
pose above the table, hold still, then press **Ctrl+C**. This workstation's
LeRobot disconnect retains the follower's holding torque. Do not reconnect
teleoperation with an unsupported elevated arm: connection briefly disables torque.

## 2. Start perception — terminal 1

```bash
cd /home/ailab/Desktop/Kinematic-Residual-VLA
DISPLAY=:1 .venv/bin/python -u -m real_robot.tracking_session perception \
  --config real_robot/config.yaml
```

Keep the tape still. Draw a tight box around it in the front view, press
**Enter**, then repeat in the side view. Wait until both views track the tape.
For grasp teaching or automatic manipulation, add **`--record-rgbd`** at startup.

## 3. Start following — terminal 2

```bash
cd /home/ailab/Desktop/Kinematic-Residual-VLA
DISPLAY=:1 .venv/bin/python -u -m real_robot.tracking_session follow \
  --config real_robot/config.yaml --execute
```

The arm must already be holding with torque on. The command selects the latest
perception session automatically. Omit `--execute` for feedback/preview only.
Once the arm reaches the stationary tape, start rolling it inside the shared view.
Following holds the fingertips above the object; it does not close the gripper.

## Settings

Edit [`config.yaml`](config.yaml) while stopped, then restart the controller.

| Setting under `servo` | Current value |
|---|---|
| `kp`, `ki`, `kd` | `2.0`, `0.0`, `0.0` |
| `max_tcp_speed_mps` | `0.5` commanded ceiling |
| `hover_height_m` | `0.05` above estimated object top |
| `tool_pitch_deg` | `60` |
| `target_smoothing_s` | `0.12` |
| `seconds` | `600` |

Workspace bounds and side-camera depth range are in the same YAML. Stale or
unreachable targets cause following to hold until valid again. The active
follower has no software current/torque caps; hardware faults, temperature,
following error, measured-speed and modeled geometry checks can stop it.
Commanded speed is not a guarantee of measured physical speed.

## Stop and torque off

**Q** in the arm window or **Ctrl+C** stops new goals and retains holding torque.
Close perception separately with **Q** or **Ctrl+C**.

Read torque state:

```bash
.venv/bin/python -m real_robot.arm_power
```

To switch torque off, first **physically support the full arm**, stop all arm
controllers, then run:

```bash
.venv/bin/python -m real_robot.arm_power --supported --execute
```

Check that every `torque_after` entry is `0`. This does not park the arm.

## Common startup issues

- **Torque unexpectedly off:** prepare the held pose using the leader first.
- **No associated tape proxy:** visible 2D masks alone are insufficient; wait
  for accepted depth/fusion. The current side depth range is `0.07–0.80 m`.
- **Blank wrist view:** see [Manipulation](README_MANIPULATION.md#common-issues).
