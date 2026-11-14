# Real-robot tracking and manipulation

Commands for the right VX300 ALOHA arm and blue tape on this workstation.

For manual data collection, run **`.venv/bin/python -m real_robot.collect`**.
It starts perception, reuses your saved tape masks, clutches leader motion at the
current follower pose, records attempts and exports DOM data when finished.
No separate LeRobot preparation command or five-degree leader matching is needed.

| Task | Guide |
|---|---|
| Prepare the arm, track rolling tape, stop/shutdown | [Tracking](README_TRACKING.md) |
| Teach grasps and run moving/stationary pick-and-place | [Manipulation](README_MANIPULATION.md) |
| Plan five-task real post-training data and held-out trials | [Real data collection](README_DATA_COLLECTION.md) |
| Record leader demonstrations and export DOM v2.1 datasets | [DOM collection runbook](README_DOM_COLLECTION.md) |

Run from `/home/ailab/Desktop/Kinematic-Residual-VLA` using `.venv/bin/python`.
Leader teleoperation uses the separate LeRobot environment shown in the tracking guide.
Only one controller may own the follower serial port.

## Current setup

- Front camera: D455 `234222300865`.
- Side camera: D405 `218722271409`.
- Wrist camera: D405 `218622278247`.
- Right follower: `/dev/ttyDXL_follower_right`.
- Settings: [`config.yaml`](config.yaml).
- Calibration, demonstrations and scene-specific assets are local under
  `scratch/real_robot/`; they are excluded from Git. Keep these when moving computers.

Tracking uses fixed-camera EfficientTAM masks and RGB-D fusion. Manipulation
uses an object-relative grasp prior, wrist alignment and a model-based state
machine with concurrent alignment and pursuit, followed by curved transport.
It targets **upright tape standing or rolling on its curved edge**; the moving
path still needs physical validation.
Camera/base, wrist and TCP transforms remain provisional.

Neither active controller adds software torque/current abort thresholds.
Existing motor firmware protections and the current table/geometry, motor-health
and feedback checks remain in place. These commands preserve existing holding
torque; they do not enable it automatically.

## Code layout

- `tracking_session.py`: perception/follow launcher.
- `task_space_servo.py`: shared Cartesian feedback/feedforward and differential IK.
- `automatic_demonstrator.py`, `vx300_demonstrator.py`: task stages and wrist alignment.
- `aloha_arm.py`: shared follower serial I/O.
- `collision.py`, `pick_geometry.py`: shared geometry checks.
- `aloha_charuco_capture.py`, `multiview_calibrate.py`: calibration tools.
