# Implementation contract

## Architecture and interfaces

`config.py` resolves YAML/CLI parameters, rejects invalid timing and bounds, and fingerprints experiments. `assets.py` inventories the downloaded DOM corpus, excludes benchmark assets, and validates layouts. `isaac_backend.py` assembles and runs one Franka environment using Isaac Lab. All Isaac imports occur after `AppLauncher`; numerical/export tests need no simulator.

`trajectory.py` generates bounded smooth paths and predicts mode occupancy. `controller.py` selects orientation and calculates joint targets. `collect.py` coordinates warm-up, state/command timing, camera capture, quality gates, and episode commits. `storage.py` handles raw logs and AV1 streams. `export.py` writes LeRobot v2.1 directly; `validate.py` checks the result and optionally invokes LeRobot 0.3.3. `hub.py` publishes and verifies a revision. `cli.py` exposes `inspect-scene`, `collect`, `export-dom`, `validate`, and `upload`.

The backend interface is `reset(path)`, `state()`, `target()`, `jacobian()`, `apply(command)`, `step(next_time)`, `capture(timestamp)`, and `contact_force()`. CPU state is copied from simulator tensors. Scalar time is simulation time, never wall time. The backend owns robot joint ordering and joint/velocity limits.

## Assets, benchmark isolation, and scene assembly

Audit requires exact agreement between `DOM-Test/test-envs.txt` and its JSON inventory. Reserve all referenced scene/object paths, not merely cases whose names suggest generalization. Exclude `objects/unseen` and container categories. Select from the complement of the full supplied scene/object corpus. This deliberately over-excludes benchmark assets; it does not claim to reconstruct the authors' original training split.

Assembly searches eligible USD scenes in sorted order and chooses the first feasible table edge. It uses the DOM house X-axis +90° rotation, table `height` metadata, world bounds, inward-facing robot +X, and broad-phase clearance probes for nearby furniture/walls/cameras. Selected assets receive dependency checks and hashes, including exact-copy checks against reserved assets. The input USD files are never edited. Layouts record base pose, table, asset hashes, and audit fingerprint. Geometry-based layout selection must be followed by physics smoke tests.

One fixed scene, object, lighting setup, and object orientation remain throughout the pilot. Randomized paths are the within-distribution variation. The requested 3D hovering task is a new behavior relative to DOM's manipulation tasks, even with shared embodiment/assets.

## Frames, orientation, and controller

Units: meters, seconds, radians. Internal quaternions: **wxyz**. A quaternion maps local vectors to the parent frame. Exported geometry and target lateral velocity use the robot-base frame. Cameras store world poses with ROS optical convention (+Z forward, -Y up) and 3×3 intrinsics. World poses are converted using the recorded base translation and rotation.

Franka observation tool offset from `panda_hand`: `[0, 0, 0.1034]` m. Control tool offset: `[0, 0, 0.107]` m. Both are explicit configuration fields. For the same orientation, shifting the point uses `p_new = p_old + R(q) (offset_new - offset_old)`. The controller accounts for `omega × delta_offset` in the action-point linear velocity. Raw observations and future labels use the observation point; issued IK commands use the control point. Any future training adapter must preserve this distinction.

The hover reference is the fixed-orientation object's initial world bounding-box top-center plus `[0, 0, 0.10]` m. The object root pose is logged separately. Zero mode uses base-frame quaternion `[0, 1, 0, 0]`: tool +Z points down. A mode angle applies a left multiplication by a base-X rotation, `R_desired = Rx(mode) R_down`.

Default hysteresis (`enter=.05`, `exit=.03` m/s):

- If `vy > enter`, select −45°. If `vy < -enter`, select +45°; these rules include direct sign reversal.
- Otherwise, a −45° mode returns to neutral only when `vy < exit`; +45° returns only when `vy > -exit`.
- Exact boundaries retain the current mode. Initialize neutral, then evaluate initial velocity.

Translation: `v = v_target + Kp (p_hover - p_ee) + Kd (v_target - v_ee)`, with `Kp=4 s^-1` and `Kd=.5`. Clamp its norm to .30 m/s. Orientation uses shortest-path quaternion error as a rotation vector, gain 8 s^-1, clamped to π/2 rad/s. The desired orientation is discrete; measured orientation transitions continuously.

The tool workspace is the configured object workspace translated by the hover offset. Clamp the next requested tool displacement to this box; outside it during warm-up, allow only bounded recovery toward it. A measured workspace excursion beyond 0.1 mm during recording rejects the episode. Saturation includes linear, angular, workspace, and joint limiting.

Generated paths reserve `trajectory.workspace_margin=.01` m inside each workspace boundary, leaving room for tracking transients without widening the acceptance region. The margin also applies to the named smoke paths and continuation samples.

Shift the PhysX spatial Jacobian from the body COM to the action point and rotate both blocks into base axes. Solve `dq = J.T (J J.T + λ² I)^-1 twist`, default λ=.03. Clamp joint velocities against articulation limits and a .02-rad joint-position margin; apply `q_target=q+dt*dq` with velocity feedforward. Finger positions stay .04 m each. These are local tracking controls, not a global collision-avoidance planner. Contact/limit failures reject an episode.

## Trajectories and balancing

Each coordinate follows `p_i(t)=c_i+A_i sin(ω_i t+φ_i)`; velocity and acceleration are analytic. Centers and amplitudes remain inside the configured workspace. Scale amplitudes using global conservative bounds `||A ω|| ≤ vmax` and `||A ω²|| ≤ amax`, so bounds hold between samples. Defaults: center `[.45,0,.16]` m in base axes; half-extents `[.10,.15,.075]`; max speed .15 m/s; max acceleration .30 m/s²; frequencies .10–.28 Hz. The object is scripted/kinematic, not free-falling physics.

Each attempt uses `SeedSequence([run_seed, attempt_id])`. Generate 64 candidate paths, run hysteresis at control frequency, and score predicted *capture-frame* occupancy against accumulated accepted mode counts. Select the lowest imbalance; ties preserve generation order. Do not alter thresholds, duplicate samples, or reorder/drop frames for balancing. Aggregate acceptance is within ±.05 of one-third per mode. Stop after 500 attempted episodes by default; report failure if balance or episode targets remain unmet.

Named stationary/positive/negative/neutral/switch paths are smoke-only. Positive and negative short arcs are intended for short directional tests; long bounded paths cannot sustain an arbitrary lateral velocity indefinitely.

## Timing and raw schema

200 Hz control/physics, 25 Hz rendering/capture: exactly eight physics intervals per observation. Each episode warms up for three seconds at the initial target position before recording. Discard the renderer warm-up image. At time `t`, read state, calculate the command, render the same physics state when due, log both, then apply the command and advance physics. Camera reads do not advance physics. Camera frame counters must advance and all three exposures are assigned the same current simulation time. Frustum checks test the target anchor, not full-object visibility/occlusion.

GPU rendering keeps Fabric enabled. The pinned Lab 2.3.2 camera adapter reads the hand's live `omni:fabric:worldMatrix` and composes the authored USD camera mount, avoiding stale child-transform caches. Fixed cameras use their static USD transforms. Every wrist capture must agree with the independently read articulation pose within 1 mm and 0.001 rad; export validation repeats that check against recorded state, including continuation frames. Metadata records the transform source. Standalone shutdown disables Lab's interactive stop callback before stopping the timeline.

A 10-second episode has 250 exported rows at `0..9.96` seconds. Record 255 observation frames and continue physics to supply `action[t] = measured_pose[t+5]`. The last label therefore comes from time 10.16 seconds. No terminal padding. Raw capture frames include continuation; videos contain only the 250 exported observations. Continuation contacts/limit failures invalidate labels and reject the episode.

`controls.parquet` or `.jsonl`: every control tick records time/index, all joint positions/velocities, arm-only joint ordering, observation-tool pose/twist, hover reference, object root pose/velocity/acceleration, mode, desired quaternion, requested linear/joint velocities, limited joint targets, action-point twist/setpoint, saturation, orientation/position errors, and contact/limit status.

`frames.parquet` or `.jsonl`: capture index/time/control-step reference, object/tool state, mode, camera intrinsics, world camera poses, camera frame counters, and anchor-frustum flags. `episode.json` stores trajectory parameters, attempt seed, purpose, metrics, rejection reasons, and lengths. `provenance.json` stores configuration, layout, versions, and joint names.

## DOM export and training boundary

Write `meta/info.json` with `codebase_version=v2.1`, `robot_type=franka`, chunk size 1000; `data/chunk-NNN/episode_NNNNNN.parquet`; and three AV1/yuv420p MP4 streams under `videos/chunk-NNN/observation.images.{wrist_cam,opst_cam,side_cam}/`.

| Feature | Type | Definition |
|---|---|---|
| `observation.state` | float32[6] | Current observation-tool XYZ and XYZ Euler angles |
| `observation.environment_state` | float32[9] | Object root XYZ, XYZ Euler, linear velocity |
| `action` | float32[7] | Observation-tool pose five frames ahead, gripper +1 |
| `timestamp` | float32 | Episode-relative seconds |
| `frame_index`, `episode_index`, `index`, `task_index` | int64 | Contiguous indexes |
| three `observation.images.*` | video | RGB 360×480×3 at 25 FPS |

Match DOM's lowercase `xyz` Euler convention; wrap X and Z to `[0,2π)`. Quaternions are retained in diagnostics. Future labels are generated from this collector's recorded trajectory, not by rerunning the original DOM replay controller. Task strings are serialized instruction dictionaries with `task="track"`, object identity, empty containers, and a tracking instruction.

Supply task/episode/camera metadata, correctly weighted episode/global statistics, raw diagnostics, dataset card, provenance, and SHA-256 checksums. All episodes remain in `info.splits.train` for format compatibility; a seeded permutation generates disjoint 80/10/10 episode lists in `split_assignments.json`. Palmetto must explicitly select those episode lists. There are no held-out assets within the pilot's own splits.

Training exports also include the frozen eligibility/exclusion audit as `meta/asset_manifest.json`, with benchmark configuration hashes and the chosen layout's audit fingerprint.

The installed LeRobot 0.5.x is not used to author or migrate the dataset. A separate LeRobot 0.3.3 check loads actual rows/videos. The user's VLA training repository was absent during initial implementation. Subsequently, upstream DynamicVLA commit `434dc7105cf6d559018d15af5fecbcdad44f036e` was audited and its actual dataset reader passed 18 pilot sample checks after an isolated tracking instruction patch. Explicit episode selection overrides upstream's modulo-1000 split. See [integration details](integrations/dynamicvla/README.md) for source pins, patches, action/history checks, and remaining native simulation/replay work. Raw logs make relabeling possible without recollection. No training or checkpoint edits occur here.

The October 2 GitHub destination already has a recovered DynamicVLA fork; it remains unchanged by this collector addition. `integrations/dynamicvla/evaluate_checkpoint.py` provides offline saved-baseline/residual evaluation with generic LeRobot 0.3.3, explicit validation/test episode IDs, saved checkpoint feature/history settings and normalizers. Predictions are unnormalized action chunks; delta pose outputs receive the current measured pose once, while absolute outputs are unchanged. Mask padded action steps, compute Euclidean XYZ error and geodesic Euler-derived orientation error, and report units separately. Reset the policy at each independent anchor, feed recorded histories, and never pass action labels into inference. This measures recorded-label agreement, not physical tracking success. Real-checkpoint verification and a learned-policy bridge into the 200 Hz tracking simulator remain separate milestones.

## Recovery, publication, and acceptance

A run holds a process-level file lock. Persist `next_attempt` before simulation. Write a complete episode into a hidden partial directory, then atomically rename it into accepted/rejected storage. On restart quarantine partial directories, reconstruct accepted counts, and skip interrupted attempt seeds. Require exact run provenance on resume. Interrupted metadata writes use atomic replacement. Export to a fresh `.partial` directory, then rename; never overwrite an existing export.

Episode gates: RMSE ≤.02 m, maximum error ≤.05 m; orientation ≤5° outside a .75-second switch allowance; at least one settled orientation sample; no monitored contact force above .5 N; no joint-limit violation; ≥95% anchor-frustum inclusion in each camera. Numerical/frame failures reject attempts. Metrics include per-mode errors, occupancy, switches, saturation, contact maximum, rejection count, and fixed-bin XYZ/speed histograms. The robot base is excluded from robot contact checks because it is mounted on the table.

Publication requires an explicit HF repository ID, non-smoke data, successful local validation, and a matching successful loader report. Default private; reject an existing public repository unless explicitly requested. Refuse unrelated remote files. Upload the complete dataset without raw scene assets, record the returned commit, check remote sizes/inventory, and re-download one episode plus metadata for checksum/decode checks. Save the receipt outside the immutable dataset. Palmetto downloads that exact revision and performs training/checkpoint resumption using its own environment.

## Verification milestones

1. Numerical tests: boundary/reversal hysteresis, coordinate and offset round trips, point Jacobian, command limits, trajectory derivatives/bounds, deterministic sampling.
2. Storage/asset tests: benchmark inventory/exclusion, interrupted recovery, provenance mismatch rejection.
3. Export tests: actual AV1 streams, Parquet schema, five-frame labels including tail, metadata, checksums, splits, deliberate corruption rejection.
4. Simulator checks: asset dependency/layout assembly followed by five smoke trajectories on the NVIDIA GPU; inspect videos and gates before the pilot.
5. Reader and handoff: LeRobot v2.1 sample decoding, supplied training-loader adaptation once available, pilot collection, repository upload and revision verification.

Do not interpret offline fixture tests as simulator demonstrations or a published dataset. Actual test/run results are recorded separately in the README verification notes and generated run reports.
