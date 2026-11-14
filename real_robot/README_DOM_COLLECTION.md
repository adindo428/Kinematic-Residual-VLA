# Easy real-data collection for DOM post-training

For tracking repair, grasp-phase review, held-out splits, base-frame validation,
and consistent Euler targets, see [Dataset quality](README_DATASET_QUALITY.md).

## Static striped-ball collection (100 episodes)

Use `--static-object` for a stationary approach. This preserves the existing
DOM v2.1 fields, camera acquisition, measured robot state, commanded action
labels, 25 FPS export, and saved image preprocessing. It adds explicit static
provenance without changing any policy architecture or the existing task text.
Each SPACE start needs a fresh visible object position; that measured position
is locked separately for that attempt. No continuous object velocity or position
track is required afterwards; fresh camera images and robot feedback still are.
The tracker continues running to measure the next placement and flag visible
displacements, but losing it during an attempt does not end recording.

Stop an evaluation using its normal Q control before opening collection. Start
with a 20-success training block, saved/exported locally in a new automatically
named `scratch/real_robot/collections/static-striped_ball-train-...` folder:

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.collect \
  --task striped_ball --static-object --episodes 20 --split train \
  --viewer rerun --keyboard-source terminal --default-success \
  --robot-max-relative-target 10 \
  --image-preprocessing scratch/real_robot/training_releases/KRVLA-5fd5ef3e4256-extrapolated-80ms-grasp-v1/meta/image_preprocessing.json
```

Repeat the training block four times (80 successes). Then run the same command
with `--episodes 10 --split validation`, and once with
`--episodes 10 --split test`, using different placements in those held-out
sessions. Each invocation creates a new directory; this command does not upload
or append to the published dataset. Q exports an early-finished block; track the
actual saved/exported counts if the 25-minute session limit ends it early.

Controls for this mode:

- **SPACE:** leave the ball stationary and visible; wait for the countdown/start
  sound, then demonstrate the pick. A missing/stale initial position cancels
  this start; retry once it is visible. A new position is locked each attempt.
- Close the gripper around the ball, lift, carry, reopen into the box, then
  **ENTER** saves success. **No L or R presses are required.** Export derives
  lift/release intervals from measured gripper aperture, commanded gripper
  motion and measured TCP height. **L/R** remain optional manual markers.
- **X:** save failure if the ball rolls or is nudged before grasping. A fresh
  measured displacement over 20 mm invalidates the original position assumption
  permanently for that attempt. Hidden motion cannot be detected automatically.
  **C** may explicitly end the stationary assumption at contact; frames between
  that boundary and supported holding remain excluded from conditioned training.
- **Q:** finish/export; **T:** existing pause/re-clutch control.

Keep the gripper open during the final descent, then close around the ball.
Demonstrate smooth controlled motion and vary placements; long idle prefixes
are unnecessary. The quality sidecar labels assumed zero velocity separately
from measured velocity and missing-data zero filling. The quality processor,
DOM/DOMV/KRV reader, and native-family reader accept that explicit declaration;
they continue rejecting ordinary missing pre-grasp velocity. Existing dynamic
collection remains unchanged. Replace the task with `marker` or `orange` for
the other two objects.

Automatic phase labeling runs offline during export. It requires an earlier
open gripper, both commanded and measured closure, and a sustained measured
TCP lift of at least 20 mm. Once that lift is supported, its start is backdated
to the measured lift onset (at least 3 mm above the bottom) rather than waiting
until the entire 20 mm has elapsed. The first reopening trend ends the holding
interval. Neutral holding velocity is a model input, not a statement that the
carried object has zero physical velocity. After reopening, the old table
position is not reactivated. These labels use future telemetry and the saved
episode success outcome; they are labeled `held_success_inferred`, never manual
visual confirmation. Telemetry cannot independently distinguish an empty lift
from an attached object: ENTER remains your episode-success confirmation. Missing
measured gripper feedback or insufficient lift produces no inferred held interval.
Save unsuccessful attempts with X; an episode marked successful without supported telemetry is retained
with unresolved/excluded holding frames for review.

## Background upload and automatic marker/orange splits

For each object, this command manages the entire 100-valid-episode collection plan:
an 80-episode training session, a 10-episode validation session,
and a 10-episode test session. There is no default 20-episode recording cap.
Session assignments are fixed before recording;
no frames from a session cross splits. Use different placements for the held-out
sessions. All splits append to the existing public `ailab-Clemson/KRVLA-dataset`,
with their memberships recorded in `meta/splits.json`; these are not three new
Hugging Face repositories. Existing episodes and split memberships are preserved.

Run one object at a time:

```bash
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN DISPLAY=:1 .venv/bin/python -u -m real_robot.collect_static_dataset \
  --task marker --episodes 100
```

```bash
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN DISPLAY=:1 .venv/bin/python -u -m real_robot.collect_static_dataset \
  --task orange --episodes 100
```

The same wrapper accepts `--task striped_ball` for a fresh automatic plan.
Default state files are
`scratch/real_robot/static-collection-campaigns/{task}-100-v1/campaign.json`.
Rerun the same command to resume, or use a new `--root` for an additional campaign.
An early Q saves/processes/uploads completed attempts and pauses the campaign;
resuming collects only the remaining count. Every new episode still waits for
SPACE. ENTER labels a successful complete placement, X a failure; L/R are optional.
The wrapper advances to the next session only after the requested count is
processed and uploaded. Network or alignment failures preserve the work and
pause progression rather than count missing uploads as completed data.
An episode failing alignment is set aside with its reason, while valid episodes
from the same batch continue uploading. Excluded episodes are never counted as
uploaded or valid toward the target. On resume, the remaining valid count includes
their replacements. Completed raw attempts remain unchanged. Existing 20-episode
plans are combined by split on resume without changing recorded split memberships;
`--block-size 20` is still available when smaller recording sessions are wanted.
The five-episode upload batch is independent of the recording-session size.
The existing 25-minute capture limit can still end a long session early; resume
the same command to collect its remaining count.

Completed successes are snapshotted and processed/uploaded in batches of five
while collection continues in the foreground. The worker has lower CPU priority
and no GPU visibility. It never opens robot ports/cameras and never uploads an
open attempt or a partially written archive. Snapshots hard-link finished raw
frames; append reconciliation also reuses immutable local episode files through
hard links on the same filesystem instead of copying the entire dataset for
every batch. Both fall back to copying across filesystems. Keep these archived
files immutable. Processing can lag behind recording; the queue drains after
collection stops. Raw originals and receipts remain local. Failed network uploads
reuse their encoded batch on retry, and unchanged published sources are deduplicated.
Training-branch refresh runs after the session's raw upload queue completes.

Validation checks schema, finite action/state values, timestamps, frame counts,
image dimensions/codec and split consistency. Full video checks now run with two
file workers (two decoder threads each) and are cached by SHA-256 video contents
plus expected dimensions, frame count, FPS and validator version. Identical videos
are decoded once, including across hard links, copied exports and worker restarts.
Every subsequent check hashes the file again; changed bytes or expectations
trigger a new full decode. The first pass still warms the cache. Numeric and split
checks, append-prefix checks and remote blob comparisons are not skipped.
Cache location: `scratch/real_robot/validation-cache/videos.sqlite3`.
`KRVLA_VIDEO_VALIDATION_WORKERS=1` through `4` selects file concurrency; the
default two limits CPU contention during live collection. For an independent
fresh decode use `real_robot.dom_dataset validate --dataset PATH --force-video-decode`.
Already-running workers retain their loaded code until they exit; the next
collection session starts a worker with the current implementation.

To enable this for a normal single collection block, add
`--hub-repo ailab-Clemson/KRVLA-dataset --public --stream-upload`
to the existing `real_robot.collect` command. `--upload-batch-episodes 5` is
the default. Status/logs are inside that session's
`stream-upload/status.json` and `stream-upload/worker.log`.

To attach processing/upload from a second terminal to an already running
local-only striped-ball collection, substitute the actual session directory
printed by the collector (the parent of `recording`):

```bash
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN .venv/bin/python -u -m real_robot.stream_upload \
  --session scratch/real_robot/collections/ACTUAL_STATIC_STRIPED_BALL_SESSION \
  --repo ailab-Clemson/KRVLA-dataset --public
```

This inherits the published image preprocessing. It does not change that
already-running session's split. Its first permission/schema checks must pass
before any upload. For a configured queue that needs a retry, only
`--session ACTUAL_SESSION` is needed. Do not attach a second worker to a session
already using `--stream-upload`; the worker lock prevents concurrent writers.

## Optional raw-image crop and perspective correction for KRVLA

Raw RGB-D acquisition is **640x480**. To manually select a trapezoid in the fixed front view and rectify it before training export:

```bash
DISPLAY=:1 .venv/bin/python -m real_robot.image_preprocessing --perspective \
  --sessions /absolute/session/recording \
  --output scratch/real_robot/image_preprocessing/front-region.json
```

Use real session paths; multiple recording folders are supported. Click corners in **top-left, top-right, bottom-right, bottom-left** order. **R** resets; **N/P** browses example frames across successful episodes. Inspect the straightened preview, then Enter saves. Q cancels. The same fixed-camera region is applied across the entire export. Include the full intended ball, grasp and placement path in the selection.

The selected quadrilateral is warped to a rectangle. Its default aspect ratio is inferred from mean opposite edge lengths in the source image, so it is approximate; `--rectified-size WIDTH HEIGHT` supplies a chosen rectangle aspect instead. The 384x384 preview shows the actual result for operator approval. Perspective correction describes a plane; objects above that plane can be distorted. It does not change 3D position/action labels or the original calibration.

Export directly from the original raw RGB-D rather than reprocessing the old compressed videos:

```bash
.venv/bin/python -m real_robot.dom_dataset export \
  --sessions /absolute/session/recording \
  --image-preprocessing scratch/real_robot/image_preprocessing/front-region.json \
  --output scratch/real_robot/datasets/new-384-export
```

Front images are rectified, bilinearly resized with `align_corners=False`, then black-padded on the **left/top** to 384x384, matching KRVLA. Side/wrist keep their full view and receive only resize/pad. AV1 compression follows these transforms. All video feature shapes, encoded dimensions, image statistics and train statistics are regenerated for 384x384. `meta/image_preprocessing.json` retains the selection; camera metadata records the raw-to-exported pixel transforms. Raw 640x480 calibration remains identified as raw-image calibration. The 25 FPS timeline, numeric action/state labels, episode filtering and raw archives are retained.

Pass the same `--image-preprocessing` settings to `real_robot.collect` to apply them on future automatic exports, or let `--hub-repo` inherit the published settings. Its acquisition remains 640x480. After validation, `real_robot.hub_upload --dataset <export> --repo <owner/repo> --public --append` adds new episodes from a standalone block or cumulative export. It constructs an unchanged published prefix when needed, verifies all published episode-file hashes, skips those files, and uploads only new files plus cumulative metadata. No files are deleted and no archive branch is required; previous metadata revisions remain in Git history. Replacement is disabled.

## Add a recording block to an existing published dataset

`--hub-repo` must be present for automatic upload. A local-only recording command exports locally and does not upload. The collector now finds the current published dataset automatically and appends each block:

```bash
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN DISPLAY=:1 .venv/bin/python -u -m real_robot.collect \
  --task striped_ball --episodes 30 --viewer rerun --keyboard-source terminal \
  --default-success --robot-max-relative-target 10 \
  --hub-repo ailab-Clemson/KRVLA-dataset --public
```

Before hardware starts, the collector looks for a local export whose metadata and source list exactly match the pinned published revision. If none exists, it downloads that revision under `scratch/real_robot/hub_snapshots/`. Startup checks the base metadata, numeric rows and video headers without decoding every old frame; export and upload still perform full frame validation. The base image preprocessing is inherited automatically, so this dataset continues to use full-frame resize/pad to 384x384. No crop or perspective warp is introduced. You may still provide `--append-to` explicitly; an outdated explicit base is rejected before recording.

At upload time, `--append` accepts either a new standalone block or a cumulative export. The uploader checks the latest published source list again. If another upload advanced the dataset during recording, it builds `dom-reconciled-<revision>-<timestamp>/` from the latest published prefix plus the pending episodes. Repeated source paths are skipped only after verifying unchanged metadata, numeric trajectory data, videos and quality files; conflicting content or raw hashes are rejected. Published episode IDs, split memberships, data, videos and quality files remain unchanged. Only new episodes are reindexed; metadata and aggregate statistics describe the cumulative dataset. No video is re-encoded during this merge, no file is deleted, and no separate branch is created.

The final commit still requires the verified parent revision. A concurrent-commit conflict refreshes the base and retries, up to three reconciliation attempts. Retrying a batch already present verifies the existing source content and returns `already_uploaded` with zero new episodes instead of duplicating it. A failed collection upload prints a retry command using the standalone `dom` and `--append`; stale cumulative exports are also accepted. The final cumulative path is recorded in `upload.json` as `dataset_path`. Local-only `--append-to` merges retain the strict duplicate rejection behavior.

The most recently verified append is `scratch/real_robot/collections/striped_ball-train-20261103T170941.455648Z/dom-reconciled-acf45087a365-20261103T173027.416275Z`: **158 episodes / 38,316 frames**, revision `055548fcbe3646fe4db734a35a6b9ac265525bc2`. It retains all 128 previously published episodes (including a seven-episode batch uploaded during reconciliation), and adds the 30 pending episodes from `striped_ball-train-20261103T170941.455648Z` exactly once. All 640 previously published episode files were preserved unchanged. The stale 130-episode `dom-combined` and the raw recordings remain local.

## Prevent upload permission failures

`HF_TOKEN` overrides the cached Hugging Face login. On this workstation, a stale `.bashrc` export caused new-video uploads to fail with 403 even though the cached login could write. That export was removed; already-open terminals still need `unset HF_TOKEN`. The command above explicitly ignores token environment overrides and uses the cached login.

The collector now checks authorization for a fresh LFS object before starting cameras or robot control. The check requests upload instructions without sending file bytes or making a commit. The standalone uploader checks authorization before decoding all videos. Test it independently with:

```bash
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN .venv/bin/python -m real_robot.hub_upload \
  --repo ailab-Clemson/KRVLA-dataset --public --check-access
```

If the cached credential is changed or revoked, run `.venv/bin/python -c "from huggingface_hub import login; login()"` and enter a token with **write access to repository contents** for `ailab-Clemson/KRVLA-dataset`. This Python login works with the installed Hub version, which has no `hf` executable in `.venv/bin`. A successful account lookup or permission to read an existing file does not establish permission to upload new video contents. Keep credentials out of commands, chat, and dataset files. Preflight verifies current permissions; later network failures or revoked credentials still require retrying the saved block or cumulative export with `--append`.

## Small orange: pick and place

Prepare its front-camera EfficientTAM reference with the orange visible on the table:

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.ramp_velocity --object orange --mask-check
```

Select the whole orange, Enter confirms, and Q finishes the preview. To redraw the mask later, repeat the same command. The saved reference is `scratch/real_robot/object_profiles/orange/on_ramp`. Collection with `--task orange` uses this reference and the instruction **Pick up the rolling orange and place it in the box.** Unlike the pickup-only bottle task, orange success includes placing/releasing it into the box before Enter. Both fixed cameras and the wrist camera are recorded; object estimates use the front mask without requiring a side mask.

After closing the preview and stopping standalone teleoperation, an example collection block using the requested per-motor limits is:

```bash
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN DISPLAY=:1 .venv/bin/python -u -m real_robot.collect \
  --task orange --episodes 100 --viewer rerun --default-success \
  --robot-max-relative-target '[20,20,20,20,20,20,20,20,35]' \
  --hub-repo ailab-Clemson/KRVLA-dataset --public
```

## Blue bottle: pickup only, with per-motor teleoperation limits

Prepare its independent EfficientTAM reference first:

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.ramp_velocity --object blue_bottle --mask-check
```

Select the whole bottle, including its neck and cap, in the front-camera window. Inspect the mask across the ramp/rollout, then Q finishes the preview. The saved profile is `scratch/real_robot/object_profiles/blue_bottle/on_ramp`.

To redraw/reapply this mask later, close any running collection or camera preview and repeat:

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.ramp_velocity \
  --object blue_bottle --mask-context on_ramp --mask-check
```

The newly accepted selection replaces the saved on-ramp reference. New collection sessions automatically use it. To inspect/reapply the saved reference without drawing a new mask, replace `--mask-check` with `--preview-only`. See `README_RAMP_VELOCITY.md` for both full commands.

After closing the mask preview and stopping standalone LeRobot teleoperation to free the right-arm ports, collect with:

```bash
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN DISPLAY=:1 .venv/bin/python -u -m real_robot.collect \
  --task blue_bottle --episodes 100 --viewer rerun --keyboard-source terminal \
  --default-success --robot-max-relative-target '[20,20,20,20,20,20,20,20,35]' \
  --hub-repo ailab-Clemson/KRVLA-dataset --public
```

The instruction is **Pick up the rolling blue bottle.** Press Enter after grasping and lifting it while it is still held; neither release nor box placement is required. Reset/remove the bottle after the save cue, outside the recorded episode. Space starts the next attempt after the one-second countdown. The export records DOM `task: pick`, object `rolling blue bottle`, and no container target. Existing placement tasks retain their labels when bottle episodes append to the same dataset.

Both collection entrypoints accept `--robot-max-relative-target` and `--robot.max_relative_target` as a positive scalar or a JSON list of exactly nine positive finite limits. The vector is retained in the recording manifest and applied in the same motor order as the right ALOHA bus: 20 for each of the first eight motors and 35 for the gripper. The per-step clamp matches the installed LeRobot implementation, with collection's existing 30 Hz control and clutched leader mapping. Raw capture is 640x480, with the dataset's inherited full-frame resize/pad to 384x384 for export. The default 25-minute session limit can end a block before 100 successes.

## Marker manipulator collection

Reuse the saved front-camera mask under `scratch/real_robot/object_profiles/marker/on_ramp`:

```bash
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN DISPLAY=:1 .venv/bin/python -u -m real_robot.collect \
  --task marker --episodes 30 --viewer rerun --keyboard-source terminal \
  --default-success --robot-max-relative-target 10 \
  --hub-repo ailab-Clemson/KRVLA-dataset --public
```

Startup prints the selected task and full instruction before the shared append-base path; a `striped_ball-...` base folder describes previous episodes, while new sessions use `marker-train-...` and the green-marker instruction. This records all three camera views and uses the front mask for marker position, following the striped-ball capture setup. No side-mask selection is required. The task instruction is **Pick up the moving green marker and place it in the box.** Marker episodes append to the same repository with their own task label and the inherited 384x384 export settings. Capture starts paused. Keep the launching terminal focused: Space starts after one second with a sound, Enter saves success with a sound, X saves failure, T pauses, and Q finishes/export/uploads. Thirty means operator-success attempts; alignment checks can reduce the accepted export count.

## Striped-ball manipulator pilot with Rerun and upload

The ball references must already exist under `scratch/real_robot/object_profiles/striped_ball/on_ramp`. Front RGB-D supplies its object estimates; both fixed RGB-D streams plus wrist RGB-D are recorded. Side target visibility is optional. Prepare masks with the ramp mask workflow before collection.

```bash
DISPLAY=:1 .venv/bin/python -m real_robot.collect \
  --task striped_ball --episodes 10 --viewer rerun --default-success \
  --robot-max-relative-target 10 \
  --hub-repo ailab-Clemson/KRVLA-dataset --public \
  --output scratch/real_robot/pilots/striped-ball-manipulator-01
```

Use a new output name for every session. Rerun 0.22.1 is installed in the robot environment (`.venv/bin/python -m pip install rerun-sdk==0.22.1` if rebuilding it). Hugging Face authentication uses the standard `HF_TOKEN` environment variable or cached token; tokens are never written into dataset files. Prefer the working cached login on this workstation; see the permission check above.

The collector starts **paused**. Support an unpowered follower and hold the leader before the first Space press. Space can enable holding torque at the follower's current pose and clutches relative leader movement there. The bound of 10 is in calibrated joint degrees / gripper percentage points per control step. It changes the target clipping bound; it does not change the 30 Hz command rate or guarantee a latency improvement.

With Rerun collection, **keep the launching terminal focused** for Space, Enter, X, T and Q. Each press immediately prints a saved/recording acknowledgment and the success count. Keys are read directly from the terminal with no global listener by default; the terminal settings are restored on shutdown. `--keyboard-source global` explicitly enables the previous desktop listener instead. Space arms a one-second countdown, then recording starts with an ascending two-tone sound. Enter stops it with a descending two-tone sound; both actions print an acknowledgment. Repeated Enter while idle cannot start another episode.

Rerun has a large **Episode confirmation** banner above the camera views. After Enter completes an episode, it displays **EPISODE 1 SAVED - SUCCESS**, the READY state and saved-success count; the next recording begins only after Space. This acknowledgment stays visible until another episode is saved, including if the viewer timeline is paused. Failed/incomplete episodes use red/amber feedback. The final saved episode remains visible when collection finishes. Restart an already-running collector to load this UI change.

- **Space** enables clutched following if paused and arms a **one-second countdown**. No episode or camera archive is open during the countdown. Recording starts automatically after the delay, accompanied by an **ascending two-tone cue** and RECORDING banner; push the ball after that cue. Repeated Space while counting down or recording does not open another episode. Enter/X cancel a pending countdown without creating an episode.
- **Enter** stops the open episode and saves it as operator-successful, with a **descending two-tone cue**. The green banner remains visible while you reset; no episode is open during this gap. Following remains enabled so you can reposition the arm.
- **X** stops the open episode and saves it as failed, with a lower descending cue. Interrupted episodes also emit that cue. Reset, then Space to start its replacement.
- **Q** (or Escape with global/OpenCV input) finishes collection. Stop a completed episode with Enter first: Q leaves an open episode incomplete. The tenth successful Enter ends the block automatically without opening an eleventh episode.
- **T** pauses following and fails an open attempt. T/Space can re-clutch after a pause. Hardware/camera faults or duration expiry leave open attempts incomplete.

Sound playback runs in a background worker and does not block robot feedback, heartbeat, camera checks or keyboard input. Desktop playback was checked with `paplay`; `aplay` is the fallback. Use `--no-audio` to disable cues or `--start-delay SECONDS` to change the default one-second delay. Sounds and delay apply to `--default-success` collection; the legacy N/V workflow retains its immediate recording behavior.

There is no automatic physical success check in this manual mode. Enter supplies the success label and the end boundary. Reset the scene after Enter and before Space so reset motion stays outside the recorded episode. The viewer shows wrist RGB and the fixed-camera previews plus current status. Preview compression is separate from the raw archives and export video encoding.

The block targets the requested number of operator successes; export rejects temporally invalid attempts, so check the actual accepted count in `dom/export_report.json`. After export validation, the command publishes the cumulative export to the **public** dataset repo and writes the commit/counts to `upload.json`. Raw sessions stay local. With `--hub-repo`, the newest published append base is found automatically.

For an upload retry after a network failure:

```bash
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN .venv/bin/python -m real_robot.hub_upload \
  --dataset scratch/real_robot/collections/striped_ball-train-20261103T165537.539526Z/dom-combined \
  --repo ailab-Clemson/KRVLA-dataset --public --append
```

The controls below describe the older explicit N/V workflow, which remains the default unless `--default-success` is supplied.

Run this **one command** from the repository:

```bash
.venv/bin/python -m real_robot.collect
```

It defaults to blue tape, training split and a block of 20 operator-successful
attempts. It starts/stops perception itself, uses 30 Hz cameras, reuses saved
object masks, names a new session and exports successful data to DOM v2.1 after
recording. No separate LeRobot teleoperation or five-degree leader/follower
matching command is needed. The stationary grasp recorder is unchanged.

## What you do

1. Focus the wrist recorder window and press **N**. On first use this also
   enables holding torque at the current follower pose if all motors are off,
   and clutches the leader at that pose. Support an unpowered arm before N;
   there is no automatic homing or move toward the leader's absolute pose.
2. Wait one second for pre-roll, then manually roll the tape into view and use
   the leader to grasp/lift/place it. The object may start outside the cameras.
3. After successful release and two seconds inside the box, press **V**.
   Press **X** for a failed attempt. Reset outside the recording, then **N** again.
4. **Q** finishes early and exports the successful recordings. Otherwise the
   block stops after 20 operator-success attempts and exports automatically.

**T** is an optional pause/re-clutch. Every re-enable anchors leader motion at
the follower's current pose, preventing an absolute-pose jump. The per-step
command bound, heartbeat and hardware-health checks remain. Pausing an open
attempt marks it failed. Q retains holding torque; support the arm before using
the separate torque-off procedure.

Markers for launch/contact/lift/release remain optional; there is no requirement
to press them or hold still during demonstrations.

## Saved masks and other objects

Your blue tape prompts have already been imported and verified with both fixed
cameras. The persistent library is `scratch/real_robot/object_profiles/blue_tape`.
Each camera has the reference image, actual EfficientTAM mask and camera metadata.
On later runs, EfficientTAM loads the reference into its memory and tracks the
current frames; it does not paste a mask at the reference image's old position.
Saved assets are checked against the camera serials. Masks use the object name,
so a tennis-ball mask cannot silently replace the blue-tape profile.

For a new object, change only its task ID:

```bash
.venv/bin/python -m real_robot.collect --task tennis_ball
```

Select that object once in the front view and once in the side view. Both masks
are saved for future sessions. Available tasks are `blue_tape`, `tennis_ball`,
`orange`, `ping_pong_ball` and `small_container`.

To deliberately replace the current object's masks:

```bash
.venv/bin/python -m real_robot.collect --task blue_tape --new-mask
```

The reference image must contain the object when creating a new mask. Subsequent
recording can start while the object is outside view. Reacquisition remains an
estimate: if the tracker loses the object, raw video is still saved and the
export records missing/stale state rather than claiming the pose is observed.
Collection recovery uses the selected object's reference appearance and
cross-view geometry, rather than a blue-only gate.

## Sessions, splits and output

Examples:

```bash
.venv/bin/python -m real_robot.collect --task orange --split validation
.venv/bin/python -m real_robot.collect --task ping_pong_ball --episodes 10
```

Default output is `scratch/real_robot/collections/<task>-<split>-<timestamp>/`:

- `recording/`: robot commands, feedback, episode/attempt ledger and self-contained
  front/side/wrist RGB-D data.
- `perception/` and `perception.log`: tracking provenance and diagnostics.
- `dom/`: validated DOM v2.1 parquet, AV1 videos and metadata after export.
- `dom.partial/failure.json`: diagnostic output if no usable export can be built.

Each session has one object and one split. Camera images are archived only
during open attempts, so idle/reset periods do not accumulate large videos.
Perception keeps only small telemetry/preview files when not recording. Nothing
is uploaded to Hugging Face. Back up raw sessions and exports before deleting data.

Twenty V presses do not guarantee twenty accepted training episodes: inspect
the final export count/report and make up any rejected examples in a later
session of the same split. Keep pilot sessions separate using an explicit
`--output scratch/real_robot/pilots/<new-name>` and do not include them in the
final dataset. The five-task size/speed/split plan remains in
[the collection protocol](README_DATA_COLLECTION.md).

## Fixed-camera interruption and recovery

Collection keeps the RGB/depth synchronization tolerance and cross-camera pairing gates unchanged. A front/side frame gap over 0.75 s now **pauses following**, cancels a pending start countdown, and closes an open episode as **incomplete**. Image archiving stops during the gap; earlier successful episodes remain saved. Rerun shows CAMERA RECONNECTING.

A camera read/synchronization error triggers a bounded stream close/reopen for that serial, with at most **three reconnects per camera per session**. Perception continues publishing health while waiting for frames. After fresh paired frames return, following stays paused: press Space to re-clutch and start a new episode. There is no automatic continuation across a camera gap. Repeated failures beyond the reconnect limit, a perception process hang, or a failed SDK shutdown stop the session.

Camera transitions are recorded in `perception/camera_events.jsonl`; `observation.json` and `summary.json` include `camera_health`. Reopening streams can recover transient SDK desynchronization but cannot repair USB bandwidth/power/cable faults. Check both fixed cameras and the wrist together, with EfficientTAM, Rerun and recording enabled, when diagnosing intermittent failures.

## What changed about stale frames

Manual collection no longer requires a fresh object pose before N. The recorder
checks that perception is publishing and keeps raw evidence when the target is
not visible. Collection processes the newest paired camera frames instead of
replaying an old queue after selection or slow processing. A permanent fixed-camera/perception failure or serial fault still stops collection;
an absent object does not. The wrist camera runs in a separate process: if it
stalls, following pauses and the open attempt is saved as incomplete. A watchdog
restarts the camera without blocking robot control. Wait for the live preview,
then press N to re-clutch and start a new attempt; following never resumes by
itself. Persistent wrist faults stop the session after three restarts. Earlier
successes are still exported when a recorder fault ends the session. Wrist
diagnostics are in `recording/wrist/camera_events.jsonl`.

The wrist-process check completed three 12-second recordings separated by reset
intervals, saving 1,079 wrist frames and all 879 submitted fixed-camera pairs.
No wrist restart was needed and shutdown took under one second. Injected native
hangs separately verified forced shutdown and automatic reconnection. All 86
regression tests passed; these checks did not connect robot motors. Report:
`scratch/real_robot/verification/wrist-process-cycles-20261022-01/result.json`.

The raw session retains the off-camera pre-roll. DOM training export begins
after the first available in-episode object estimate, rather than filling the
initial off-camera interval with an invented pose. Later occlusion uses the
documented held-position/zero-velocity policy and quality sidecars below.

The initial N-triggered recording failure was archive queue overflow, not lost
calibration: a single default-compression worker could not sustain two fixed
camera streams. Collection now uses four background workers and lossless NPZ
zlib level 1; the RGB and float32 depth arrays are unchanged. Delayed files are
buffered until atomic publication, retaining their original observation receipt
times separately from disk completion. Failures report the actual perception
error. A 20-second real recording test saved all 498 submitted fixed-camera
pairs plus 600 wrist frames, with peak queue depth one and no robot connections.
Results: `scratch/real_robot/verification/recording-throughput-20261022-01/result.json`.

The camera-only restart verification reused both saved tape masks without
selection, produced valid positions on all 161 processed pairs, and had p95
processed-frame age approximately 110 ms. This did not command or energize motors.
Supervised grasp performance and roll-in reacquisition still need to be checked
on your next physical attempt.
An additional offline image test removed the tape for 16 frames and reintroduced
it shifted by 30 pixels: both saved view models returned visible masks on all
12 subsequent frames. This is a synthetic vision check, not a real rolling or
grasping trial.

## Advanced manual export

The one-command path normally exports automatically. To combine completed raw
sessions into a new version later:

```bash
.venv/bin/python -m real_robot.dom_dataset export \
  --sessions /absolute/session-one/recording /absolute/session-two/recording \
  --output scratch/real_robot/datasets/dom-real-v1

.venv/bin/python -m real_robot.dom_dataset validate \
  --dataset scratch/real_robot/datasets/dom-real-v1
```

Use real existing session paths. Outputs are never overwritten. Interior
camera/robot gaps reject the attempt instead of shortening physical time.
Default acceptance ages are 250 ms for cameras, 120 ms for robot samples/actions
and 300 ms for current object state; these are acceptance limits, not guaranteed
latency. Estimates do not require external ground-truth validation.

## DOM format and estimated state semantics

The exported **feature schema matches the published DOM `meta/info.json`**:

| Feature | Values |
|---|---|
| `action` | 7 float32: commanded TCP xyz, xyz Euler radians, gripper (+1 open / -1 closed) |
| `observation.state` | 6 float32: measured-encoder FK TCP xyz and xyz Euler radians |
| `observation.environment_state` | 9 float32: estimated object xyz, assumed xyz Euler, estimated linear velocity xyz |
| `observation.images.opst_cam` | Fixed front RGB, 480×360, AV1, 25 FPS |
| `observation.images.wrist_cam` | Wrist RGB, same format |
| `observation.images.side_cam` | Fixed side RGB, same format; extra perception evidence |
| Index fields | DOM `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` |

Video metadata includes DOM's `video_info` and `info` fields. Layout, chunk size
(1,000), per-episode parquet/videos, task JSON strings and episode statistics
follow LeRobot v2.1. Robot type is truthfully `aloha_solo` rather than Franka.
Task metadata supports both DOM's original instruction generator and forks
that use the explicit `instruction` string.

All translations/velocities are in the right robot base frame, in meters/m/s.
Euler conversion matches DynamicVLA's helper: SciPy `xyz`, radians, with X/Z
modulo 2π. Commands come from **issued, clipped follower targets**, converted
back through the saved motor calibration and FK; labels are not next measured
positions. Raw-to-command conversion is checked against the pinned LeRobot bus.
The existing DOM loader performs its normal delta conversion; export actions
remain absolute. No checkpoint/model architecture change is needed for these
feature keys and shapes.

Position and velocity are **estimates**, accepted without external ground-truth
validation as requested. Position is the tracker's visible-surface proxy, not a
verified center. Current EfficientTAM/depth tracking does not measure spin or
full object orientation. The three object Euler fields therefore hold the
declared initial heading (zero roll/pitch); `--heading-deg` is an assumption.
When position disappears, the exporter holds its latest causal estimate within
the same attempt; unavailable velocity becomes zero. This preserves full pick
and placement sequences through object occlusion. `quality/*.jsonl` explicitly
records stale position, zero-filled velocity and assumed orientation, together
with per-frame source paths/ages. These flags are sidecars so the DOM input
schema is unchanged. The standard loader does not automatically mask these
values; inspect the report and use the same fill policy at deployment. The
translation-only KR path can consume xyz/velocity directly.

The dynamic path requests **30 Hz** from all three cameras. Live device-profile
enumeration confirmed RGB BGR8 and depth Z16 at 640×480/30 Hz on the collection
D455/D405s; no 25 Hz profile is advertised at that resolution. An eight-second
simultaneous camera-only test delivered approximately 30 FPS on all three with
zero frame-number skips; both fixed-camera profile checks passed. The report is
`scratch/real_robot/verification/camera30-profile-check.json`. This measures
camera acquisition, not EfficientTAM and archive throughput. Stationary recording
keeps its 15 Hz default. The exporter reports requested rates and achieved
unique-frame cadence for each episode.

Images/state use causal zero-order hold from samples already available at the
anchor. Issued action labels use the next command within the age limit and
never cross the end of the attempt. Native 30 Hz images are sampled onto a
regular 25 FPS timeline. If acquisition or perception falls below 25 Hz, repeated
source frames remain explicitly visible in quality sidecars; a requested 30 Hz
profile is not a sustained full-pipeline rate claim.
Raw hardware timestamps and host receive/record times remain available. This
export does not assert hardware exposure synchronization. Do not drop interior
anchors or use future-smoothed object estimates as online policy inputs.

## Fine-tuning split wiring

The exporter writes explicit episode IDs in `meta/splits.json` and arranges
episodes in contiguous train/validation/test ranges in `meta/info.json`.
`meta/train_stats.json` aggregates **training episodes only**. Standard v2.1
`episodes_stats.jsonl` remains available for whichever episodes a reader selects.

**Pass the explicit IDs to the DOM loader.** Its historical default modulo-1000
split does not represent these recording sessions. In the training environment:

```python
import json
from pathlib import Path
from utils.datasets import LeRobotDataset

root = Path('/absolute/path/to/dom-real-v1')
splits = json.loads((root / 'meta/splits.json').read_text())
train = LeRobotDataset(
    repo_id='local/dom-real-v1', root=root, episodes=splits['train'],
    split='train', pin_memory=False, delta_action=True,
    required_features=[
        'action', 'observation.state', 'observation.environment_state',
        'observation.images.opst_cam', 'observation.images.wrist_cam',
    ],
    delta_timestamps={'observation': [-2, 0], 'action': list(range(20))},
)
```

For config-driven runs, place/copy the exported root under the training
environment's `HF_LEROBOT_HOME/<owner>/<dataset>`, set `DATASET.NAME` to that repo
ID, and set `DATASET.EPISODE_SPLIT_FILE` to `meta/splits.json`. Use validation IDs
for model selection and keep final test separate. Keep each architecture's
checkpoint, horizon, required inputs and normalization adaptation explicit;
matching DOM storage does not make inherited Franka statistics appropriate for
the right follower. Start from the intended DOM-trained checkpoint and use the
same real training IDs and normalization policy for baseline/KR comparisons.

Offline verification:

```bash
.venv/bin/python -m unittest \
  real_robot.tests.test_dom_collection \
  real_robot.tests.test_grasp_capture \
  real_robot.tests.test_rgbd_archive -v
```

The format tests generate synthetic raw sessions, encode/decode real AV1 videos,
check parquet/time/split consistency, verify the actual LeRobot v2.1 reader and
the repository's DOM training reader (two camera views, t−2/t, 20-step delta
actions and final-chunk padding), and exercise missing samples, action conventions
and lease expiry/exclusion. This workstation's old Conda FFmpeg has no AV1
decoder; the isolated DOM-reader test selects the installed system FFmpeg 4.4
libraries for that subprocess only. It does not change the robot environment.
Actual fine-tuning should use the existing DOM training environment with working
AV1/TorchCodec support. Tests do not move hardware. A supervised physical pilot
is still needed to measure collection timing and grasp performance.

## Cleanup performed before implementation

On 2026-10-22, removed **300,756 unreferenced old tracking RGB-D NPZ files**,
freeing **602.47 GiB (646.90 GB)**. Preserved frame references found in saved
grasp/pick observation/event records, as well as calibration, grasp data,
templates, pick recordings and telemetry logs. Audit:
`scratch/real_robot/cleanup-20261022/{plan.json,deleted.jsonl,result.json}`.
Old `fusion.jsonl` logs may reference deliberately deleted idle tracking frames;
they are historical telemetry, not complete training episodes.

Schema reference: [published DOM metadata](https://huggingface.co/datasets/hzxie/DOM/raw/main/meta/info.json).
Local convention references: `third_party/DynamicVLA/scripts/create_lerobot_dataset.py`,
`third_party/DynamicVLA/utils/helpers.py`, and `utils/datasets.py`.

For a camera-only blue-tape speed study across two ramps, see [the ramp velocity instructions](README_RAMP_VELOCITY.md).
