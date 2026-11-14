# Check and calibrate the fixed front camera against the right arm

Current decision: reuse the existing October 19 camera-to-right-base transform,
as explicitly accepted by the user in `base_transform_acceptance.json`. It
matches all 319 collected episode records. The processed v2 datasets and
depth-corrected rig record this acceptance separately from numerical validation.
The contact procedure below is optional for future independent validation;
it is not required to continue under the user's decision.

This procedure uses the current front camera, a known chessboard or ChArUco board, and
supervised clutched leader control. It checks the **physical fingertip midpoint**
against camera observations at locations excluded from fitting. It also checks
RGB-D depth against those contacts. It does not launch automatic probing.

## Current board: plain 10×10 chessboard, 31 mm squares

The replacement board is a plain checkerboard. It has no ArUco markers, so the
earlier ChArUco command cannot recognize it. Interpreting 10×10 as **squares**
gives a 310×310 mm pattern with **9×9 internal intersections (81 points)**.
If the count instead describes ten internal intersections, the physical board
has eleven squares per side; use `--board-squares 11 11` in that case.

Finish the old process with Q in its window. Leave the fixed cameras/base in
place and mount the board flat and rigidly in the pickup workspace. Run:

```bash
CALIB_SESSION="scratch/real_robot/base-calibration-chessboard-$(date -u +%Y%m%dT%H%M%SZ)"
DISPLAY=:1 .venv/bin/python -u -m real_robot.base_calibration capture \
  --output "$CALIB_SESSION" \
  --board-type chessboard --board-squares 10 10 --square-mm 31 \
  --tcp-offset-m 0.160 0 0 --mounts-unchanged --execute
```

Focus the calibration window. Wait for **81/81 corners** and **BOARD READY**,
then press B with hands/arm clear to lock the reference. Confirm the numbered
pattern is the board on the table, not another checkerboard in the background.
Hold the leader and support any unpowered follower before T enables following.

The highlighted fit points are IDs **0, 8, 72, 80**; reserved checks are **20,
60**. IDs are viewer labels for internal intersections, not printed markers or
square centres. Use the displayed pitches, approximately 55° for IDs 0,72,20
and 90° for IDs 8,80,60. The detector matches symmetric flips/transposes to its
initial/locked image grid to retain the same labels within each placement.
It does not claim a plain chessboard has unique physical marker identities.

The preview now displays **actual pitch** from encoder FK alongside the target.
S requires the measured pitch to be within 2° of the displayed 55°/90° target.
This guarantees sufficient separation of the low/high pitch groups instead of
discovering inadequate orientation diversity after all twelve contacts.

Before locking B, the tool checks camera-space RGB-board/depth agreement. A
maximum gap over 10 mm is rejected before any contacts: two observations more
than 10 mm apart cannot both agree with the same physical contact to within the
5 mm acceptance limit. This is separate from incorrect gripper contact. If this
check fails despite independently confirmed print dimensions, inspect native
color versus aligned-depth camera profiles before collecting more contacts:

```bash
CAM_CHECK="scratch/real_robot/camera-check-$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python -u -m real_robot.camera_check \
  --session PATH_TO_PREVIOUS_CHESSBOARD_SESSION --output "$CAM_CHECK"
```

Leave the complete board visible with hands/gripper clear. This camera-only
diagnostic opens no robot serial ports and changes no calibration or depth-scale
parameters. It records twelve board observations with native color and aligned
depth ray models in `camera_check.json`, plus the first RGB-D image. A successful
diagnostic capture does not itself validate a camera-to-base calibration.

The 2026-11-07 camera check found identical native-color/aligned-depth ray
parameters and a median maximum board/depth disagreement of 70.5 mm. The user's
approximate straight-line RGB-lens-to-board-center measurement, 865–870 mm,
is close to the RGB estimate (862.6 mm), while depth estimated 917.9 mm. This
supports the RGB range at that point; it does not establish a global depth-scale
correction. The confirmed printed grid is 310×310 mm. Retain the original data.

An RGB-referenced software depth correction has now been fitted from the saved
first placement, using the RGB geometry as the user-confirmed reference:

`corrected_z = 1 / (1 / raw_z + 0.0747559004574122)`, with Z in metres.

Artifacts are in `scratch/real_robot/depth-calibration-rgb-reference-20261107/`:
`depth_correction.json`, `depth_fit_report.json`, corrected copies of the three
saved RGB-D frames, and an opt-in `config-depth-corrected.yaml` pointing to
`rig-depth-corrected.json`. The first placement's RMS disagreement decreases
from 63.1 to 4.6 mm; the other saved captures decrease from 58.1/58.2 to
4.0/4.5 mm. Maximum residuals remain 9.9–10.7 mm. Those other captures were
inspected during model comparison, so these are evaluation figures, not blind
validation. The existing 5 mm maximum-error criterion has not passed.

The original fit-only correction covers raw depths approximately 0.799–1.033 m.
The collection config now uses `depth_correction_collection.json`, which applies
the same fitted function across the existing 0.1–1.5 m front tracking range and
records extrapolation outside measured support. This accommodates the recorded
manipulation workspace; extrapolated accuracy is not independently established.
Missing/out-of-application-range values remain unavailable. The model is scoped
to matching front-camera profiles, with identity/counts recorded in frame timing
and object-level extrapolation flags in tracking quality. The corrected config
can be supplied with `--config` to collection or
base-contact capture. It retains the original base transform and its validation
status, with **explicit operator acceptance** permitting training use. Historical
positions, velocities and training statistics have been regenerated in
`scratch/real_robot/KRVLA-quality-319-v3-depth`; see the dataset-quality README
for its missing-source exclusion and extrapolation flags.

Reproduce the offline fit without opening any hardware:

```bash
.venv/bin/python -m real_robot.depth_calibration \
  --session scratch/real_robot/base-calibration-20261107T220513Z \
  --camera-check scratch/real_robot/camera-check-20261107T224757Z \
  --output scratch/real_robot/depth-calibration-NEW
```

Optional investigation of whether depth error depends on the board texture
(not required to produce the saved RGB-referenced correction):

```bash
TEXTURE_CHECK="scratch/real_robot/texture-check-$(date -u +%Y%m%dT%H%M%SZ)"
DISPLAY=:1 .venv/bin/python -u -m real_robot.camera_texture_check \
  --session scratch/real_robot/base-calibration-20261107T220513Z \
  --output "$TEXTURE_CHECK"
```

Keep the entire board visible, flat and stationary. Focus the preview and press
B when it says BOARD READY. After the exposed samples finish, lay a thin, matte
piece of paper with irregular text/marks, approximately 10×10 cm, flat over the
board center. Keep the board fixed and the outermost internal intersections
visible. Clear hands and press S. Q exits. The red circle marks the sampled
center; green points mark outer corners checked for image consistency. A change
in center depth on changing only the surface texture is evidence of target
dependence. An unchanged error leaves the cause unresolved. Neither outcome
validates the rig. Image consistency cannot prove an unchanged 3-D plane; the
operator must keep the support fixed and paper flat. Results and RGB-D snapshots
are saved in `texture_check.json`, `exposed.*`, and `covered.*`. This check opens
no robot serial ports and changes no camera calibration or depth scale.

**For each point: align the closed-gripper fingertip midpoint, hold still, then
press S.** S freezes the measured contact TCP. The gripper can hide the complete
checkerboard during contact, so then lift the gripper and hands clear while
leaving the board fixed. The tool saves the staged contact only after a fresh
complete-board observation confirms the reference geometry is unchanged. It
does not replace the recorded contact with the withdrawn gripper's position.
Wait for `Saved N/12` before the next point. An occluded or moved board cannot
silently qualify a contact. An unverified pending contact is excluded from
`contacts.json`; complete verification within 30 seconds or restart the session.

After six contacts, the tool pauses following. Clear the arm, support the board
approximately 50 mm higher, press B, and repeat the six contacts. Use a rigid
support and keep the board still within each placement. The fitting command is
unchanged:

```bash
.venv/bin/python -m real_robot.base_calibration fit \
  --session "$CALIB_SESSION" --max-error-mm 5
```

The 5 mm physical contact/depth acceptance checks and robot timing guards remain
unchanged. After a pass, use the generated `config-validated.yaml` for collection
as described below. The new board's dimensions and target plan are recorded in
its session and validation report. Old board captures/rigs are retained with
their original dimensions.

## Earlier ChArUco board and recovered measurements

The earlier measurements have been recovered; do not repeat the whole measuring
exercise first:

- User-confirmed board: 6×6 squares, 50 mm square sides, 300×300 mm overall.
- User-reported wrist-bend pivot to fingertips: approximately 220 mm.
- Corrected user-reported wrist-rotation centre to fingertips: approximately
  160 mm. The earlier 110 mm answer was superseded by that correction.
- The model bend-to-gripper-link distance is 69.744 mm. The two approximate
  ruler measurements disagree with that spacing by 9.744 mm. The existing
  collector uses `[0.160, 0, 0]` m from `gripper_link`, not from the bend pivot.

The old empty `tcp-reference/measurement-reference.json` has been reconciled
with those saved statements. This is provenance recovery, not a new accuracy
claim. The fit below can refine the axial TCP length using only fit contacts,
with varied tool pitch to distinguish it from a camera translation error.

## 1. Prepare the board and arm

Finish any collection/tracking process with its normal Q control so the camera
and serial ports are free. Leave the fixed camera mounts and robot base in their
recorded positions. Place the board flat on a rigid support in the actual pickup
workspace, with its printed face visible to the front camera. Keep it stationary
for the first six contacts. Have another rigid support ready to raise the board
by approximately 50 mm for the second six contacts.

The known print geometry is the 6×6 `DICT_4X4_50` non-legacy board used by the
existing calibration, with nominal 37.5 mm black marker sides. If this is a
different print, correct `--square-mm` and `--marker-mm` before capture. Do not
move the camera to improve the view; adjust the board inside the workspace.

Use the same closed-gripper fingertip midpoint for every point. Align that point
gently with the highlighted chessboard intersection; do not press on or flex the
board. Use the highlighted IDs, not square-centre labels from the old hover-grid
experiment. Board corners and square centres are different target definitions.

## 2. Start the supervised capture

From the repository root:

```bash
CALIB_SESSION="scratch/real_robot/base-calibration-$(date -u +%Y%m%dT%H%M%SZ)"
DISPLAY=:1 .venv/bin/python -u -m real_robot.base_calibration capture \
  --output "$CALIB_SESSION" \
  --tcp-offset-m 0.160 0 0 \
  --square-mm 50 --marker-mm 37.5 \
  --mounts-unchanged --execute
```

Focus the **OpenCV calibration window** for these keys:

Startup initializes the camera, detector and window before starting the robot
controller. It limits OpenCV/BLAS processing threads and requires eight
consecutive read-only motor-feedback cycles within 150 ms. No following starts
during this warmup. The running controller retains its 150 ms feedback and
200 ms cycle guards. `startup.json` records warmup timing, and
`capture_summary.json` records completion/errors and the latest health/state
read timing. If startup fails, retain that folder and retry with a new
`CALIB_SESSION` timestamp; an existing output folder is never overwritten.

The preview reports detected ChArUco corners/markers and the actual reason B
is unavailable. A still board with zero detections is a visibility/recognition
issue, not a request to hold it still longer. Small markers are retried at 3×
image scale, then their coordinates are mapped back to the original 640×480
image before pose estimation/deprojection. Calibration and depth images remain
at the original resolution. Wait for **BOARD READY**, then press B in the
calibration window. If no numbered intersections appear, keep the fixed camera
in place and check that the entire marked print is visible, unoccluded and lit.
`board_status.json` and `preview.jpg` are saved once per second for diagnosis.

1. Keep the arm and hands clear of the board; hold the board still for a second.
   Press **B** to record the board's camera pose and unoccluded depth.
2. Hold the leader and support any unpowered follower before pressing **T**.
   T clutches at the current follower pose, using the same control as collection.
   It may enable holding torque. Further T presses pause/resume following;
   pausing preserves holding torque.
3. Move the fingertip midpoint to the red highlighted intersection. Hold still
   for approximately one second and press **S** to confirm/save the contact.
   The tool requires fresh camera data and stationary encoder/TCP readings.
4. Repeat for the six highlighted corners. Corners 0, 4, 20 and 24 are fit
   points; corners 6 and 18 are reserved checks. Use the displayed tool pitches:
   approximately 55° downward for IDs 0, 20, 6 and 90° for IDs 4, 24, 18. Pitch
   is the gripper's long axis relative to horizontal. Keep the fingers' midpoint
   on the target while changing pitch. Diverse orientations are necessary to
   separate an incorrect TCP length from camera translation.
5. After contact 6, following pauses. Move the arm clear, raise the supported
   board by approximately 50 mm, hold it stationary, and press **B** again.
   Repeat the six contacts at this second height.
6. After 12 contacts the capture closes. **Q** exits early and retains partial
   records, but a partial session cannot pass calibration. Finish with the arm
   supported/holding as in ordinary collection; the script does not drop torque.

The board must remain fixed between B and the next six saves. If it moves,
start a fresh session. Use another new output folder when repeating a capture.
There is no automated contact detection: pressing S asserts that the same
physical fingertip midpoint is at the highlighted intersection.

## 3. Fit and evaluate once

In the same terminal:

```bash
.venv/bin/python -m real_robot.base_calibration fit \
  --session "$CALIB_SESSION" --max-error-mm 5
```

Only the eight fit contacts estimate camera pose and axial TCP correction. The
four check contacts do not affect the fitted parameters. TCP correction is
bounded to ±20 mm around the initial reported value. Insufficient orientation
diversity, height variation or numerical observability rejects the fit.

The default working criterion is **maximum 3D error <=5 mm** for every fit
contact, every reserved check contact and the board-surface RGB-D observations.
This is a project criterion, not a universal robotics accuracy standard. A low
pixel reprojection error alone cannot pass this physical check. If a test fails,
inspect the saved contact images, point residuals, estimated TCP and depth error;
do not discard inconvenient points or tune on the reserved checks. Diagnose the
cause and collect a fresh check set. Re-evaluation of the same capture is refused.

The result is saved to `base_validation.json`. On a pass, the tool additionally
writes:

- `rig-validated.json`: corrected camera-to-base transform, recomposed side
  transform, refined TCP offset and matching embedded validation evidence.
- `config-validated.yaml`: a new collection configuration referring to that rig.

The old rig and default configuration are retained. A validation flag alone is
insufficient: the live rig loader checks that the evidence matches the transform,
TCP definition and error limits. The result validates local camera/TCP alignment
in the sampled workspace, conditional on the robot FK model; it is not an
independent survey of global robot-base accuracy. Neither autonomous motion nor
object-centroid accuracy is certified. Visible-surface and assumed-orientation
quality flags remain relevant.

## 4. Continue collection after a pass

Add this option to the normal orange/marker collection command:

```bash
--config "$CALIB_SESSION/config-validated.yaml"
```

The launcher now takes the validated TCP offset from the rig, and the raw/export
records retain the base-validation evidence. Keep the planned independent train,
validation and test sessions. Inspect the physical validation result before a
large new collection block.

The older data need a **new offline version**, using the old and new transforms
and the unchanged-mount declaration. Object positions transform by rotation and
translation; velocities by rotation only. If the fitted TCP changes from 160 mm,
the old state **and action** XYZ labels also need their rotated TCP-offset change;
updating only camera/object fields would leave the geometry inconsistent. Use
the original raw/calibration provenance; do not merely flip old validation flags.

The current default rig is still provisional until this physical procedure
actually passes. Synthetic tests of this tool do not validate the real hardware.
