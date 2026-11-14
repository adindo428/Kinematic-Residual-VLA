# Ramp object velocity study

## Small orange: EfficientTAM mask for pick and place

With the orange on the table in the front-camera view, run:

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.ramp_velocity --object orange --mask-check
```

Draw a tight box around the whole orange and press Enter/Space inside the selection window. Inspect the contour and valid depth while gently moving/rotating it, then Q ends the preview. This saves the collection reference at `scratch/real_robot/object_profiles/orange/on_ramp`; the folder name selects the reference slot and does not require placing the orange on a ramp during selection. Repeat the same command to redraw the mask. Use `--object orange --preview-only` to reuse it without redrawing. Manipulator collection with `--task orange` loads this front-camera reference automatically and uses the instruction **Pick up the rolling orange and place it in the box.**

## Blue bottle: cylinder mask

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.ramp_velocity --object blue_bottle --mask-check
```

Draw a tight box around the whole bottle, including body, neck and cap, then Enter/Space confirms the selection. Inspect the contour as the cylinder rolls, then Q finishes the preview. Its profile is separate at `scratch/real_robot/object_profiles/blue_bottle/on_ramp`. For a camera-only velocity pilot, use `--object blue_bottle --runs 1`; metric speed requires valid depth within the mask. Pickup-only manipulator collection and the requested nine-motor target limits are documented in `README_DOM_COLLECTION.md`.

To redraw/reapply the blue-bottle mask, stop any running collection or camera preview first, place the bottle in view, then run this from the repository root:

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.ramp_velocity \
  --object blue_bottle --mask-context on_ramp --mask-check
```

`--mask-check` always requests a fresh selection and replaces the saved on-ramp reference when the new mask is accepted. Press Enter/Space inside the selection window, inspect the live contour, and Q finishes the preview. Subsequent `real_robot.collect --task blue_bottle ...` sessions automatically load this saved mask; restart a running collector to load the new reference.

To reapply the saved reference for a preview without redrawing it or recording runs:

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.ramp_velocity \
  --object blue_bottle --mask-context on_ramp --preview-only
```

## Marker: separate object mask

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.ramp_velocity --object marker --mask-check
```

Draw a tight box around the whole marker, including its cap, and press Enter or Space in the front-camera selection window. Inspect the contour along the ramp and rollout, then Q ends the preview. The reference is saved separately under `scratch/real_robot/object_profiles/marker/on_ramp`. To select a table reference instead, add `--mask-context off_ramp` and use that same context for profiling. Valid mask depth is required for metric speed. After checking the mask, run `--object marker --runs 1` for a one-run-per-ramp pilot, or `--runs 10` for repeated trials and across-run statistics.

## Lemon: mask and rollout velocity

Create a separate lemon reference using the fixed front RGB-D camera:

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.ramp_velocity --object lemon --mask-check
```

Place the lemon at the visible ramp start, draw a tight box around its entire outline, and press Enter or Space in the selection window. Gently rotate it and move it from the ramp onto the rollout table to inspect the contour and valid-depth readout. Q ends the preview. The reference is saved under `scratch/real_robot/object_profiles/lemon/on_ramp`; other object profiles are separate. Metric velocity uses depth inside the EfficientTAM mask and the existing calibrated 3D velocity filter. Missing or unreliable depth leaves speed unavailable. For an irregular rolling lemon, this measures a visible-surface proxy rather than a fixed center-of-mass point, so rotation can affect the estimate.

After checking the transition and depth, run a one-trial-per-ramp pilot:

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.ramp_velocity --object lemon --runs 1
```

Enter each ramp height when prompted, then reset the lemon and press Enter once to arm each trial. Wait for **Tracking stable. Push now.** Motion detection, stopping and saving are automatic; the lemon must stop within camera view. The speed trace covers the tracked motion from the ramp through rollout; it is not automatically restricted to frames after the ramp exit. Use `--runs 10` for ten accepted trials per ramp and across-run mean-speed standard deviation. To test just one ramp, add `--ramp ramp_1`. Results include `velocity.svg`, `runs.csv` and `summary.json` in the printed session folder.

If a table reference works better, select one with `--object lemon --mask-check --mask-context off_ramp`, then use `--mask-context off_ramp` during profiling. The two context references are separate and are not combined automatically.

## Renamed small ball: `yllow_baseball`

The requested object ID is `yllow_baseball`. Select a fresh mask with:

```bash
.venv/bin/python -m real_robot.ramp_velocity --object yllow_baseball --mask-check
```

`yllow_baseball` now defaults to a 25-pixel eroded EfficientTAM mask cutoff (other objects retain 80). Override with `--min-mask-pixels`; this only changes the mask-area gate. It uses a separate profile under `scratch/real_robot/object_profiles/yllow_baseball/on_ramp`; old green-ball references and sessions remain identifiable as originally recorded. The front preview shows mask area and valid-depth percentage. A correct silhouette can still lack stereo depth. The metric estimator requires at least 20 valid depth pixels and 50% valid coverage, with depth spread checks; these limits are retained. Do not infer reliable 3D velocity from a visible mask alone. To inspect a saved mask without redrawing it or recording runs, use `--object yllow_baseball --preview-only`. If the camera reports zero depth at a location, the speed tracker leaves that observation unavailable; it does not borrow table/background depth or fill it with a guessed range. A separately calibrated plane-based RGB measurement is an alternative for constrained ramp/table motion and requires known geometry and ball size. After finishing the mask check with Q, a one-run-per-ramp pilot is:

```bash
.venv/bin/python -m real_robot.ramp_velocity --object yllow_baseball --runs 1
```

## Green rubber ball: quick ramp pilot

```bash
.venv/bin/python -m real_robot.ramp_velocity --object green_rubber_ball --mask-check
```

Draw a tight box around the entire ball in the FRONT view and press Enter/Space. Check the contour while moving the ball gently on/off the ramp; Q finishes this preview. Its profile is saved independently under `scratch/real_robot/object_profiles/green_rubber_ball/on_ramp`. To create separate on/off-ramp references, use `--object green_rubber_ball --prepare-masks` instead.

Then run one accepted push per ramp:

```bash
.venv/bin/python -m real_robot.ramp_velocity --object green_rubber_ball --runs 1
```

Enter each ramp height when prompted. Reset the ball, press Enter once to arm, wait for **Tracking stable. Push now.**, then push. Start/end detection and saving are automatic. One run per ramp is a pilot; across-run sample standard deviation needs at least two runs per ramp. Use `--runs 10` for the original comparison protocol.

Default front-position speed studies now open **only the front RGB-D camera**; the side preview is marked DISABLED and is not substituted with a duplicate frame. Side-camera faults cannot block this mode. `--position-source fused` or `--refresh-side-mask` explicitly opens both cameras. Robot ports are never opened by a speed study.

## Orange and white striped ball: prepare masks first

```bash
.venv/bin/python -m real_robot.ramp_velocity --object striped_ball --prepare-masks
```

This guides two separate camera-only sessions: ball on the ramp, then ball off the ramp in the rollout area. At each terminal prompt, position the ball in the front view and press Enter. In the FRONT selection window, draw a tight box around the **entire ball**, including both colors, and press Enter or Space. Check that the green contour covers the ball and excludes the ramp/background; gently rotate and move it to expose different stripe orientations. Wait for the terminal's inspection message, then press Q in the camera preview to end the stage. The next stage opens after its positioning prompt. The side camera is shown as a raw preview and requires no selection or ball visibility.

References are separate under `scratch/real_robot/object_profiles/striped_ball/{on_ramp,off_ramp}`. These are manual references for the same ball, not a fine-tuned model; the two references are not automatically combined. Once both look correct, test transition from ramp to rollout in a preview seeded with the on-ramp reference before relying on velocity measurements. If its contour fails during that transition, stop and review the saved evidence before collecting a full block.

Then start velocity profiling:

```bash
.venv/bin/python -m real_robot.ramp_velocity --object striped_ball
```

This reuses the on-ramp reference, with 10 accepted runs per ramp and one Enter press to reset/arm each run. To use the other reference, add `--mask-context off_ramp`. To replace just one reference and inspect it again, use `--object striped_ball --mask-check --mask-context on_ramp` (or `off_ramp`). Each check saves its reference image, actual EfficientTAM mask and live preview/log evidence in a newly named session folder. Inspect the contour visually: completion of a mask session does not certify segmentation or depth accuracy. The ball workflow never imports the blue-tape profile.

From the repository, run:

```bash
.venv/bin/python -m real_robot.ramp_velocity
```

This opens the existing calibrated front and side cameras and saved blue-tape masks. It does not connect to or command the robot. The study now defaults to **front-camera RGB-D metric tracking** (`--position-source front`). The side camera is disabled by default for speed studies; only front RGB-D is opened. The ordinary collection workflow retains its fusion behavior. Use `--position-source fused` explicitly to compare the previous two-view method. Keep the tape visible in the front camera preview; select it in each view if prompted. Use the **terminal**. Enter each ramp's height when prompted. For each run:

1. Put the tape at its starting position and press Enter once to arm the episode.
2. Wait for **“Tracking stable. Push now.”** Then push the tape. The tracker detects motion and automatically ends and saves the run after the tape slows down.
3. Reset the tape and press Enter once to arm the next run. No keypress is needed after the push.

The program collects 10 accepted runs per ramp; ramp 1 and ramp 2 are assigned in the order you set them up. Press Ctrl+C to stop early; accepted results are saved after every run. Closing the camera preview with Q ends perception.

The detector requires a stable visible track for 0.5 s before arming; it starts after two consecutive accepted speeds at or above 0.04 m/s and ends at the first speed at or below 0.025 m/s once speed stays below that threshold for 0.4 s. The thresholds are provisional and may need adjustment after a physical pilot. Push only after the ready message and keep the tape visible in the selected metric camera(s) throughout the motion and the stop. It must come to rest within view: losing tracking during motion discards the run rather than treating the loss as a stop. Motion lasting more than 15 s is also discarded. A run needs at least three usable velocity estimates, 0.1 s of usable intervals, and 70% coverage of its automatically detected interval. Failed runs are repeated after one reset/arm keypress. A passing coverage check does not establish accuracy. Filter lag and speed thresholds can omit early and late motion, so short or slow pushes may not be detected reliably.

Side visibility is optional in the default front mode. For a study using `--position-source fused`, you can select a fresh side-view mask:

```bash
.venv/bin/python -m real_robot.ramp_velocity --position-source fused --refresh-side-mask
```

Put the tape at the ramp starting position where both cameras can see it. In the side selection window, draw a tight box around the tape and press Enter or Space; leave the tape still until tracking starts. The fresh side mask is saved only inside this study folder and does not replace the main collection profile. If no saved front mask is available, select the tape in the front window too. Check the side contour over the full path in the preview; reposition the camera or ramp if the tape is occluded or leaves its field of view. Moving a calibrated camera invalidates its calibration, so recalibrate before using the resulting 3D measurements if a camera is moved.

Results are in the printed `scratch/real_robot/ramp_velocity/<timestamp>/` folder:

- `velocity.svg`: per-run speed traces for the two ramps (open in a browser; hover a trace for its run number).
- `runs.csv`: each accepted run's time-weighted mean speed, tracked duration and coverage.
- `summary.json`: per-run samples and, for each ramp, the mean and **sample standard deviation across run mean speeds** in m/s.
- `notes.txt`: entered ramp heights. `perception/fusion.jsonl` and `perception.log` retain tracking diagnostics.

Speed is the magnitude of the existing causal, filtered 3D velocity estimate for the visible surface proxy, in m/s; it is not independently validated ground truth or tape center velocity. The mean integrates only adjacent accepted estimates within 0.25 s; missing intervals are not filled. The speed plot has gaps where tracking is unavailable. Estimates can be affected by surface appearance, camera geometry, processing delay and tracking source changes. Detection boundaries use host processing time and the integral uses camera timestamps; processing and filter delays affect the boundaries. No raw video is archived by this study.

For a short pilot use `--runs 1`. The camera session lasts up to 30 minutes; use `--seconds` to set a shorter limit. Each invocation creates a new output folder.
