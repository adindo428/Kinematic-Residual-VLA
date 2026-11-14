# Real-robot dataset quality and repair

**Depth provenance checked on 2026-11-09:** the current processed commit is
`439d8d9051079b0c4459f48256ac83f1760116cf` (564 episodes / 118,973 frames).
Both workstation fine-tuning runs used its local 80ms velocity-extrapolated
derivative. All 69,163 usable geometry rows carry the fitted RGB-referenced depth
correction; the 424 frames from episode 0 remain uncorrected and excluded.
The first 318 historical episodes were reprocessed offline; 245 newer
episodes were corrected during capture. Raw `main` retains the original 319
historical exports, so it differs from the processed `training` branch. Live
policy evaluation uses the same correction. The audit and numerical equation
check are in `scratch/real_robot/depth-provenance-audit/audit.json`.

**Current published training dataset:**
[ailab-Clemson/KRVLA-dataset, revision `training`](https://huggingface.co/datasets/ailab-Clemson/KRVLA-dataset/tree/training).
The first processed release is commit `dc25fe788826b513207439acd1a4e30388cfa67b`:
394 episodes / 86,470 frames, split **301 train / 50 validation / 43 test**,
with 51,085 usable geometry frames. The new 75 orange episodes were already
depth-corrected during capture; processing repairs 243 position and 1,995
velocity frames without applying the depth correction twice. New grasp/lift
intervals remain unconfirmed unless reviewed using wrist images and telemetry.

`main` preserves raw append-only exports. Use `revision="training"` for training,
or pin a training commit for reproducibility. The updated DynamicVLA reader
defaults to that branch for this repository, uses a separate cache directory,
and downloads splits, quality policy, camera acceptance, train statistics and
selected per-episode quality sidecars before checking the contract. Set
`euler_delta_convention="wrapped_xyz_v1"` and the matching policy convention.

Download the processed version on a training machine with:

```python
from huggingface_hub import snapshot_download
root = snapshot_download(
    "ailab-Clemson/KRVLA-dataset", repo_type="dataset", revision="training"
)
```

Future `real_robot.collect` invocations default to `real_robot/config_collection.yaml`,
which references the accepted transform and depth model under
`real_robot/calibration/`. KRVLA uploads reject a mismatched/uncorrected config
before robot startup. After raw upload, the workflow processes new episodes,
retains the corrected prefix, preserves explicit splits, recalculates train-only
statistics, publishes `training`, and verifies remote hashes and video pointers.
The upload CLI retry path performs the same refresh. If raw upload succeeds but
processing/publication fails, the command reports failure and gives this retry:

```bash
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN \
.venv/bin/python -m real_robot.training_release --repo ailab-Clemson/KRVLA-dataset
```

The latest local release is recorded in
`scratch/real_robot/training_releases/latest.json`. Initial release directory:
`scratch/real_robot/training_releases/KRVLA-41d24d3aeb12`.
Splits remain provisional while collection continues. Once a task has dedicated
train, validation and test collection sessions, the release uses those explicit
assignments and retires its older provisional holdouts. Pin the final release
before model selection. Operator acceptance and depth extrapolation remain
explicit; publishing does not claim independent calibration validation.

Use a combined approach: retain causal measurements and quality/phase evidence
during collection, then repair bounded gaps offline. Do not replace unavailable
live velocity with an interpolated value that uses future frames.

The existing orange collection command still works. The launcher reports the
operator-accepted existing base transform and writes `quality-audit/quality_audit.json` beside
each new DOM export. Rerun terminal and global keyboard modes now pass through
`L` (lift) and `R` (release), plus the existing phase markers. Previously these
keys were filtered out by the Rerun keyboard adapter. Markers help video review;
they do not by themselves prove that an object was grasped. Robot motion and
leader/follower limits are unchanged.

## Findings from the local data

The 294-episode copy has 64,630 frames. The audit reproduced 3,081 frames with
an action/state Euler difference exceeding pi. Among frames with an open
commanded gripper, 8,037 have stale position and 17,125 have missing velocity.
These are telemetry categories, not ground-truth manipulation phases. The
missing tracking cannot all be attributed to carrying an already grasped object.

Wrist/front review sheets exist for all 294 episodes and the additional 25
orange episodes:

- `scratch/real_robot/grasp-review-294-v1/index.html`
- `scratch/real_robot/grasp-review-orange-25-v1/index.html`

Candidate lift detection requires a previously open gripper, at least 0.20 s
of measured closure (normalized measured gripper <= 0), and TCP rise >= 0.03 m.
This is only a way to find useful video frames. The midpoint closure threshold
is conservative and can miss large-object grasps; absence of a candidate is
not evidence of failed grasping. Every episode also includes its worst stale
interval in the review sheet.

Confirm holding only when wrist images show object retention, measured gripper
closure is consistent with that object, and the EE lifts concurrently. Episode
50 includes an initial failed acquisition followed by a successful grasp: a
closure/lift heuristic alone would mislabel the first attempt. The reviewed
orange interval in episode 295 visibly retains the orange during a TCP rise
from about 0.050 to 0.198 m, with measured gripper 47.43–47.58 percent.

Only conservative intervals in episodes 50, 210, 260, 290 and 295 have been
confirmed so far, totaling 163 frames. The rest of the gallery is **unreviewed**;
no other tracking gaps have been automatically excused as post-grasp. Annotation
files contain per-interval review notes and exclusive end frames:

- `scratch/real_robot/grasp-reviewed-294-v1.json`
- `scratch/real_robot/grasp-reviewed-319-v1.json`

## Local processed versions

The earlier `scratch/real_robot/KRVLA-quality-319-v3-depth` is the corrected
baseline incorporated into the published release above. It contains 319 episodes / 70,059 frames, with 69,635
object positions corrected using the RGB-referenced depth model and the accepted
existing camera-to-base transform. Velocities were recomputed by replaying the
causal filter over complete saved perception logs at native timestamps, followed
by the same bounded gap repair and reviewed phase handling. Segmentation and
the recorded representative rays were retained. Robot states/actions, assumed
object orientations, timestamps, splits and RGB videos were preserved.

There are 41,793 individually usable geometry frames; observation-history
requirements can reduce the number of training anchors. Episode 0's raw
tracking folder is missing: its 424 rows are retained, but excluded from geometry
anchors/statistics. Episodes 247 and 269 already had no usable geometry anchors.
No episode was deleted. The 930 repaired-position frames and 9,364 repaired-
velocity frames retain explicit imputation flags. Per-episode environment
statistics use usable corrected rows; aggregate normalization uses training
episodes only. Other feature statistics retain their original populations.

The measured board depth support is approximately 0.799–1.033 m, narrower than
the manipulation workspace. The collection model in
`scratch/real_robot/depth-calibration-rgb-reference-20261107/depth_correction_collection.json`
explicitly applies the fitted function across the existing 0.1–1.5 m front
tracking range, with `depth_correction_extrapolated` flags outside measured
support (30,960 rows in v3, including held samples). These extrapolated estimates
are included under the empirical calibration assumption; independent accuracy
outside the board range is not established. They are not missing or stationary
measurements. The original fit-only model is retained separately.

The output includes `depth_reprocessing_report.json`, `quality_audit.json`, and
`dataset_validation.json`. Row/schema, split coverage and video-header checks
passed. Videos are reused by symlink. Raw recordings and Hub files were not
rewritten. Reproduce the update into a new directory with:

```bash
.venv/bin/python -m real_robot.recalibrate_dataset \
  --dataset scratch/real_robot/KRVLA-quality-319-v2 \
  --model scratch/real_robot/depth-calibration-rgb-reference-20261107/depth_correction_collection.json \
  --output scratch/real_robot/KRVLA-quality-319-NEW
```

After the new 75 accepted episodes, **100 orange episodes remain**: collect
60 train, 20 validation and 20 test in separate sessions. The corrected config
is now the collection default:

```bash
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN DISPLAY=:1 \
.venv/bin/python -u -m real_robot.collect \
  --task orange --episodes 60 --split train --viewer rerun \
  --default-success \
  --robot-max-relative-target '[20,20,20,20,20,20,20,20,35]' \
  --hub-repo ailab-Clemson/KRVLA-dataset --public
```

For subsequent invocations change only `--episodes` and `--split` to the next
block. Counts are operator-success attempts; the exporter can reject attempts
that lack synchronized streams, so check accepted export counts after each run.

Raw sessions, original exports and the published Hub dataset are retained.
The following earlier local versions contain repaired parquet, quality sidecars,
explicit splits, episode statistics and train-only aggregate statistics:

| Version | Episodes / frames | Position / velocity frames repaired | Train / validation / test |
| --- | --- | --- | --- |
| `scratch/real_robot/KRVLA-quality-294-v2` | 294 / 64,630 | 855 / 8,752 | 206 / 48 / 40 |
| `scratch/real_robot/KRVLA-quality-319-v2` | 319 / 70,059 | 931 / 9,451 | 226 / 50 / 43 |

Videos are local symlinks to the source export, without re-encoding. These are
local working versions, not portable upload bundles. Do not append processed
episodes to the existing repository: its append contract preserves old episode
files byte for byte. Publishing a processed version needs a separate versioning
decision and materialized videos.

Repairs use unique accepted position measurements and their acquisition times,
deduplicating held samples. A gap must be bracketed within 0.20 s, imply speed
<= 2 m/s, and avoid gripper transitions and confirmed held-phase boundaries.
There is no extrapolation. Finite-difference velocity is an estimate, not a new
measurement. Original stale/zero-fill flags remain intact; separate repair
flags identify imputed values and use of future evidence. Multi-second gaps
remain unresolved. Every original row, timestamp and action label is retained.

After confirmed lift, object velocity remains physically nonzero while carried.
The processed model input uses a neutral zero with `velocity_required=false`;
it is explicitly labeled as neutral input, not measured stationary velocity.
KRV's entire geometric correction, including mean-centering, is gated off for
these anchors. Unknown pre-grasp velocity remains invalid. This preserves good
action demonstrations without feeding missing tracking into the dynamics path.

The 319-episode version has 42,171 individually usable tracking frames. A loader
requiring multiple observation times can select fewer anchors because every
requested history frame must be usable. Filtering anchors never compresses the
physical time or skips rows inside an action chunk. Future environment fields
can still contain invalid tracking and are not suitable as unmasked auxiliary
trajectory supervision.

## Splits and their limits

The current splits are provisional while collection continues.
The remaining targets are **100 orange and 100 green-marker accepted episodes**.
The prospective allocation is saved in
[`remaining_collection_plan.json`](remaining_collection_plan.json):

| Object | Remaining train | Remaining validation | Remaining test | Total remaining |
| --- | --- | --- | --- | --- |
| Orange | 60 | 20 | 20 | 100 |
| Green marker (`--task marker`) | 60 | 20 | 20 | 100 |

Collect each validation/test block in its own invocation, setting `--split`
explicitly. Orange training can be one session of 60; marker training
can be one session of 60. Continue using the existing collection options and
relative-target limits, changing `--task`, `--episodes` and `--split` per block.
Targets count accepted exports, so alignment rejection may require more attempts.

Once these new sessions are available, the proposed final export puts all
existing 25 orange and 92 marker episodes into training and replaces their
provisional held-outs with the new dedicated sessions. Preserve the existing
striped-ball session assignments. With the 319-episode baseline, the projected
final dataset has **594 episodes: 464 train / 69 validation / 61 test**. This
removes the orange/marker session overlap described below. Future episode IDs
remain unassigned until export; no current dataset split files were changed by
this prospective plan. Freeze the final assignments before model selection.

Striped-ball splits use disjoint recording sessions. Marker has only two
sessions: the 54-episode session is training and the other session is divided
into contiguous validation/test blocks. Thus marker validation and test share
a session, although neither shares marker training's session. Orange has only
one session: its 25 episodes are provisionally split 20/2/3 chronologically.
Orange train/validation/test share that session. These limitations are recorded
in `quality_audit.json` under `shared_split_sessions`; do not describe these
results as fully session-independent.

For stronger evaluation, collect separate orange validation and test sessions
using the existing command with `--split validation` and `--split test`, and
collect a third marker session. Freeze split assignment before model selection.
The CLI can produce strictly disjoint session splits once each task has at
least three sessions; it fails rather than silently falling back when that is
impossible.

## Accepted existing camera-to-base transform

The user explicitly chose to reuse the existing October 19 camera-to-right-base
transform. Its composed front/side rig is
`scratch/real_robot/front-side-calibration-20261020-01/rig.json`.
All 319 episode records (and all 294 in the earlier subset) contain exactly the
same front/base, side/base and front/side matrices. Acceptance is recorded in
[`base_transform_acceptance.json`](base_transform_acceptance.json), with the
source, transform fingerprint, user's instruction, and earlier verification
results. The processed 294/319 v2 quality policies now permit this transform
for training as `operator_accepted_existing_transform`. No new physical
calibration is required to proceed under that explicit decision.

The depth-corrected config at
`scratch/real_robot/depth-calibration-rgb-reference-20261107/config-depth-corrected.yaml`
uses this same accepted transform. Supply it via `--config` for collection using
the new depth correction. The v3-depth version described above includes offline
depth/velocity reprocessing; the earlier v2 labels retain their recorded depth.

The nominal conventions are consistent in code: robot state/actions come from
right-base FK; the tracker transforms camera positions into the candidate base
frame *before* estimating velocity. This does not establish physical accuracy.
All fixed-camera records still have `base_calibration_validated=false`.

Existing evidence at
`scratch/real_robot/right-handeye/session-20261019-04-verification/fixed-candidate-verification.json`
reports 7.24 mm mean and 8.72 mm maximum position residual, failure of its
proposed 5 mm threshold, and `absolute_base_frame_accuracy_measured=false`.
The original validation flags remain false. Operator acceptance is stored
separately and bound to the exact dataset camera metadata; a changed dataset or
different transform cannot silently inherit this acceptance.

The [physical base-calibration procedure](README_BASE_CALIBRATION.md) remains
available if independent accuracy validation is desired later. It is not a
requirement imposed on this operator-accepted training run.

If the rig was unchanged and independent evidence establishes a corrected
transform, old object fields can be corrected offline:
`C = T_new_base_front @ inverse(T_old_base_front)`,
`p_new = C.R @ p_old + C.t`, `v_new = C.R @ v_old`.
Never translate a velocity or transform already-base-frame robot state twice.
Recompute statistics and maintain per-session calibration provenance. Camera
movement or depth/surface bias can require raw RGB-D reprocessing. Assumed
object orientation remains unmeasured after such a correction.

## Training and deployment contract

The supported reader/policy changes are in `third_party/DynamicVLA`. The archived
`experiments/tracking75/source` snapshot remains frozen. Prepare a new training
family/checkpoint version for the changed source and labels; do not resume an
old frozen family with changed targets.

- `LeRobotDataset` consumes explicit local `meta/splits.json`, rejects empty
  requested splits, and requires processed tracking quality for these real data.
- Velocity-conditioned loading accepts either validated base calibration or
  explicit operator acceptance bound to the dataset's camera metadata. The
  processed 294/319 v2 versions use the latter; their calibration blocker is
  removed. Tracking quality, phase and Euler-convention requirements still apply.
- Usable observation-history anchors come from `meta/quality_policy.json` and
  per-frame quality sidecars. Missing pre-grasp tracking is never stationary.
- Set `POLICY.EULER_DELTA_CONVENTION = "wrapped_xyz_v1"` in generic training;
  direct readers take `euler_delta_convention="wrapped_xyz_v1"`. New frozen
  family manifests use `euler_delta_convention` with the same value. The loader
  checks the dataset convention and the checkpoint stores the policy convention.
- Training wraps componentwise XYZ Euler action-minus-current-state differences
  into [-pi, pi). Streaming and queued deployment reconstruct rotation-equivalent
  absolute angles and canonicalize X/Z to [0, 2pi). This fixes Euler boundary
  crossings; it is not a switch to SO(3) tangent-vector targets.
- Real-data policy construction enables `require_tracking_quality`. Live callers
  must provide `dynamic_object.velocity_valid` and
  `dynamic_object.velocity_required` as `[batch, observation_history]` or `[batch]`.
  A false `velocity_required` must mean positively confirmed holding using wrist
  evidence and measured telemetry, never tracking loss. Provide a finite neutral
  vector for confirmed holding. Current/past observations drive live confirmation;
  future-based offline repair is not a deployable estimator.
- Invalid required velocity raises an error. Controllers must reacquire/pause
  rather than sending a fabricated velocity into the policy. Automatic live
  wrist grasp confirmation has not been implemented by this change.
- Generic policy initialization uses the processed version's train-only stats.
  Frozen family velocity normalization must likewise be recomputed from its new
  training data when preparing that family.

## Re-run the offline work

```bash
.venv/bin/python -m real_robot.dataset_quality \
  --dataset PATH_TO_ORIGINAL_DOM_EXPORT \
  --output NEW_LOCAL_OUTPUT --repair \
  --split-file EXPLICIT_SPLITS.json \
  --phase-annotations REVIEWED_INTERVALS.json

.venv/bin/python -m real_robot.dataset_quality \
  --dataset PATH_TO_ORIGINAL_DOM_EXPORT \
  --output NEW_REVIEW_DIRECTORY --prepare-review
```

Use `--make-session-splits` instead of `--split-file` only when each task has
enough independent recording sessions. A new output path is required so earlier
versions cannot be overwritten accidentally.

## November 8 marker tracking recovery

`real_robot.recover_marker_session` is an offline detector for the green marker
and fixed front-camera view in `marker-train-20261108T233827.476765Z`. It reads the
session's `resume_stream_audit.json` and selects only successful attempts with
no live object positions, excluding the 11 previously exported attempts.
It is not a replacement for the general live tracker or a detector for other
objects/camera placements.

The source recording remains unchanged. The derived recording hard-links image
archives, copies robot logs, and writes new observation measurements. Do not edit
the shared image archives. The saved depth is already corrected and is never
corrected a second time. Green cap/barrel segmentation uses the reviewed image
region; ambiguous detections are missing. Sparse depth requires at least 40
measured pixels, 15% coverage and median absolute deviation no greater than
10 mm. Depth holes are not filled with table depth. Position jumps are gated;
velocity uses the causal filter restarted independently for each attempt.

`recovery_report.json` records source/code hashes and per-attempt counts. Five
mask overlays per attempt are saved under `review/`. Export retains recovery
provenance in `meta/sources.jsonl` and coverage flags in quality sidecars. This
recovery does not confirm grasp phases or establish independent calibration
accuracy. Use the processed `training` branch's quality masks; recovery does not
make every frame's geometry usable.

## Workstation batch profile and bounded velocity extrapolation

The RTX 4090 profile selects microbatch 16, gradient accumulation 1 (effective
batch 16), BF16 autocast, two loader workers and prefetch factor 2. Shuffled
real-data profiles include loading, forward/backward and optimizer updates.
KRV100 completed epoch 3 and DOMV completed epoch 6 both reached about 54
samples/second. Batch 20 was slower and reserved about 20.7 GiB; batch 24 ran
out of memory. Four workers did not improve throughput materially over two.
Results are in `scratch/real_robot/batch-profiling/summary.json`; these are
short profiles, not full-epoch timing or completed fine-tuning runs.

`real_robot.velocity_extrapolation` creates a separate local dataset version.
The selected 80 ms maximum age is measured from the last actual measurement,
not from a held export row. It requires either three stable original velocity
measurements or five original positions spanning 120–300 ms with a tight
linear-fit residual. Predictions never seed other predictions, and the method
does not cross phase/gripper transitions or replace confirmed-held neutral
inputs. It leaves object positions and all robot/action/timing fields intact.
Missing positions are not made usable merely by filling velocity.

`velocity_extrapolated` and per-frame source/age/fit provenance distinguish the
new estimates; original missing-data flags remain. The existing loader consumes
these through the explicit `velocity_repaired` validity flag. Train statistics
and usable-anchor lists are recomputed. Backtesting compares withheld tracker
values, not independent physical ground truth; horizon selection uses training
results only. The final pass added eight values beyond existing interpolation,
leaving unsupported gaps masked. The local version is
`scratch/real_robot/training_releases/KRVLA-5fd5ef3e4256-extrapolated-80ms-v2`;
it has not replaced the published Hugging Face revision.

### Sequential workstation fine-tuning

`workstation/train_real_finetune.py` runs KRV100 followed by DOMV in separate
processes. The launch snapshot in `scratch/real_robot/finetuning-runs/` pins the
local audited velocity-extrapolated derivative and completed initialization
checkpoints. Both runs retain FastViT, use train-only normalization, wrapped
Euler deltas, batch 16, BF16, and AdamW at 1e-5. Validation uses fixed flow noise;
the test split is not loaded. Training stops at 30 epochs or after 7 epochs
without validation improvement. Best and latest checkpoint pointers are JSON
files inside each run directory. Epoch checkpoints include weights, config,
optimizer state and RNG state. Periodic recovery snapshots are not completed
epochs, and exact mid-epoch data replay is not implemented by this launcher.

W&B logs to `aparame-clemson-university/Kinematic-Residual-VLA`. The queue starts
DOMV only after successful completion of KRV100; failure stops the queue.
`finetuning-runs/latest.json` identifies the supervisor PID and log path.
LAAS/continuous streaming is a deployment requirement; gradient training keeps
streaming disabled. No robot hardware is opened by the training launcher.

### Outcome-supported grasp annotations

`real_robot.grasp_annotations` builds a separate local derivative using the
operator's declaration that all saved demonstrations succeeded. Of 564 source
episodes, 563 retain a saved `success` outcome; the first episode's original
metadata/robot log is unavailable. Success establishes that a pickup occurred,
not the timestamp of pickup or that every earlier close/lift succeeded.

The offline pass requires an earlier open state, concurrent measured and commanded
closure, sustained measured EE lift of at least 20 mm, and object appearance
retained relative to the wrist/fingertips. Normalized gripper aperture up to 0.35
is allowed: a ball/orange obstructs the jaws before they reach empty fully-closed.
Marker white lettering is handled as fragments of one colored barrel, not as
separate objects. Reopening ends a held interval. Missing measured telemetry,
unseen objects, relative object motion or ambiguous appearance remain unresolved.

- Automatic concurrence is `held_success_inferred`, with `phase_confirmed=false`
  and explicit method/provenance. It is not presented as human-reviewed ground truth.
- Visually inspected sampled wrist strips, with measured closure/lift, are recorded
  separately as `held_confirmed`. Previous reviewed annotations are preserved.
- Both accepted kinds set `velocity_required=false` and use a neutral velocity
  input. The original stale/zero-filled/repair flags remain recorded; this does
  not claim the physical object is stationary. Object positions are unchanged.
- Marker episodes carry `task_table_contact_expected=true`, reflecting the
  operator's task-level explanation. This is not a per-frame force/contact label.
  Low or negative recorded TCP heights and their expert actions are preserved.

Review artifacts are under `scratch/real_robot/grasp-annotations-success-v2/`.
The derivative is
`scratch/real_robot/training_releases/KRVLA-5fd5ef3e4256-extrapolated-80ms-grasp-v1`.
Read its `grasp_annotation_report.json`, `meta/grasp_annotations.json` and
`meta/quality_policy.json` for counts and provenance. Videos are reused, splits
and expert actions are unchanged, and normalization is recomputed from training
episodes only. These local annotations do not automatically update Hugging Face.

The live controller still requires **H plus operator wrist confirmation**; it
does not assume future episode success. The shared aperture/lift support bounds
now allow a grasped ball/orange to keep the jaws apart. This offline annotation
pass itself does not change the controller. Evaluation now separately removes
the XYZ workspace box and uses a grasp geometry profile permitting fingertip/table
contact; other link/camera table checks and joint/self-collision checks remain.
See `README_POLICY_EVAL.md` for the current settings and limitations.

The next batch uses the common-original-DOM initialization and revised recipe in
`README_ROBUST_FINETUNING.md`; the preceding sequential-run section describes the
earlier simulation-initialized experiment.
