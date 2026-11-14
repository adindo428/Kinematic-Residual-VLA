# Original pretrained SmolVLA: static orange

Run directory:
`scratch/real_robot/finetuning-runs/static-orange-smolvla-pretrained-20261113`.

Initialization is exclusively the original official `lerobot/smolvla_base`, pinned
to revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` and verified against
SHA256 `7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`.
The fresh upstream loader strictly loads those weights directly. It bypasses
the DOM checkpoint loader and rejects continuation checkpoints. No DOM-adapted
or prior real-robot weights initialize this run. Shared source code and local
configuration/tokenizer assets do not load learned DOM weights.

The dataset is the exact static-orange derivative used by the corrected KRV run:
80 train / 10 validation / 10 test episodes, 12,564 eligible training anchors and
1,110 validation anchors. Only training episodes enter optimization and affine
statistics. The lossless image cache is reused without re-encoding. Test episodes
are excluded from optimization and checkpoint selection.

The prompt is fixed verbatim in training, validation, and live family evaluation:
`Pick up the rolling orange and place it in the box.`

Native SmolVLA uses one current image per front/wrist camera and current six-value
EE state. It has no neural object-velocity input, KRV drift or training-time velocity
innovation. Continuous seven-channel Cartesian delta targets are converted back
to physical absolute poses with the same wrapped XYZ Euler convention as KRV.
The standard native trainable set is retained: action expert and state projection,
99,880,992 trainable parameters out of 450,046,176 total; vision/language weights
are frozen. This is not an all-parameter fine-tune.

Training settings: effective batch 16; benchmark-selected flow microbatch 16,
auxiliary microbatch 2, bf16, fused AdamW, uniform permutation of training anchors,
no augmentation or weight decay, learning rate 1e-5 with 100-update warmup and
plateau reduction to a 1e-6 floor. Flow loss and the existing physical-action
auxiliary loss are retained. Raw and EMA weights are validated; the better
combined action score determines checkpoint selection. Cap: 100 epochs; patience:
20 completed epochs without improvement.

The disposable benchmark measured about 75.6 samples/s at microbatch 16, versus
70.6 at microbatch 8. Native/gradient sampler parity was exact and optimizer
and architecture checks passed. Profiling updates are discarded; actual training
reloads fresh original upstream weights.

Each epoch writes held-out raw/EMA action metrics and a separate 64-anchor raw
training-action probe. Training reconstruction errors therefore do not need to
be inferred from flow-matching loss. The fixed validation indices and noise seed
match the KRV static-orange experiment, while native model inputs and normalization
conventions remain family-specific.

Monitor:

```bash
tail -f scratch/real_robot/finetuning-runs/static-orange-smolvla-pretrained-20261113/training.log
cat scratch/real_robot/finetuning-runs/static-orange-smolvla-pretrained-20261113/runs/smolvla_baseline/status.json
```

Provenance and checks: `upstream_source.json`, `campaign.json`, `inspect.json`,
`benchmark.json`, `smolvla_baseline-gpu-readiness.json`, `loader_check.json`.

## Best and last closed-loop evaluation after completion

The following wrapper requires completed training and final saved-data evaluation,
or a verified user-requested stop with a frozen `evaluation_release.json`.
The stopped-run release explicitly records that the final before/after evaluation
was skipped; it does not mark training as normally completed. It checks selection and freezes the selected file
SHA256 before opening any device. `best` restores the selected raw/EMA weights
from `best.pt`; `last` uses raw weights from the final completed `latest.pt`.
Partial recovery snapshots are rejected. Best and last can be the same epoch;
the wrapper prints the actual epoch and weight selection.

From the repository root, check readiness without accessing devices:

```bash
.venv/bin/python workstation/smolvla_orange_eval.py best --check
.venv/bin/python workstation/smolvla_orange_eval.py last --check
```

Run one session at a time after training finishes:

```bash
.venv/bin/python workstation/smolvla_orange_eval.py best --execute
.venv/bin/python workstation/smolvla_orange_eval.py last --execute
```

Each command selects a fresh orange mask, records inputs/action chunks, and
requests one 30-second trial. It starts paused; SPACE at READY begins execution.
Omit `--execute` for a preview session. `--max-trials 3` requests three trials.
For a matched best/last comparison, pass the first session's
`controller/start_pose.json` to the second command with `--start-pose PATH`.
Use the same stationary orange placement. The exact training prompt and native
single-frame model inputs are preserved. Velocity innovation defaults to off for
this plain SmolVLA comparison. Existing motion, freshness, reset and geometry
checks remain active. No evaluation process is launched by this setup step.

The best/last commands now inherit the shared two-second occlusion profile.
For this static-only checkpoint, `--object-motion auto` selects the prior
contact-stop hypothesis. For a moving-object trial, pass `--object-motion moving`
to use constant-velocity position prediction instead. Existing travel and
uncertainty bounds still apply; moving predictions may expire well before 2s.
Restart a current session to use the new defaults. See README_POLICY_EVAL.md for
source logging and the distinction between predicted state and visible RGB.

## Automatic mask initialization and trial controls

The SmolVLA wrapper now defaults to `--mask-mode auto`. It matches the saved
camera-specific reference object's hue/saturation appearance on the fresh live
frame, then supplies the unique plausible box to EfficientTAM for segmentation
and tracking. This is a reference-colour proposal, not a general semantic object
detector. Modest colour changes are tolerated. Missing/ambiguous candidates or
failed segmentation fall back to the existing manual box selection. A first-time
object without a saved reference still needs manual selection. Automatic proposals
are used for initialization/reselection, not silent target switching in an active
trial. New masks and their selection provenance are saved; previous profiles are
backed up. Object-state geometry/freshness checks still gate READY.

For three trials, each with a 30-second timeout:

```bash
.venv/bin/python workstation/smolvla_orange_eval.py best --execute \
  --mask-mode auto --max-trials 3 --trial-seconds 30
.venv/bin/python workstation/smolvla_orange_eval.py last --execute \
  --mask-mode auto --max-trials 3 --trial-seconds 30
```

Run one command at a time. SPACE at READY still starts each trial. Timeout counts
as a failed trial and uses the existing guarded return. `--mask-mode manual`
forces box selection; `--mask-mode reuse` retains the old saved-mask behavior.
`--max-trials 0` removes the trial-count cap within the evaluator's session time
limit. The per-trial timeout defaults to 30s and is independent of session length.
Direct `real_robot.evaluate` also accepts `--auto-mask`, `--max-trials`, and
`--trial-seconds`. Restart existing sessions to apply these changes.

## RGB/state-only execution for the plain SmolVLA baseline

The plain `smolvla_baseline` checkpoint has neither analytical KR nor explicit
neural object-velocity conditioning. Its evaluator now uses front/wrist RGB and
measured robot state only, matching the actual trained neural inputs. It does
not require an EfficientTAM mask, tracked XYZ, velocity, or an occlusion-prediction
timer to keep executing. An occluded object no longer causes a tracker-loss hold
in this path. RGB freshness, wrist freshness, robot health, fixed start, action
expiry, geometry and the trial timeout remain enforced.

The existing best/last commands automatically select this path. Mask settings
are unused for this baseline; the wrapper omits mask flags and prints
`object_tracking_required: false`. Native family baselines without velocity
conditioning behave the same way. KR variants and explicitly velocity-conditioned
policies retain required object estimates and bounded occlusion prediction.
No object measurements are fabricated for the plain baseline, and archived
requests record `object_inputs_used_by_policy=false`. The unused auxiliary
velocity fields are omitted before the native plain sampler, which does not
read them. Model weights, instruction, image preprocessing, robot state and
physical action decoding are unchanged. This does not reconstruct the occluded
object's appearance in the camera image. Restart a running session to apply it.

## Guided RTC and the existing action controls

The best/last launcher now enables **guided real-time chunking (RTC)** by default.
Inference remains in its own GPU process while the 25 Hz controller continues
consuming the time-aligned action queue. This separate-process asynchronous path
already existed; RTC adds guidance toward the accepted pending chunk instead of
replacing overlapping predictions without guidance. It uses the training-free
soft-prefix formulation described in [LeRobot RTC](https://huggingface.co/docs/lerobot/rtc)
and the [RTC paper](https://www.pi.website/download/real_time_chunking.pdf).
No RTC-specific checkpoint or retraining is required for guided inference.

The controller snapshots contiguous, unconsumed actions on the current observation
clock. Each next inference waits for an observation acknowledging the preceding
result, so it guides against the plan actually accepted by the controller. Old
trial tokens, consumed timesteps, gaps and source-age-expired targets cannot enter
the prefix. Actions are reanchored to the new robot state, Euler differences are
wrapped, any analytical KR contribution is removed, and the result is normalized
with the native training affine. Only the seven physical output dimensions are
guided; padded channels remain unconstrained. The native ten denoising steps and
checkpoint parameters are unchanged. There is no hard inpainting assumption.

Delay is estimated conservatively from the rolling 95th percentile of observed
observation-to-result time, plus one 40 ms control tick. The default guidance
execution horizon is ten steps (400 ms), with maximum guidance weight ten.
Single-frame models publish their current observation immediately instead of
waiting for the two-frame history used by native DOM/KRV models. Chunk indices
remain on the original 25 Hz clock; elapsed actions are discarded.

Run one command at a time, then SPACE at READY:

```bash
.venv/bin/python workstation/smolvla_orange_eval.py best --execute --max-trials 3 --trial-seconds 30
.venv/bin/python workstation/smolvla_orange_eval.py last --execute --max-trials 3 --trial-seconds 30
```

Append `--rtc off` for the native-sampler comparison. Direct family evaluation
accepts `--rtc guided --rtc-execution-horizon 10 --rtc-max-guidance-weight 10`.
The guided adapter supports `smolvla_baseline` and `smolvla_kr`; other model
families retain their existing asynchronous evaluation and reject this RTC flag.
Plain SmolVLA still requires only RGB/state. Prompt, weights and input
normalization remain matched to training. Restart evaluation to load these changes.

**Existing controls found in the latest session, retained by this change:**

| Stage | Actual behavior |
| --- | --- |
| Cartesian target | 80 ms quintic pose blend |
| Differential IK | Bounded weighted DLS; damping 0.01 plus singularity term; orientation weight 0.1 m |
| Joint tracking | 6 degree maximum goal-to-measured following error; calibrated joint bounds |
| Joint trajectory | Ruckig acceleration 120 degrees/s², jerk 600 degrees/s³ |
| Velocity in hardware mode | Motor read-back joint ceilings (about 180 degrees/s); no additional TCP/angular ceiling |
| Workspace | No additional Cartesian workspace box; joint/path geometry checks remain |
| Gripper | Prediction clipped to [-1, 1], calibrated endpoint conversion, firmware-derived rate bound |
| Timing | Camera/state freshness, action expiry and control deadline checks can cause holds |

These stages alter the executed motion relative to raw model predictions. In the
latest saved session the median pose blend lag was about 2.46 mm, and one logged
Ruckig replan had a 0.395 s duration (that duration is not an additional fixed
sleep). Control lag and repeated freshness holds can contribute to slow behavior;
RTC addresses prediction continuity and inference delay, not all control lag.
This update does not alter differential IK, motor limits, blend or governor values.

Device-free GPU checks on saved observations passed for both frozen snapshots:
best epoch 14 and last completed epoch 26. Empty-prefix RTC exactly matched native
sampling; guided chunks were finite 20-by-7 outputs, with ten guided denoise steps
and frozen parameters. Warmed guided latency was approximately 196 ms / 190 ms,
respectively, within the 800 ms action horizon. Reports are in the run's
`evaluation_checks/best-rtc.json` and `last-rtc.json`. These checks do not establish
physical rollout success. `workstation/audit_rtc.py` reproduces the saved-input
check without opening cameras or motors.
