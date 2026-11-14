# Recorded-action replay without a VLA

The verified batch is `scratch/real_robot/expert-replay-five-marker-clear-v1`.
Its actions come from Hugging Face `ailab-Clemson/KRVLA-dataset`, revision
`439d8d9051079b0c4459f48256ac83f1760116cf`, episodes 473, 502, 519, 541, 563.
Each complete command/measured path passed the existing grasp geometry check;
selection did not use IK error. `geometry_selection.json` records eligible and
rejected candidates. Actions, coordinates and original 25Hz timing are unchanged.

Local original joint records are supplemental start references and verification
data. Every supplied joint/FK row was matched against the downloaded HF action
and state. During replay, joint commands are computed from HF Cartesian actions
by DLS; recorded joint commands are not sent as the replay trajectory.

Run with the operator supervising and the box/loose objects clear:

```bash
DISPLAY=:1 .venv/bin/python -u -m real_robot.replay_actions run \
  --batch scratch/real_robot/expert-replay-five-marker-clear-v1 \
  --output scratch/real_robot/replays/NEW_OUTPUT \
  --pose-smoothing-ms 0 --governor off --execute
```

SPACE first requests a guarded joint return to that episode's recorded start.
At READY, SPACE starts its recorded action clock. After completion, SPACE
prepares the next episode. P stops the episode; Q holds/quits. Hardware velocity,
joint-position/following and grasp geometry checks remain. No torque, gain or
firmware-limit settings are changed. No VLA, cameras or object tracking are used.

The start gate requires 10mm TCP, 2 degrees per joint, 3 degrees total physical
rotation and 0.4s quiet measured feedback. The 3-degree replay rotation bound
prevents a recorded transient 1.85–2.05 degree orientation offset from repeatedly
crossing a 2-degree gate. Actual initial offsets are retained in each episode log;
targets are never recentered. Live VLA evaluation retains its own start settings.

Resume completed episodes from a stopped run using `--resume-from OLD_OUTPUT`.
Completed logs are copied intact, the first unfinished episode is selected, and
the original session-home reference is preserved. Changing the input contract or
controller mode during a resume is rejected.

`audit` performs four device-free ideal-plant comparisons: direct DLS,
pose blend only, joint governor only and both filters. Those traces are
simulations, not live actual EE measurements. Physical execution writes target,
filtered target, IK result, final command FK, measured FK, timing and currents.
Measured feedback precedes that tick's new command; following error is expected.

The completed physical run is
`scratch/real_robot/replays/marker-expert-resumed-20261111T020812Z`. It includes
all 898 frames without skips, preserving episode 473 from the original run.
The final home return was not verified; this is separate from completion of the
five recorded action sequences. Results and plots are in `diagnosis.json` and
`plots/overview.{png,pdf}` with individual episode plots.

Across 898 physical samples: median target-to-DLS residual is 0.443mm, live EE
versus recorded expert measured EE is 3.90mm, and target versus live EE is 19.0mm
(original target versus recorded measured EE: 21.1mm). Measurements are computed
from encoders using the shared FK; they do not independently calibrate the TCP.

The ideal-plant ablation found 38–62mm per-episode median tracking error from the
joint governor alone, versus 2.4–3.2mm from the pose blend alone. The additional
governor targets a nearby, bounded IK waypoint with zero terminal velocity on
each update, substantially slowing these recorded motions. This demonstrates a
controller-postprocessing problem; it does not establish VLA prediction quality.

The direct replay result does not justify disabling the governor for learned
actions. The subsequent KRV100 trial (20261111T022039Z) produced target jumps up
to 100mm and 28 degrees, repeated joint-reference reversals, and severe vibration
reported by the operator. It stopped on the following-error guard. The previous
recommendation to transfer the unfiltered replay settings to VLA execution was
incorrect. Learned-policy execution now requires the acceleration/jerk governor;
unfiltered mode remains available for offline previews and this separate expert
replay diagnostic. Governor tuning still needs offline tracking assessment and
physical validation. Do not compare model success rates from different controller
settings.

The subsequent device-free chunk-handoff audit is saved at
`scratch/real_robot/evaluations/marker-krv100-20261111T022039Z/chunk_handoff_audit.json`.
It reproduces all 188 delivered raw action targets before testing optional
time-aligned overlap blending. The 80/160ms variants reduce position jumps but
increase the worst rotational jump (28 to 49/46 degrees); they are rejected for
live deployment. `ActionStream.handoff_steps` is an offline experimental option,
defaults to zero, and is not enabled by the live evaluator. Ideal-plant governor
comparisons remove rapid reference reversals but retain tracking lag. These
results do not validate physical motion or task success.

For the single governed validation, `run --max-episodes 1 --governor on
--pose-smoothing-ms 80 --joint-acceleration-deg-s2 600 --joint-jerk-deg-s3 6000
--no-return-home --execute` selects episode 473 from the verified batch. These
are candidate limits assessed offline, not established physical stability limits.
The replay clock starts at the first control sample so frame zero is retained.
After the recorded frames, the controller continues toward the final target with
the same governor until its reference velocity/acceleration settle (5s timeout),
then holds without a final home return. Tail samples are logged separately from
dataset samples. A run is fully complete only if every source frame was sent
without skips; resume also checks the acceleration/jerk limits and selection.

The first governed single-episode run `marker-single-governed-20261111T023834Z`
settled without a guard fault, but sent only 150/169 frames and was reported
mildly jerky. Its floor-based source clock duplicated/skipped indices under
sub-millisecond timer jitter despite a nearly 25Hz loop. `replay_frame` now
selects the nearest nominal sample; auditing the original timestamps yields all
169 indices in order without duplicates/skips. Genuine missed cycles still skip
source time rather than sending catch-up bursts. This is a clock fix, not a
claim that physical jerk or tracking lag has been resolved.
