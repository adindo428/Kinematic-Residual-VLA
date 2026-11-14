# Action-error retraining

## Completed static collection continuation

The new queue is
`scratch/real_robot/finetuning-runs/static-continuation-constant-20261112/`.
Its order is KRV100 with **constant** gating, DOMV, DOM, SmolVLA baseline,
then SmolVLA+KR. Each continues its own selected best real-robot checkpoint
with a new optimizer. Native SmolVLA continuation weights are hash-pinned and
loaded strictly before applying the new training-only statistics. Architecture,
observation layouts, action horizon, and each model's existing input convention
remain unchanged. The old v6 supervisor remains stopped; this is a separate queue.

The frozen local derivative contains all **859 episodes**, including 295 new
static demonstrations. It preserves all earlier train/validation/test episode
assignments and the processed v5 prefix (including 80 ms bounded extrapolation
and grasp annotations), instead of moving old holdouts into training. Expert
robot state/action labels were checked byte-for-value against the latest release.
New static episodes retain their recorded split and explicit assumed/inferred
quality metadata. Combined splits are **706 train / 80 validation / 73 test**,
with **101,781 common eligible training anchors**. The parent Hugging Face training
commit is `370df1b682f05d5ab68a3188f40f892e25b7b958`; `dataset/` is a local
derivative, not a claim that the parent remote branch has the same processed labels
or splits. `preparation.json`, `dataset/continuation_dataset.json`, and
`source-manifest.json` record provenance.

Sampling balances task, static/dynamic source and phase using capped inverse
square-root frequencies. The fixed 256-anchor validation probe uses the same
three noise seeds and decoded physical action metrics as the previous runs;
test episodes are not used for checkpoint selection. Loss scales, first-four-step
weighting, maximum 15 epochs, learning rate 3e-6, scheduler/patience and existing
photometric ranges are retained.

All five save disposable GPU checks/throughput reports before production training.
DOM-family runs use effective batch 16, four persistent loader workers, bounded
512 MiB per-worker episode caches, union/batched decoding with one decoder thread,
and vectorized photometric augmentation. Their full sampler is checked against
native inference with fixed noise, and gradients/optimizer execution are checked.
The native SmolVLA pair starts with the previous optimized settings and reruns
the existing throughput search on the selected continuation weights/new data;
selected settings are saved before training. No changed benchmark weights are
saved. SmolVLA raw/EMA checkpoint selection remains enabled.

Read `queue.json`, `logs/`, and `benchmarks/` for progress. DOM-family checkpoints
and status are in `dom-runs/{family}/`; SmolVLA checkpoints/status are in
`runs/{family}/`. `supervisor.json` records the detached queue PID.

## Previous v5 queue

The launched queue is under
`scratch/real_robot/finetuning-runs/action-v5-fullsampler-20261110/`.
Read `queue.json` and each family's `status.json` for progress, and `training.log`
for the shared log. `launch_contract.json` records the exact experiment and source
hashes. DOM started in W&B run `4qp2loiv`; subsequent families get independent runs.
`final-evaluation-status.json` tracks a waiting worker that repeats the shared
ten-validation-episode protocol when all training finishes, writing
`final-validation-10/before_after.json`. No improved results are claimed before
that comparison finishes.

The v5 continuation recipe is prepared by `workstation/prepare_action_retraining.py`.
It continues DOM, DOMV and KRV100 from their own already selected best v4
checkpoints. It does not re-rank old epochs. The dataset, splits, masks and clean
training statistics stay frozen. This is a continuation comparison, not a new
common-initialization ablation.

The audit checks all episode lengths and finite monotonic timestamps, summarizes
recorded sensor/command ages, verifies the earlier ten-episode decoded XYZ labels
against absolute command-FK targets, and compares a hold-current-position baseline.
It does not prove camera/command synchronization or physical calibration, and
does not shift timestamps or rewrite command labels automatically.

Training gives the first four action steps weight 4 and the other steps weight 1,
normalizing over valid timesteps per anchor. Task/phase sampling uses inverse
square-root group counts, mean-normalized and capped at 0.25–4, with replacement.
Unknown phase weights are additionally capped at 1.
Candidate, inferred-held and visually confirmed labels retain their distinctions.
The existing phase vocabulary does not supply separate placement ground truth.

Every fourth update, a rotating two-example slice also receives a fully
differentiable ten-step sampled action loss. This restores KRV drift once,
unnormalizes using the checkpoint statistics with gradients, and converts deltas
to absolute poses. The loss adds XYZ MSE / (20 mm)^2, SO(3) chordal error /
(10 degrees in radians)^2, and gripper MSE / 0.2^2, scaled by 0.1. These scales
are initial experiment settings, not measured tolerances. The flow and auxiliary
backward passes accumulate before one clipped optimizer step; every update still
has a flow objective. Reconstruction uses random noise during training.

Selection uses the same scaled physical objective on 256 fixed, task/phase-balanced validation anchors
and three fixed independent noise seeds. It averages all samples, not the best
sample. Raw first-action and full-horizon metrics are also logged. Full validation
flow loss is retained for diagnostics. Learning-rate scheduling and patience use
the physical score. The untouched starting checkpoint is retained if training
never improves its score. Test episodes are never used for selection. Final
comparisons should repeat the earlier ten-episode protocol and full validation,
with per-task/phase results and physical trials assessed separately.

Learning rate is 3e-6, AdamW weight decay 0.01, warmup 200 updates, maximum 15
continuation epochs and patience 5. Photometric augmentation stays enabled;
proprioceptive augmentation stays disabled. No controller or robot settings change.

## Expert recovery data

New recovery demonstrations need an operator and real recorded corrective
commands. No recovery data is invented or added to the current frozen release.
Use the existing `real_robot.collect` workflow in new sessions to demonstrate
corrections for observed offsets, missed grasps, delayed interception and regrasping.
Record successful stationary grasps too. Keep failed attempts and rollout context
as diagnostic evidence; do not relabel policy mistakes as expert actions.

For an actual policy deviation, retain its policy recording/checkpoint identifier.
If a human recreates an observed deviation instead, label that explicitly; it is
not on-policy DAgger data. Use separate collection sessions for training and
validation, preserving the final held-out test sessions.

After recording, create a context JSON with these fields:

```json
{
  "schema": "krvla_expert_recovery_v1",
  "operator": "OPERATOR",
  "parent_checkpoint": "PATH_OR_RECORDING_IDENTIFIER",
  "scenario": "DESCRIPTION_OF_OBSERVED_ERROR_AND_CORRECTION",
  "deviation_source": "actual_policy_deviation",
  "expert_corrective_commands": true,
  "synthetic_labels": false
}
```

Use `expert_recreated_observed_deviation` for a recreated state. Attach it to the
raw recording before export:

```bash
.venv/bin/python -m real_robot.recovery_demonstrations \
  --session PATH_TO_SESSION/recording --context PATH_TO_CONTEXT.json
```

The exporter carries this provenance into `meta/sources.jsonl`. Export, quality
audit, annotate grasp timing and create a new training-only-statistics release
before mixing new recovery sessions with the original demonstrations. The current
v5 queue uses only the existing frozen release and contains no new recoveries.
