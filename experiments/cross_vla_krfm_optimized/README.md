# Throughput-tuned A100 paired VLA campaign

This frozen replacement covers the six pi0.5/SmolVLA/VLASH baseline/KR pairs
on two A10080 GPUs each; original H200 queue is retained unchanged.
The official recipes, audited split, pinned weights, horizon20, privileged
analytic KR and native trainable parameter sets match the preceding campaign.

Execution improvements: bounded per-run microbatch/worker sweeps, pinned
host buffers and asynchronous copies, cached affine normalizers, reduced
microbatch-level synchronizations, and gradient bucket views. Benchmark
microbatches preserve effective batch32/64/32 and identical global anchor
sets at each optimizer update. Per-anchor padded loss normalization preserves
the prior microbatch1 objective for mixed valid horizons. Numerical training
trajectories may differ with microbatching; no bitwise claim is made.

Reliability: clear existing GPU optimizer moments before strict checkpoint
reload; mmap trusted CPU checkpoint state; coordinate time-based saves across
all ranks to prevent ALLGATHER/ALLREDUCE mismatch. A deliberately unequal-clock
two-rank CPU test supplements full decoded checkpoint/RNG/inference readiness.

For each model, test bounded microbatches then worker counts4/8/12, keeping
CPU count within32. Choose highest measured decoded samples/s retaining15%
GPU and conservative node RSS headroom; final20-update readiness includes
an asymmetric save trigger, optimizer/scheduler/RNG reload and inference reload.
Production afterok plus matching runtime/source/data/hardware identity gate
is mandatory. Separate tuning scratch is never used as production progress.

Previous production checkpoints are explicitly migrated only after old jobs
stop and the immutable checkpoint SHA256, recipe, preparation, hardware and
world size match. Restore optimizer, scheduler, per-rank RNG and consumed-data
cursor; retain existing W&B ID only for an actually checkpointed continuation.
Never infer saved progress from tracker logs. Models without a production
checkpoint restart from the pinned initialization, not readiness weights.

No datasets or weights are stored on the Mac. Failure history and source
snapshots remain preserved. Logs/tuning/profiles/run directories are separate
under /scratch/aparame/Dynamic_VLA/cross-vla-krfm-a100-opt-20261019.

## Submitted optimization jobs

Checkpoint migration: 17709582.

| Model | Tuning baseline / KR | Production baseline / KR |
|---|---|---|
| pi05 | 17709583 / 17709585 | 17709584 / 17709586 |
| smolvla | 17709587 / 17709589 | 17709588 / 17709590 |
| vlash | 17709591 / 17709593 | 17709592 / 17709594 |

## Required resume policy

For every future resubmission, use the latest complete production checkpoint
for that exact run. Prefer its own latest.pt over an immutable migrated parent;
restore model, optimizer, scheduler, per-rank RNG, update/sample counters and data
cursor, retaining the original total update budget. If an expected checkpoint
is missing or incompatible, stop and reconcile instead of starting fresh.
Never substitute readiness checkpoints or tracker-only progress.

At initial submission, validated parent manifests resumed SmolVLA baseline956,
VLASH baseline745, and VLASH KR738 updates. SmolVLA KR then had no saved state
and initialized from pinned weights. All four now have their own production
checkpoints, which take precedence for any future continuation. Both SmolVLA
runs completed their20,000-update budgets; do not restart them. Pi05 baseline/KR
never reached production and their rows here are superseded; see the separate
cross_vla_krfm_pi05_cache profile-only recovery and its submission receipts.
