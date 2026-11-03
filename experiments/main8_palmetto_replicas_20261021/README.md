# Independent Palmetto replicas of pi0.5 and VLASH

The user explicitly authorized separate replicas alongside the existing
Delta and DeltaAI runs on October 21, 2026. There are four independent models
(pi0.5 baseline/KR and VLASH baseline/KR) on each of two placements:

- Four A10080 GPUs per model, on two nodes with two GPUs, 32 CPUs and 224 GiB per node.
- Two H200 GPUs per model, on one node with 32 CPUs and 256 GiB.

Each allocation is capped at 48 hours and checkpoints before requeue. Existing
SmolVLA, DOM, Delta, DeltaAI, workstation and benchmark jobs are unchanged.

These replicas branch from the same frozen trained parent checkpoints as the
main-eight 20k stage. They use distinct output roots and separately reserved
W&B identities, with the original tracker recorded only as lineage. Parent
weights, optimizer moments, scheduler and available rank RNG streams are
restored. Four-rank replicas initialize the two additional rank streams
independently and deterministically. Numerical trajectories may differ with
hardware and topology.

All replicas use the same 20,000 training episodes and 2,406,773 anchors.
Ten additional subset sampler epochs at global batch32 add752,110 updates:
pi0.5 baseline19,397 ->771,507; pi0.5 KR60,004 ->812,114;
VLASH baseline/KR30,000 ->782,110. Historical full-data exposure is separate
from the new subset epoch/cursor. The original30,000-update LR decay is
preserved, followed by its final rate. No new warmup or early stopping is added.
Parent normalization, audited data, native trainable sets, precision, horizon20
and analytical privileged-current-velocity KR remain unchanged.

Production requires the exact placement's own200-update decoded GPU check,
coordinated save, strict optimizer/scheduler/per-rank RNG and physical20x7
inference reload, plus15% GPU and per-node RAM headroom. Production reloads
the trusted parent or its own newer complete production checkpoint. It never
uses readiness weights. Model/source/data/runtime/hardware identity is checked
on every launch. Startup signal handlers are installed before model imports.

Submission receipts are in maintenance/main8_palmetto_replicas_20261021.
Preparation17859068 is a CPU-only two-node job. It passed15 CPU tests and an
actual four-rank/two-node collective and forwarded-USR1 contract before
copying and validating all trusted parent checkpoints. A100 jobs17859069–72;
H200 jobs17859073,17859076–78. All eight have afterok17859068, followed by
their own GPU profile and production within the same allocation. Submission
and CPU tests are not evidence of GPU readiness or production progress.

Remote roots:
/scratch/aparame/Dynamic_VLA/main8-replicas-a100-20261021
/scratch/aparame/Dynamic_VLA/main8-replicas-h200-20261021

Frozen Python source fingerprint:
f5f0dad6be2ae19c43618a9f5e838c19ae2ac19e778c001f8890efd741b6db53

No datasets, weights or checkpoints were stored on the Mac. DOM adaptations
are distinct from exact original pretraining reproduction; validation MSE
is not simulator success.
