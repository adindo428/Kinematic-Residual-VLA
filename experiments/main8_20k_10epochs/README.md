# Main eight runs: checkpoint continuation on the 20k DOM split

The main study has DOM+velocity versus local KRV100, plus baseline/KR pairs
for π0.5, SmolVLA and VLASH. The seven cluster members continue their own
previously trained checkpoints. The local KRV100 run is retained as requested.
Its current full-data split differs from this 20k stage, so it must be reported
with its actual training provenance; it is not automatically a matched subset
run. The optional phase-gated KR experiment is outside these eight.

| Site | Allocation | Models |
|---|---|---|
| Delta | one node, four H200 GPUs | π0.5 baseline and KR, two GPUs each |
| DeltaAI | one node, four GH200 GPUs | VLASH baseline and KR, two GPUs each |
| Palmetto | three separate allocations, four A10040 GPUs per model | SmolVLA baseline, SmolVLA+KR, DOM+velocity |

Delta currently exposes H200 rather than H100 resources; the user approved
four H200s. Separate GPU visibility and CPU affinity prevent models sharing
devices within each allocation. Effective batches remain32 for π0.5/VLASH,
64 for SmolVLA, and352 for DOM. SmolVLA moves from two GPUs to four;
existing rank streams are retained and two new streams initialized deterministically.
DOM moves from eight GPUs to four. These topology/software
changes can alter the numerical training trajectory.

All cluster runs use exactly the same20,000 training episodes:9,000 pick,
9,000 place,2,000 long horizon;2,406,773 frames across13 task/speed strata.
Original duplicate groups and2,015 validation/2,011 test episodes are retained.
Selection IDs and parent split hashes are in `split.json`,
`selected-episodes.json` and `subset-report.json`.

Each run must complete at least **ten additional sampler epochs on this
subset**, independent of historical full-data epochs. Baseline/KR budgets
visit the same shuffled anchors, dropping an incomplete global batch at the
end of each pass. The formula is `10×floor(2,406,773/global_batch)`:
SmolVLA adds 376,050, π0.5/VLASH add 752,110 and DOM adds 68,370 updates.
No early stop can end
this stage before the ten-epoch minimum.

This is a **continuation after earlier full-data exposure**, not a claim that
the models have only ever trained on20k episodes. Explicit user authorization
to resume checkpoints supersedes the prior fresh-subset initialization rule.
Parent action/state normalization, velocity centering, weights, Adam moments,
LR scheduler and available per-rank RNG streams are restored. The sampler
starts at subset epoch0/cursor0 because episode membership changed. Historical
update/sample counters are retained separately from the new subset epoch.
Original20k/30k LR schedules continue at their current rates; DOM retains
its original100-epoch scheduler horizon. No warmup or LR increase is added.

The newest completed production state is frozen under `parents/<name>/`,
with original origin paths, progress and SHA256 in `migration.json`.
The active π0.5+KR run was checkpointed at60,004 updates before migration;
its preserved checkpoint remains on Palmetto and is transferred to Delta.
DOM+velocity resumes generation `u000218468-ff527846` at218,468 updates.
Other origins: π0.5 baseline19,397 updates; SmolVLA±KR20,000 each;
VLASH±KR30,000 each. Parent artifacts and tracker IDs are preserved.

`supervise.py` manages the disjoint model processes inside one allocation.
Each model first performs200 decoded effective-batch updates from its actual
restored checkpoint, finite-gradient and physical20×7 inference checks,
strict checkpoint/optimizer/scheduler/RNG reload, and15% memory-headroom checks.
Readiness weights are discarded. Production reloads the frozen parent or its
own newest complete checkpoint, then continues the original W&B identity.
Signals checkpoint every active model before requeue. A model failure stops
the allocation's other models after their own saves; it does not bypass
admission or restart with random weights.

New run roots:

- Palmetto: `/scratch/aparame/Dynamic_VLA/main8-20k-10epochs-20261021-r2`
- Delta/DeltaAI shared storage:
  `/work/nvme/bivv/aparameshwaran/dynamicvla/main8-20k-10epochs-20261021`

Environments are isolated per architecture. Model/source hashes and
site-specific preparation locks are checked on every launch. Data and
checkpoints stream between cluster storage through existing authenticated
connections; they are not stored on the Mac. Submission receipts and actual
scheduler states are recorded separately from GPU readiness and training
completion. Closed-loop success still requires simulator evaluation.

## Submitted allocations

| Models | Site | GPUs | Job ID |
|---|---|---|---|
| SmolVLA baseline | Palmetto | 4× A10040 |17839146|
| SmolVLA + KR | Palmetto | 4× A10040 |17839147|
| DOM + velocity | Palmetto | 4× A10040 |17839148|
| π0.5 baseline and KR | Delta | 4× H200, two per model |23100076|
| VLASH baseline and KR | DeltaAI | 4× GH200, two per model |3429279|

Palmetto GPU jobs depend on CPU preparation17839145. The previous unstarted
GPU attempt17837949–17837951 was retired before training; r2 retains native
DOM local training buffers and canonical leader buffers at checkpoint and
validation, matching its prior trainer. Original CPU audit17837948 completed
and supplies source checkpoint digests for the Delta transfer check.

The two Delta systems have separate schedulers. Their GPU jobs were submitted
held, with bounded release helpers. The H200 request releases after existing
transfer processes finish and the source digest receipt arrives, then verifies
all copies before GPU admission. The GH200 request releases after that shared
verification succeeds. This avoids using a job ID from one scheduler as a
dependency on the other. Current states are recorded in `deployment.json`;
submission is separate from successful GPU readiness and training completion.

## Original trained checkpoint locations

- SmolVLA and VLASH pairs:
  `/scratch/aparame/Dynamic_VLA/cross-vla-krfm-a100-opt-20261019/runs/<name>/latest.pt`
- π0.5 baseline:
  `/scratch/aparame/Dynamic_VLA/cross-vla-krfm-a100-pi05-cache-20261020/runs/pi05_baseline/latest.pt`
- π0.5 + KR:
  `/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/runs/pi05_kr/latest.pt`
- DOM + velocity:
  `/work/nvme/bivv/aparameshwaran/dynamicvla/domv-scale-ddp-20261014/runs/domv100/generations/u000218468-ff527846/`
  (`model.safetensors` and `training_state.pt`).

Immutable stage copies live in each new run root's `parents/<name>/`.
The local KRV100 output remains under the existing workstation family;
its separate full-data provenance is retained. No original trained artifact
is deleted when retiring a superseded allocation.
