# DOM 20k episode experiment

This is a fresh, reproducible dataset budget experiment for the six matched
π0.5, SmolVLA and VLASH baseline/KR runs. The default training allocation is
9,000 pick + 9,000 place + 2,000 long-horizon episodes, selected only from the
audited parent training partition. Existing validation (2,015) and test (2,011)
episodes keep their exact membership. Thus 20k refers to training episodes;
the complete derived split has 24,026 episodes.

The selected training set contains **2,406,773 frames**, compared with the
parent's 22,370,201: a 9.29× reduction in frame work per epoch. Five epochs
require 188,025 updates per SmolVLA run (global batch 64) and 376,055 per
π0.5/VLASH run (global batch 32). Each baseline/KR pair differs only in its
name and KR switch. Exact IDs and counts are in `split.json`,
`selected-episodes.json`, `subset-report.json` and the six `recipes/` files.

`subset.py` allocates each task's quota across existing speed bins using capped
square-root weights. This reduces the original dominance of place and middle
speed bins while retaining scarce motion regimes. It selects whole duplicate
trajectory groups using a seeded hash order and exact group-size subset sums.
Cross-partition duplicate groups, missing metadata, overlapping partitions and
unattainable quotas fail explicitly. Selection does not use model scores or
the official simulator test outcomes. The report records task/speed counts,
frame counts, exclusions and parent hashes.

All three task families are retained by default. An explicitly narrower study
can pass e.g. `--quotas '{"pick":10000,"place":10000,"long_horizon":0}'`
to `prepare.py` in another fresh directory. A different allocation requires its
own experiment and disclosure; omitted/zero task families are reported.

`prepare.py` reuses immutable pinned released models and audited parquet/video
hashes without downloading or duplicating the dataset. It recomputes action
and state statistics on a deterministic uniform sample of 200k anchors from
the selected training episodes, following the existing physical-coordinate
estimator with delays 0..3. The current object velocity mean is recomputed
over **all selected training frames**. Neither held-out episodes nor inherited
full-data normalization contribute. Each baseline/KR pair shares these same
statistics. The setup verifies hashes before reading parquets; training verifies
both parquet and video bytes on access.

`budget.py` sizes each recipe to five complete native sampler epochs:
`steps = 5 * floor(selected_training_frames / global_batch)`. It drops only
the trailing incomplete batch per shuffled epoch. Cosine decay spans the new
budget, with at most 1,000 warmup updates (5% for smaller budgets). Existing
per-model peak/final learning rates, global batches, optimizer parameters,
native trainables, physical action objective and horizon20@25Hz are retained.
This is a newly specified DOM subset fine-tuning recipe, not an exact published
training reproduction. Wall-clock savings depend on selected frame lengths
and actual hardware throughput.

Training starts from the pinned **released pretrained weights**. Full-DOM
trained parent checkpoints must not initialize this experiment: those models
have already seen episodes outside the 20k budget. Production can resume only
its own checkpoint, with matching preparation, source, recipe, hardware/runtime,
optimizer/scheduler state, RNG streams and sampler cursor. New W&B identities
identify the subset experiment. Readiness weights are never production weights.

Each production requires its own matching GPU profile: 200 decoded effective
batch updates, finite gradients, weight change, physical20x7 inference, strict
checkpoint/optimizer/scheduler/RNG restoration and at least15% memory headroom.
The training source and all preparation receipts are hashed and checked at
launch. Preparation does not cancel, modify or replace existing training jobs.

Remote destination:
`/scratch/aparame/Dynamic_VLA/cross-vla-krfm-20k-20261020`.
Parent preparation/environment:
`/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020`.

On Palmetto, after staging the source:

```sh
python3 launch.py render
python3 launch.py submit --command prepare
# Once preparation succeeds, submit each profile and then its production.
python3 launch.py submit --command profile --recipe smolvla_baseline --dependency PREPARE_JOB_ID
python3 launch.py submit --command train --recipe smolvla_baseline --dependency PROFILE_JOB_ID
```

Preparation runs on 8 CPUs/32GiB; π0.5 uses two H200s and SmolVLA/VLASH use
two A10080s for profiling and training. The scripts checkpoint and requeue
across48h allocations. Each submitted job ID is saved immediately; an unknown
submission result blocks resubmission until reconciled. GPU submission is a
separate action from preparing this subset. Do not edit a submitted source;
repairs require a new attempt directory and new readiness.

Local checks: `python3 -m unittest discover -s experiments/cross_vla_krfm_20k -p test_subset.py -v`.
These check deterministic selection, exact task budgets, duplicate integrity,
held-out leakage rejection, explicit exclusions, speed allocation, complete
epoch accounting, scheduler endpoints and rejection of full-data initialization.
GPU readiness and closed-loop simulator success remain separate measurements.
