# Bounded H100 optimization for future model runs

This diagnostic compares microbatch size and4/8/12 data-loader workers on
exactly two H100 GPUs. Every comparison restores the same trusted trained
parent, optimizer/scheduler and rank RNG; global batches, math, data and
normalizers are unchanged. All seven main cluster variants are included.
Short eight-update timings are separate from200-update readiness timings,
which include coordinated save/reload and physical inference checks. Initial
allocation peaks and conservative summed per-node RSS retain15% headroom.
Failed/OOM attempts are ineligible and their receipts are retained.

Production is explicitly forbidden in train.py and W&B is disabled. There
is no H100 production submission. No optimized speed claim is made until
actual results exist. H200 production is unchanged.

CPU preparation17913477; bounded GPU diagnostic17913478, one node2H100,
32CPUs/256GiB,4-hour cap (at most8GPU-hours). Read-only parent references
point to retained H200 family copies, avoiding additional stored checkpoints.
Root /scratch/aparame/Dynamic_VLA/main8-h100-optimization-20261022.
Results diagnostic-results.json, diagnostic-complete.json and
 tuning/<name>/attempts.json/selection.json. Reconcile existing receipts before
any retry; never rerun the submit helper for recorded jobs.
