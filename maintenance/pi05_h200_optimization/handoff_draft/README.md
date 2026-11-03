# Exact-run handoff validation draft

Nine local standard-library safety contracts passed. This is validation code only,
not a deployed allocation controller or submission helper. It cannot signal, cancel,
copy weights or submit. Original GPU profile17740320 completed51m31s exit0; both
variants passed200-update full readiness and exceed1.25x current A100 medians.

Remote controller construction/validation and submission await restored Palmetto
authentication. Before shipping any orchestration, inspect actual A100 Slurm script
identity, verify source/recipe/runtime/receipt SHA on Palmetto again, implement
allocated-H200-first graceful USR1/checkpoint/auto-requeue reconciliation, inspect
trusted newest complete production checkpoint with CPU mmap, terminal parent before
immutable copy, unchanged optimizer/scheduler/per-rank RNG/cursor/30k cap and W&B
identity, then matching native target readiness/release gates. Test the controller
and scheduler preflight; freeze before submission and persist every returned ID.
Use a separate remote cross-vla-krfm-pi05-h200-control-20261020 folder. Never add
Python files to submitted training source. A100s continue until own H200 allocation.
