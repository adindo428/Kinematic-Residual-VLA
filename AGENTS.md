# Standing training requirements

## Always retain latest and best checkpoints

User instruction, October29,2026: for every future training run, always save both the latest and the best checkpoints. This applies to all models, variants, sites and hardware placements in this project, including new frozen retries and continuations.

- **Latest** is the newest complete training state. Save periodically, at epoch boundaries, at graceful allocation/stop signals and at completion. Resume from this run's newest valid complete latest state, retaining its original total budget.
- **Best** is the checkpoint with the best finite score on the run's declared held-out validation metric. Default for DOM/VLA runs is minimum physical horizon20 action MSE; preserve a study's explicitly declared metric and direction. Never select using training loss or test/simulator success results. Keep the previous best on ties. Before the first successful validation, record that best is pending rather than inventing a best score.
- Both retained checkpoints must include, or resolve to an immutable generation containing, model weights, optimizer, LR scheduler, per-rank RNG, epoch/update/sample counters, data cursor, original budget and source/data/recipe provenance. Record the best metric, value and saved progress separately from latest progress.
- Use atomic complete saves. `latest.pt` and `best.pt`, or `latest.json` and `best.json` pointing to complete immutable generations, are acceptable. Never implement best as a moving symlink to latest. If latest and best are the same immutable generation, both pointers may share it without duplicating bytes.
- Retention/cleanup must protect both checkpoints, their metadata and every referenced target. Prune superseded unprotected generations only. Never label the current/latest weights as an earlier historical best unless that exact state was saved and validated.
- Before submitting any future family, verify that its physical staged training source implements both retention paths, best-score comparison/metadata and exact latest-state resume. Preserve these guarantees through profiling, microbatch/worker tuning, migrations and requeues.
- Do not edit source used by active or already-submitted jobs. Apply required changes in a fresh frozen attempt, preserving the current run's latest complete checkpoint whenever it has already trained.

Policy record: `maintenance/training_run_register/checkpoint_policy.json`. This requirement does not imply that historical runs already retained separate best checkpoints.
