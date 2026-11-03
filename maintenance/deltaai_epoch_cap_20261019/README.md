# Ten-total-epoch operational stopping cap, October19,2026

User requested both current DeltaAI runs stop at10epochs for now. This applies to3381947 (`domv100`,8ranks) and3382533 (`krvp100`,16ranks). DOM already saved3.437627896epochs (218468updates,76900427samples;3complete epochs/cursor9789824), so the cap leaves6.562372104epochs. Phase-gate has not started. Both jobs remain PENDING/Priority with existing IDs, queue age and48hour allocation requests.

## Continuation choice

The original recipe/manifest hashes are part of the saved checkpoint fingerprint, and the100-epoch cosine schedule has already been used for3.44epochs. Rewriting recipe.epochs would invalidate exact resume and change the LambdaLR trajectory. Therefore this deployment uses a separate, explicit stopping control:10total epochs, while preserving the original100-epoch scheduler horizon, optimizer, weights, per-rank RNG, run ID and checkpoint identity. Both variants retain the same scheduler trajectory for comparison and future continuation if requested. Validation stopping remains min5epochs/patience3/relative0.5percent improvement in h20/action_mse.

Remote control: `/work/nvme/bivv/aparameshwaran/dynamicvla/setup-epoch-cap-20261019/control.json`; fingerprint `18c2228e46ed0db13d16e66fcd5f263d3903c93eccaac97c93ad936304f6ef46`. `sitecustomize.py` SHA256 `a8e8c8b6fc2acb996baeca3158a8cad63c954dc2a1cf0a0b2aba1bbf74940d20`.

The existing activation script gets a guarded PYTHONPATH/control-file export only for these two job IDs. Their already-spooled batch scripts source that activation at each allocation. A checked import adapter applies only when DOM100_NODE_MODE=train. It checks its own digest, control digest, exact trainer path/hash, job ID, family/manifest, recipe consistency and distributed world size. It changes the loop stopping limit and publishes the control in checkpoint progress, run provenance and W&B metadata. It explicitly retains the original scheduler horizon and fingerprint expression. Readiness/smoke/benchmark modes use the original functions. Installation errors fail the worker instead of silently training to100epochs. Other jobs are outside the guarded scope.

`controlled_train_distributed.py` is a rendered review of the exact small transformation. Actual frozen model/trainer sources and family manifests/recipes remain unchanged. `deploy.py` checked all frozen family artifact hashes, preserved the prior activation script, verified both jobs were pending, and recorded installation in `deployed.json`. `before.json` preserves pre-change state; `startup-check.txt` records actual activation/import checks for both environments. No GPU optimizer work or new dataset verification was run for this control change. Original source SHA f991015a..., trainer SHA336c1ecd... retained. Latest DOM checkpoint u000218468-ff527846/fingerprint74e28089... is unchanged.

## Budget and scheduling

789GPU-hours were available. Historical DOM training_seconds implies13.741998training hours/epoch, or roughly90.18more training hours/721.44GPU-hours to epoch10 before startup/checkpoint/validation overhead. The phase-gate16-GPU rate is unmeasured. Ten epochs reduce planned training, but both existing48hour requests still total1152GPU-hours, and the remaining balance may not cover both runs. No wall-time/resource request or priority was changed. New NCSA Delta access is user-reported pending.

Read-only register refresh now records effective target_epochs=10 separately from scheduler_horizon_epochs=100. Online Page was updated with the cap and current saved progress; other local/Palmetto/cross-backbone training budgets are unchanged.
