# KRVLA training run register

The live reference is [KRVLA training run register](https://chatgpt.com/space/page_4ff536c9060081919225467494390460).

It records the six-model DOM manipulation study, superseded production jobs, documented tracking training, the separate DeltaAI DOM launch, and preparation/profiling jobs. The online Page refreshes hourly through automation `refresh-krvla-run-register` in the originating task. Access uses the same authenticated ChatGPT account; the link does not grant access to other people.

`registry.json` stores the records and timestamps. `refresh.py` reads Palmetto and DeltaAI through reusable SSH connections and writes `page-blocks.json` for native Page edits. It never submits or cancels training. `probe.py` runs read-only on Palmetto; frozen training sources and recipes are not modified.

Agents should append new submissions by cluster plus job ID, retain earlier rows, and record source/recipe fingerprints, method, GPU count/class, learning rate, epoch cap, stopping rule, log/output paths and verified W&B links. Update the online Page with fresh read-derived block IDs and hashes, preserving unrelated notes. Distinguish scheduler status, checkpoint metadata, optimizer verification and actual task success. If access fails, keep the last verified values and mark the refresh stale.

For a manual cluster refresh, run `python3 maintenance/training_run_register/refresh.py` from the repository root. It refreshes JSON locally; use the Pages connector to commit updated tables online. SSH socket overrides are `KRVLA_PALMETTO_SOCKET` and `KRVLA_DELTAAI_SOCKET`.
