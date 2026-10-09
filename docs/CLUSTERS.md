# Cluster launchers and experiment provenance

For the lab workstation use the local launchers linked in the [root README](../README.md). Slurm scripts and old result-merging utilities contain cluster-specific paths; they are not local-PC entry points.

## Palmetto

The current DOM training code is under `third_party/DynamicVLA`. Set `DVLA_PROJECT_DIR` to that directory and `DOM100_ROOT` to a newly prepared or verified existing frozen family, then follow [DOM100.md](../third_party/DynamicVLA/DOM100.md). The active deployment details, source/family names and early-stopping recipe are maintained there. Four/eight-GPU distributed scripts support A100/H100/H200 classes; training profiling evidence is tied to exact hardware, software and source fingerprints.

The preserved 75-epoch tracking experiment is in `experiments/tracking75/source`. Its original recipes retain training provenance paths; use `scripts/local_training_config.py` for a workstation copy. `source/slurm` and `palmetto_reference` preserve historical launch examples. Do not resubmit an old script without checking dataset, output, normalization-reference and checkpoint paths; some old result directories were intentionally consolidated.

The official DOM baseline evaluation uses a pinned unmodified upstream checkout in `/scratch/aparame/Dynamic_VLA/runs/official_dom_table1_20261009`. Its remote L40S launch helpers are preserved in `evaluation/official_dom/scripts/run_suite.py`, `submit.py` and `slurm/`. Results are under `/home/aparame/Research/Dynamic_VLA/RESULTS/official_dom_table1/20261009`. They are independent of the research-fork training and tracking adaptation. The workstation guide uses the same source/checkpoint revisions with configurable local paths.

Cluster batch directives (partition, account, GPU resource syntax, walltime and log paths) must match the target cluster. Use a fresh source snapshot/family when changing recipe or model code; a Git pull must not mutate the source beneath an already frozen run.

## DeltaAI

[DELTA_AI.md](../third_party/DynamicVLA/DELTA_AI.md) documents the separate ARM deployment and environment adapter. `configs/deltaai_torch_constraints.txt` is an ARM/DeltaAI constraint file, not the RTX 3080 CUDA 12.6 workstation requirements. Simulator evaluation is separately deployed; model-training validation on datasets does not require Isaac Sim.

## Files to retain with results

Keep protocol/manifests, checkpoint configs and hashes, split/statistics artifacts, raw telemetry, runtime versions/GPU identity, commands/logs and verification reports. This permits offline rescoring without rerunning other models. The tracking success definition is version 4: mean per-episode time within 3 cm; do not silently mix old binary-success or 2 cm scores into the current table.
