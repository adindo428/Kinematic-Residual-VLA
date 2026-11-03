# DOM retained dataset audit — 12 October 2026

Read-only full audit on Palmetto (17119970, sixteen CPU shards, four concurrent) and DeltaAI (3369190, one 32-CPU task; one GH200 reserved because DeltaAI forbids CPU-only submissions). DeltaAI depends on corrected smoke 3369185. Both smoke checks passed; first DeltaAI attempt 3369177 lacked the training FFmpeg library path, which was corrected in the audit launcher only.

Scope: 201,227 retained episodes per cluster, 22,828,297 labeled frames, both training cameras (opst_cam and wrist_cam). Verify all frozen artifacts, split uniqueness/coverage and duplicate grouping, all parquet and video hashes, row counts, action/state/environment dimensions and finite values, task IDs, monotonic nominal timestamps, every frame queried by the training adapter through TorchCodec approximate seek, paired presentation timestamps, and phase-cache hashes/shapes. The unused side camera and excluded episodes are outside scope. This cannot establish demonstration success, correct physical labels, or guarantee model task completion.

Known metadata warning: manifest.episodes remains 201233, but episodes.json, split and coverage correctly contain 201227. Frozen manifests and training remain unchanged.

Reports: /scratch/aparame/Dynamic_VLA/dataset-audit-20261012/results on Palmetto; /work/nvme/bivv/aparameshwaran/dynamicvla/dataset-audit-20261012/results on DeltaAI. status-*.json is atomic progress, episodes-*.jsonl contains every checked episode and errors. summarize.py prints total progress. A clean result requires all expected shards and episodes completed, zero failed checks, and zero metadata errors. Scheduler completion alone is insufficient. SHA256 of audit.py is recorded in every status.

The existing hourly run-register probes now collect audit counters and aggregate array scheduler states. No data repairs, exclusions, training submissions, cancellations, or changes were made by the audit.
