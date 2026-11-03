# Palmetto H200 consolidation October 22

The user cancelled A100/slower Palmetto production and requested H200
continuation from the latest valid saved state, plus H100 optimization for
future runs. Other clusters and the workstation are unchanged.

Seven current A100 jobs are confirmed cancelled:
17839146/47/48 and17859069/70/71/72. Four existing H200 replicas
17859073/76/77/78 continue their own newer production checkpoints. They
were not overwritten with older independent A100 replica weights.

CPU preparation17912899 COMPLETED exit0,1m20s. It passed15 contracts and
verified independent copies of SmolVLA baseline/KR20,000-update parents and
DOM+velocity218,468-update parent. SmolVLA H20017912900/01 passed actual
200-update strict readiness and resumed production; observed new logged
updates22,030/20,740. DOM H20017912902 is pending allocation and its own
readiness. The existing20k episode stage and ten additional subset epochs,
optimizer, original LR schedules, normalization and tracker identities remain.

Storage cleanup is deliberately split:

- Four exact output checkpoint files belonging to cancelled A100 pi05 jobs
  17859069/70 were deleted. Absence verified;161,827,229,200 file bytes.
  Source, logs and JSON/failure/provenance receipts remain.
- Eight redundant parent copies total129,348,096,607 bytes. Each has a
  separately owned/inode H200 copy with matching verified migration SHA.
  See redundant-parents-review.json. Automatic approval rejected shared-parent
  cleanup. The user explicitly approved these exact eight files. Job17923816
  rechecked both full SHA256s, deleted the eight A100 copies, and verified
  all H200 copies retained (COMPLETED exit0,6m20s).
- Historical checkpoint families and all datasets/pretrained assets remain.

The earlier broad cleanup_checkpoints.py plan was blocked before execution.
Do not execute it. Any approved parent deletion must use the exact eight
reviewed paths and recheck the independent H200 copies and cancelled jobs.

H100 future-runtime diagnostic17913478 is bounded to two H100s/four hours,
maximum eight GPU-hours. CPU preparation17913477 passed six contracts and
an actual production-forbidden guard. Source, measured selections and submissions are recorded
under maintenance/main8_h100_optimization_20261022. It benchmarks microbatch
and loader-worker choices for all seven model variants, then verifies full
200-update checkpoint/RNG/physical inference/headroom. No production or W&B
training is allowed. H100 results are pending; no speedup is claimed.

Page222 was saved with fresh guards and exact readback. Unrelated Page
blocks were preserved and cancelled job rows omitted. Local registry keeps
all cancellation and deletion history. No dataset/checkpoint files were
stored on the Mac.

Final cleanup:291,175,325,807 bytes of checkpoint files removed across12 exact files.
All seven H200 models are verified production training; the H100 diagnostic
COMPLETED exit0 in2h09m34s and approved all seven variants. No H100 production.
