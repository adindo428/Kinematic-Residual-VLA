# DeltaAI interactive π0.5 KR comparison

User requested the comparison on a DeltaAI interactive node, 2026-10-21.

Status: the full40,456,772,509-byte authorized transfer completed and matched
SHA25609c0466b8809bc8e5f0dc248da135bcb186c64bb0759557def0bfdaee9119e6c.
The210-update anchor sequence matches across clusters. Sole current interactive
diagnostic3429682 is queued PENDINGPriority under control_queue_r3. Two GH200s,
32CPUs,256GiB, at most two allocated GPU hours. The controller can wait for
resources without a premature queue deadline. No GPU comparison has run yet.
Previous requests3429648 and3429662 are cancelled, both with zero GPU runtime;
the latter was replaced before its queue watchdog could expire. No production
changed. Do NOT rerun any already-recorded launcher or transfer helper.

Earlier automatic export approval rejection and expired SSH sessions are resolved;
their receipts remain in `prior-transfer-approval-block.json` and staging logs.

Verified DeltaAI access through the existing SSH master with ControlMaster=no.
No existing interactive allocation was running; unrelated job3429279 is held and
was left unchanged. Partition ghx4-interactive has a two-hour cap and GH200120GB
GPUs. No production or checkpoint was modified and no W&B run was created.

An isolated environment was created at
`/work/nvme/bivv/aparameshwaran/dynamicvla/pi05-comparison-20261021/env`.
Verified PyTorch2.12.0+cu130, Transformers4.53.3, tokenizers0.21.4,
draccus0.10.0. These are separate from production environments. The existing
DeltaAI DOM environment uses LeRobot0.3.3 and lacks π0.5; the proposed diagnostic
therefore needs the pinned π0.5 implementation. TorchCodec0.16.0 is the existing
DeltaAI decoder version, being installed only in the isolated environment.

Frozen diagnostic, CPU-preflight validated, not yet GPU-tested:

- `stage.py`: authenticated encrypted cluster-session stream with no checkpoint
  or dataset file written on the Mac. **Explicit transfer approval received.**
- `prepare.py`: isolated path adaptations and constructor that immediately
  restores trusted full production state, skipping a redundant pretrained load.
- `benchmark.py`: separate processes per variant; same60004-update snapshot,
  optimizer/scheduler/two-rank RNG/cursor; actual decoded anchors; global32,
  micro16/rank,10warmup and200timed updates; loss/all805gradient checks and15%
  actual GPU/conservative node RSS headroom. No checkpoint IO in timed rate.
- `candidate.py`: CPU state tokenization and side-stream CUDA transfers from
  the completed Palmetto experiment, without production edits.
- `sweep.sh`: repeated controls plus CPU tokens, prefetch,12workers, autocast
  weight caching, and disabling activation recomputation. Per-candidate failures
  are preserved and controls still run. Maximum14minutes/candidate.

Actual source snapshot SHA:
`09c0466b8809bc8e5f0dc248da135bcb186c64bb0759557def0bfdaee9119e6c`,
60004updates,1920128samples,epoch0,cursor1920128, canonical π0.5 KR17782830.
Snapshot and audited dataset remain on Palmetto. Existing DeltaAI DOM data would
be checked by the original per-episode parquet/video hashes during decoding.

Pinned implementation imports, actual two-camera train/validation decoding,
BF16 cache and optimizer continuation CPU contracts, two-rank coordinated flags,
and the scheduler test-only allocation check all passed. Test-only3429509 is not
an actual job. The final manifest SHA is
`591ae5d7af07b30e93a73ceaaadbd90790781cd598dfbbef139c34f7376d8354`.
After the copy and checksum complete, `interactive.sh` submits only once and
persists the actual allocation ID. No script restarts a production run.

Interpretation: compare variants within the same DeltaAI allocation. GH200120GB
and PyTorch2.12 differ from the Palmetto H200150GB/PyTorch2.7 setup; absolute
cluster differences are not evidence of a code speedup. Failed correctness or
headroom checks make a candidate ineligible regardless of throughput.

Latest resource audit: physically fitting GPU pairs are DRAIN/RESERVED/PLANNED. No eligible healthy unreserved fit was found. Administrative states were not changed. Current3429682 remains pending Priority; a90minute minimum backfill window is allowed within the two-hour cap. `collect.py` reads scheduler/results only, refuses mixed job IDs, and compares eligible candidates only after both controls complete. Page220 exact readback verified.
