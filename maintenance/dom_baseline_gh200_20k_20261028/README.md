# Missing DOM baseline: 20k episodes on DeltaAI

Submitted 2026-10-28 as job **3479152**, account `bicn-dtai-gh`, user `anayak2`.
Variant `dom100` has both neural velocity conditioning and KR disabled.
The existing VLASH, pi05, SmolVLA, DOM+velocity and KR runs are separate.

Final verified state: **released, PENDING/Priority**. The52,119-file old-account
transfer completed successfully;66,045 parquet/video files,4 metadata files and
22,015 phase-cache files passed checksums. Actual GPU readiness has not run yet.
Frozen manifest: `7e5e4b88245380d326ce4266ab0e6f6eae2a04e6a3d191ecd242fb8abc208c93`.
Recipe: `b46392ea6e06c24eb35c961866d7550eafc7817b37d1a5947fd06f3db1cdb6ab`.

Remote root: `/work/nvme/bicn/anayak2/dynamicvla/dom100-20k-gh200-20261028`.
The actual training output is `family/runs/dom100/` under that root.

The split is the established 20k fine-tuning subset: 9,000 pick, 9,000 place,
2,000 long horizon, 2,406,773 training frames. The original 2,015 validation
and 2,011 test memberships are preserved. Only training and validation files
are staged; test episodes are reserved. Every required parquet, camera video,
metadata file and phase-cache file is checked against the preserved audit hashes.
Metadata and 66,045 parquet/video files must finish verification before release.

This is a fresh fine-tuning run from the same preserved released DOM initialization
as the earlier DOM family, SHA256
`e3da8b1e0326d7a3ee6873f7fa17cdf7015201251901be5107377cb851ce3af8`.
It does not resume the domv100 training state. The released initialization has
prior DOM pretraining; the 20k count describes fine-tuning data, not lifetime
data exposure. Its native state/action normalizers are retained.

Ten complete subset epochs, global batch352, LR3e-5, AdamW betas(.9,.95),
eps1e-8, weight decay1e-10, 1,000-update warmup and ten-epoch cosine decay.
Prediction/execution horizon20 at25Hz; observations[-2,0]; full fine-tuning;
seed42; frame sampling. No early stopping before the ten-epoch cap.
The native sampler includes each epoch's final partial global batch, for
68,380 updates and24,067,730 real samples in total.

Two nodes, four GH200s/node,32CPU/node,384GiB/node,24-hour allocation with
exact checkpoint continuation and requeue. BF16, per-rank microbatch44,
accumulation1, four workers/rank, prefetch2, no pinned memory or fused optimizer.
This candidate comes from successful historical profile3381859 (DOM+velocity):
373.45samples/s,46.38GB maximum reserved GPU memory and84.93GB maximum summed
process RSS per node. This historical result is a candidate reference; it does
not approve the new DOM baseline or establish its throughput.

Launch chain:

1. Hub downloads were stopped at the user's request. `prefetch.py --plan`
   verified13,930 existing files and identified52,119 files still needing transfer.
   The current `prefetch.py` performs planning/checksums only and contains no
   Hub download path.
2. `transfer_data.py` relays those remaining files from the old account's
   `/work/nvme/bivv/aparameshwaran/dynamicvla/huggingface/lerobot/hzxie/DOM`
   through the two existing encrypted SSH connections into the new account's
   cache. Dataset bytes are streamed through the Mac without a local dataset copy.
   It then runs full verification and invokes `finalize.py --job-id 3479152`.
   That step freezes source/recipe/data hashes and releases only the verified job.
   Any transfer or verification failure leaves the submitted job held.
3. The allocation runs a fresh two-update uneven-tail/RNG smoke followed by
   a200-update decoded benchmark, checkpoint writes25..200, strict reload,
   post-training validation, current-velocity poisoning and15% GPU/RAM headroom.
   Readiness checkpoint weights are separate from production initialization.
4. Only a passing baseline profile enters the frozen native trainer. Each
   restart restores its own model/optimizer/scheduler/RNG and global sampler cursor.

Watch local `evidence/dataset-transfer.json` during copying, then remote
`finalization.json`, `prefetch-progress.json`, `status.json`, `readiness.json`,
`logs/train-3479152.out`, and `family/runs/dom100/latest.json`.
W&B runs in offline mode with local tracking files preserved. Automatic approval
review rejected the attempted read of the external W&B project; it was not retried.
No remote W&B access is required for this launch.

Local validation: launch Python files parsed and shell scripts passed syntax checks;
the copied ARM environment imported the DOM model and TorchCodec successfully,
and the native source digest matched `f991015a...`. Slurm accepted the request.
Actual GPU readiness and production progress remain pending allocation.
