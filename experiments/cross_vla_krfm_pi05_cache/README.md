## Latest verified status

October20 7:08AM EDT: both current pi05 productions are verified training. Baseline17719993 saved1,615updates/51,680samples and logged1,860updates; KR17720948 saved1,077updates/34,464samples and logged1,480updates. Own latest complete production checkpoints now exist for BOTH runs and take precedence for every future continuation, including optimizer/scheduler/per-rank RNG/progress/data cursor under the original30k total budget. Both own GPU readiness and CPU verification receipts remain approved. Do not rerun launch/release helpers or resubmit healthy recorded jobs. See current-production-status.json and the central registry; the initial launch summaries below are historical.

# Pi05 A100 autocast-cache recovery profiling

Only pi05 baseline and KR are affected. Native fused AdamW alone failed the
unchanged15% GPU headroom gate. Diagnostic17714677 completed three decoded
MB4/global32 variants: control77.43GB reserved/74.50GB allocated peak; initial
cache trim77.42/74.49GB; expandable allocator+trim74.72/74.34GB, on85.09GB GPUs.
Allocator cleanup therefore does not suffice; backward activation/cast storage
must be reduced. All failure/diagnostic receipts remain preserved.

Test native BF16 autocast with its weight cache disabled, expandable allocator,
and startup-cache trim. FP32 parameters/moments, native gradient checkpointing,
all model trainables, same paired official recipe and audited data are preserved.
Both variants now passed full actual-A10080 readiness with71.80GB reserved
GPU peak and15.62% headroom, including startup/reload/inference peaks. CPU and
actual-rank tests compare cached/uncached BF16 checkpointed forward and gradients,
and native fused AdamW equations and checkpoint continuation. Full20-update
readiness includes startup peak, coordinated saves, strict optimizer/scheduler/RNG
and physical inference reload, retaining15% GPU/node RAM headroom.
PyTorch cache option: https://docs.pytorch.org/docs/2.7/amp.html

Submit profiles only first. Production submission requires a measured approved
receipt and scheduler COMPLETED profile, then owns afterok and identity-matching
runtime/source/data/hardware gate. Every job exactly2 A10080/32CPU/224GiB.
Baseline and KR have no production checkpoint and explicit pinned initialization;
readiness weights are never used for production. Future continuation always uses
own latest.pt with optimizer/scheduler/RNG/progress/data cursor, same30k total cap.
Missing expected production weights fail closed. Active SmolVLA/VLASH, workstation,
DeltaAI and original queued H200 campaigns remain untouched. No weights or data
are copied to Mac.

Current submitted production: baseline17719993 RUNNING with matching provenance,
90loggedupdates/2880samples at9.59samples/s; KR17720948 PENDINGPriority at5:27AM
EDT. Both actual GPU profiles17716903/04 COMPLETEDexit0. Slurm purged their live
dependency records; separately verified CPU releases17719992/17720947 completed
18s/19s exit0, revalidating exact original GPU receipts,source/data/runtime and
successful accounting. Productions own afterok their respective verification
plus native matching GPU-readiness gates. No active source was edited. IDs and
hardware_candidates.json are persisted. Do not resubmit recorded jobs. Baseline
has no saved production checkpoint yet; future continuation uses its own latest
complete state, never readiness weights or logged-only progress.
