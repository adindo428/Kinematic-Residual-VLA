# Pi05 H200 throughput tuning and conditional checkpoint migration

Fresh isolated attempt. The six original duplicate H200 productions were cancelled
and their outputs deleted by the user; this directory does not recreate those runs.
One bounded two-H200,32CPU,256GiB allocation benchmarks microbatches1/2/4/8/16
and4/8/12 workers for pi05 baseline and KR separately. Global batch32,30k total
updates,LR2.5e-5,AdamW(.9,.95),eps1e-8,WD.01,1000warmup,cosinefloor2.5e-6,
seed42,horizon20 and the audited DOM split/weights/train-only normalizers stay fixed.
FP32 parameters/moments,BF16 cache disabled,fused AdamW,pinned/nonblocking transfers,
gradient bucket views and per-anchor padded objective match the current A100 code.

The benchmark records first-step/startup memory as well as steady peaks. A chosen
runtime must retain15% GPU/conservative node RAM headroom and pass200 decoded
updates with coordinated checkpoint,strict optimizer/scheduler/per-rank RNG reload
and physical20x7 inference reload. Sampled CPU busy cores and GPU activity are
reported independently from throughput; checkpoint-inclusive training timing and
end-to-end readiness timing are separate. Short benchmark ratios use the same-job
optimized MB1 control,not the retired H200 profile or current A100 production rate.

The user authorized migration only if clearly faster. Require both variants to
show at least25% sustained throughput improvement over verified current A100
production intervals,with the full readiness/memory checks passing. The current
A100 runs continue while profiling. No production is submitted by submit_profiles.py.
A production release file must bind the exact GPU receipt SHA and measured advantage.
Only then coordinate checkpoint/stop of the two A100 runs,verify newest complete
saved production states,freeze immutable migration manifests and submit matching
H200 continuations. Restore model,optimizer,LR scheduler,per-rank RNG,counters,data
cursor and original W&B IDs with the same30k TOTAL budget. Never readiness weights,
logged-only progress,or fresh initialization. A missing expected checkpoint fails
closed. Cross-hardware/microbatch kernels may change numerical trajectories.

Shared code/environment/prepared metadata reuse verified bytes from
/scratch/aparame/Dynamic_VLA/cross-vla-krfm-a100-pi05-cache-20261020.
No datasets/weights/checkpoints are stored on the Mac. Do not modify any active
A100 source or recreate cancelled H200 duplicates. SmolVLA remains completed and
unchanged; any extra SmolVLA budget needs clarification from the user.


## R2 staging recovery

Profile17737992 failed before model loading because the isolated folder omitted
parquet_hashes.json and video_hashes.json. Its source and failure receipts remain
untouched. This fresh attempt copies and SHA-verifies all eight required audited
preparation files from the healthy canonical A100 campaign and verifies dataset
construction plus decoded train/validation anchors before scheduler submission.
The batch launcher rechecks the frozen preparation manifest before using GPUs.
Only the unique profiling job name changes Python source; training, objectives,
budgets, 200-update readiness,15% headroom and25% migration criterion stay fixed.
Canonical A100 jobs continue while profiling. No production submission.
