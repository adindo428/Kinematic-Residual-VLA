# Pi05 A100 memory diagnostic

The native fused AdamW attempt17713148/50 completed all benchmark updates but
still exceeded the unchanged15% GPU headroom requirement. Production17713149/51
was cancelled without starting; no pi05 production checkpoint exists.

Single two-A10080/32CPU/224GiB job compares identical decoded pi05 baseline
microbatch4/global32 under default allocator, startup-cache trim, and expandable
allocator plus trim. Report actual allocated/reserved/current/peak memory after
model load, DDP construction, backward and optimizer steps; preserve first-step
and steady-state evidence. This is a diagnostic, not readiness or production.
No headroom threshold, recipe, model, precision, split or normalizers change.
No active SmolVLA/VLASH or original H200 source/job is modified.
Only submit a new frozen recovery after these results identify a defensible fix.
