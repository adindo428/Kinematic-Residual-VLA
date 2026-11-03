# H200 restart repairs and exact20k split audit

Root cause: the PI05KR and advanced VLASH baseline approvals/checkpoints bind
NVIDIA H200 NVL. Generic h200 requests sometimes restart on NVIDIA H200 SXM,
changing their exact readiness identity. Source/preparation/recipe/runtime hashes
and Torch2.7.1+cu126 still match; CPU-calculated NVL identities equal the original
approvals while SXM identities differ. No check was weakened or bypassed.

Separate immutable control root: restart-repair-20261029-r2. CPU18661723
COMPLETED exit0 in59s and validated both actual own newest production checkpoints,
model/Adam state/scheduler/two-rank RNG/counters/cursor, all prepared source/data
locks, exact20k membership and approval identities. Eight checkpoint contracts
passed locally and remotely. GPU continuations18661724 PI05KR and18661744 VLASH
baseline are PENDING Priority. They request2H200NVL/32CPU256GiB48h, run exact
allocated-CUDA hardware checks and unchanged original training sources, and
resume301262/364331 updates to original812114/782110 total caps. No extra ten
epochs, optimizer reset, new LR warmup or readiness weights are introduced.

Site phase constraints were definitively rejected in test-only before any job
submission; receipts preserved. Supported graphics-H200 restriction plus exclusion
of unverified node0102 selects known NVL nodes0107/0110. Actual two-device CUDA
H200 NVL identity remains mandatory. No package or active source was edited.

Site policy also rejects in-place user job resource edits. Only idle standby
17931273/17931276 were cancelled after successful replacement script preflights.
New standbys18661943/18661944 retain exact afterany17859073/17859078 dependencies,
matching Intel/SXM-H200 resource constraints, original guarded resume checks,
unchanged total budgets and completed-run skip. Running parents are unchanged.
Test-only IDs are never real submissions. Returned IDs were persisted immediately;
reconcile recorded intents and queues rather than blindly rerunning launchers.

All seven current cluster models use splitSHA98436815f85fbfc78d48e4332e8603684f6bae1620400b181767bab1c5736d88,
20000 training episodes/2406773 anchors. Smol pair and DOM+velocity completed10
subset epochs. PI05 baseline and VLASHKR are already training; their standbys
remain queued. These are continuations with historical pre-exposure, not fresh
20k-only training claims. The historicalPC KRV100 also transitioned to the same
split at globalepoch3. User's currentW&B664747c0 is a separate running fresh20k
DOM100 baseline at logged8.02epochs. Local checkpoint progress is not inspected;
whether KRV100 follows it on the PC remains a user clarification.

Evaluators are now local-PC-only at user direction. No more Palmetto simulator
evaluation jobs were launched. The PC and existing training were not modified.
No datasets/checkpoints were stored on the Mac.
