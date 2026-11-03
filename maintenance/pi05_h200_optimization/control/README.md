# Guarded two-H200 continuations for the canonical pi05 pair

The user explicitly requested H200 training continuations on October20. Both
GPU200-update profile receipts passed and measured sustained2.78x/2.88x current
A100 reference rates. This separate controller never edits submitted source.
A CPU-only verifier revalidates exact completed GPU receipt SHA/source/data/native
implementation/recipe/runtime/hardware and paired measurements; it creates bound
production releases, not new GPU readiness. Every production owns afterok that
verification and still passes the frozen actual-GPU native identity gate.

Queued H200 jobs leave A100 training running. Only the corresponding replacement
with actual2H200 allocation, CUDA and frozen release checks may signal that exact
A100 job. It proves a new complete saved production checkpoint, detects changed
allocation/restarts, cancels only that saved original/pending automatic requeue,
waits for terminal accounting, freezes immutable checkpoint SHA/metadata and
original W&B ID, then restores optimizer/scheduler/per-rank RNG/progress/data cursor
under the original30k TOTAL cap. An A100 already completed at allocation causes
the H200 job to release without repeating training. Own target saves take priority
on requeue. Expected missing/incompatible state fails closed. No readiness weights
or fresh initialization are accepted; no weights or data are stored on the Mac.

Do not invoke submit.py for recorded jobs unless state and pending markers have
been reconciled. It persists each returned scheduler ID before the next submission
and writes discoverable hardware_candidates.json. Unknown submission outcomes keep
a pending receipt and stop; definitive rejection is preserved. Old six H200 copies
stay retired. SmolVLA,VLASH,workstation,DeltaAI and all other jobs are out of scope.
