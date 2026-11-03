# Pi05 A100 fused optimizer recovery

Only pi05 baseline and KR are replaced after optimized profiles17709583/85
completed decoded updates but failed the unchanged15% GPU headroom gate.
Old production17709584/86 are cancelled; no saved production checkpoints exist.
Preserve old logs and receipts. Active SmolVLA/VLASH and original H200 campaigns
are untouched.

Use native fused AdamW instead of default CUDA foreach, preserving FP32
parameters/moments, AdamW(.9,.95),epsilon1e-8,WD.01,LR2.5e-5,batch32,warmup1000,
30k total updates,horizon20 and all audited preparation/weights. Native fused
and unfused equations, optimizer/scheduler save-and-continue are checked on CPU
and each actual GPU rank before model loading. Numerical trajectories may differ.
GPU outcome: both variants completed all benchmark updates, but reserved peaks
remained76.8–78.8GB on85.1GB devices. No candidate passed15% headroom; profiles
17713148/50 failed and dependent production17713149/51 was cancelled without
starting. The separate memory diagnostic17714677 investigates allocator state.
PyTorch2.7 docs: https://docs.pytorch.org/docs/2.7/generated/torch.optim.AdamW.html

Two A10080,32CPUs,224GiB per job. Own afterok and identity-matching readiness
remain mandatory. Never lower15% headroom or use readiness weights as production.
Any subsequent resubmission must restore latest.pt with optimizer,scheduler,RNG,
progress and data cursor; if expected latest.json exists but weights are missing,
fail closed. Current variants explicitly have no saved production checkpoint.
No datasets or weights are stored on the Mac.
