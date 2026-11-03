# Older checkpoint cleanup on Palmetto CPU

User authorized deletion of older Palmetto checkpoint copies on October26 and
recommended CPU execution. CPU-only job18232434 uses2 CPUs,32GiB and no GPUs.
Exact41-file plan totals806,490,103,923 bytes. No glob deletion is performed.
The six allowlist/protection contracts passed; scheduler test-only passed.

The worker first validates all seven retained own latest model/optimizer/scheduler/
two-rank RNG/identity/progress states. It refuses current or symlink-target paths,
changed candidate size/inode/mtime, non-independent copies, regressed protected
progress or any historical family with a current scheduler command. It hashes each
older file, rechecks state, unlinks only that exact file, and immediately persists
its receipt. On restart it reconciles already removed paths against receipts.

Current latest checkpoints, current parent checkpoints, source, logs, JSON metadata,
datasets and pretrained assets remain protected. CURRENT_CHECKPOINTS.json in the
Palmetto Dynamic_VLA root lists canonical latest paths without duplicating weights.

The job COMPLETED exit0 in1h14m16s. All41 files totaling806,490,103,923bytes
were removed; deletion-receipt.json has complete:true. All seven current
checkpoints remain protected and their existence was reverified on October27. Do not blindly rerun
submit.py: submission.json records the existing job. Retrieve the receipt from
/scratch/aparame/Dynamic_VLA/checkpoint-cleanup-20261026/deletion-receipt.json.
No training code, current jobs, other clusters or workstation were modified.
