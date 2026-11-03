# DOM+velocity full-dataset continuation,2026-11-02

Submitted job3508313,aparameshwaran,bivv-dtai-gh;2nodes x4GH200,48h with checkpoint/requeue.
Full audited training split197198episodes/22370201frames;global352;microbatch44;BF16;LR3e-5.
Resume latest218468updates,76900427samples,3.437627896epochs. Preserve10total-epoch stopping cap,
100-epoch original cosine schedule and existing validation early stopping(min5,patience3,relative0.5%).
Latest and best are both immutable complete generations with native atomic pointers and protected pruning.
The physically staged save/resume code was inspected and its hashes frozen in launch-lock.json.
Starting latest and best generations were preserved via hardlinks under remote starting-checkpoints/.
No historical source,manifest or recipe was edited. New epoch-control bootstrap is scoped to this job.
The same frozen family/run directory is continued to preserve exact checkpoint identity.
Allocated-node preflight verifies full model/state file presence,optimizer,scheduler,8-rank RNG,
provenance,profile and dataset receipt,then the native trainer strictly restores the latest state.
Remote launcher:/work/nvme/bivv/aparameshwaran/dynamicvla/setup-domv-resume-20261102.
Run:/work/nvme/bivv/aparameshwaran/dynamicvla/domv-scale-ddp-20261014/runs/domv100.
Tracking is offline; no W&B API access is attempted. Queue at submission:PENDING/Resources.
