# Conditional pi05 A100 to H200 migration control

User authorized migration of both pi05 baseline and KR only if clearly faster.
One bounded profile17737992 is submitted. Frozen source7ced7acae17243f8760e5ece930c49091c0329b868711a73b89874028d6529d7
lives at /scratch/aparame/Dynamic_VLA/cross-vla-krfm-pi05-h200-opt-20261020.
The local frozen source is experiments/cross_vla_krfm_pi05_h200_optimized.
Do not add or edit Python files in either frozen source directory: all .py bytes
participate in readiness identity. Put new handoff orchestration in this control
directory and a separate remote cross-vla-krfm-pi05-h200-control-20261020 directory.

The profile sequentially tunes baseline and KR,then200-update strict state/RNG/
physical inference readiness. Production release requires BOTH variants to pass
own hardware/source/data/runtime/headroom receipts and sustained training rate
at least1.25 times median of20 recent10-update current A100 production intervals.
Use new observations,not the old retired H200 MB1 run or short benchmark ratio.
H200200-update timing includes an intermediate checkpoint; end-to-end readiness
and8-update benchmarks are separate. Persist exact comparison evidence and
readiness SHA256 in production-release/{name}.json using migration.py's schema.
No production was submitted by the profile-only launcher.

Keep A10017719993 and17720948 running while profiling AND while any replacement
H200 jobs wait in queue. Each H200 production must obtain actual2H200 allocation,
verify matching frozen source/runtime/GPU release and usable CUDA first,and only
then coordinate exact corresponding A100 checkpoint/stop. If the A100 run already
completed its30k budget,release that H200 allocation without repeating training.
Original A100 shell scripts trapUSR1 and may requeue themselves; watch new complete
latest.json/stopped.json and scheduler state,then cancel only that saved parent
and any pending automatic continuation. The site previously rejected hold. Do not
kill active training before a complete latest production checkpoint is verified.
No shared source edits. Reconcile uncertain submissions and immediately save IDs.

After the parent allocation is terminal and absent from squeue,freeze immutable
copies of newest own latest.pt/latest.json and SHA256,plus original provenance,
recipe/prepared/core implementation hashes,Torch version,two-rank RNG,optimizer,
scheduler,progress/cursor and original wandb.json. Validate CPU checkpoint contents
with trusted torch.load(map_location=cpu,mmap=True,weights_only=False),then record
resume/{name}.json exactly as train.py and migration.py require. The target H200
hardware is explicit; parent NVIDIA A10080 is not mislabelled as target hardware.
Do not use original cancelled H200 outputs,readiness.pt,logged progress,old r2
parents or fresh initialization. Preserve30k TOTAL cap and original scheduler.
Copy original W&B identity before torchrun; production uses resume=must. The frozen
engine strictly loads model/optimizer/scheduler/per-rank RNG,counters/data cursor,
checks scheduler.last_epoch==saved updates,exercises physical-h20 inference on
restored weights,and restores training RNG before the next anchor. Record actual
resume-migration.json,resume-inference.json,provenance and positive NEW updates
beyond inherited counters. Cross-hardware/microbatch trajectories need not be bitwise.

Original six H200 duplicates stay CANCELLED/deleted;never recreate them. SmolVLA
completed20k and stays untouched without a new user budget. VLASH,workstation,
DeltaAI and unrelated jobs are out of scope. Preserve local history and cancellation
records;omit cancelled rows online. Update registry and Page with fresh guards,
verified receipts/readback,America/New_York times. Existing hourly monitor stays
read-only. No datasets/weights/checkpoints on Mac. If not clearly faster,keep A100
training and report measured result;do not migrate. Delete the temporary profiling/
migration continuation after the final decision or verified checkpointed H200 starts.


R2 repair October20 11:18AM EDT:17737992 failed before model load because isolated staging omitted parquet/video hash manifests. Current profile17740320 uses /scratch/aparame/Dynamic_VLA/cross-vla-krfm-pi05-h200-opt-r2-20261020; local /Users/aditya/Documents/ChatGPT/Kinematic Residual VLA/experiments/cross_vla_krfm_pi05_h200_optimized_r2. Source 562087215a107ad0f5ce81ed35fa914400bbd0616b5a73b241257c2481f93ed7. All eight preparation assets are byte-hash checked; both camera views decoded from train/validation anchors;16 contracts and two-rank test pass. Old attempt and failure receipts remain. Canonical A100 jobs continue. Use updated probe.py. No production submitted and no migration approval yet.


October20 12:11PM EDT: R2 paired GPU profile17740320 COMPLETED51m31s exit0, both full200-update receipts approved. Sustained baseline26.7752/KR27.6556samples/s versus current A100 medians9.6181/9.6007 (2.7838x/2.8806x). Exact paired comparison/receipt SHA/progress in paired-readiness-decision.json. Authenticated socket disappeared during subsequent parent-script inspection, detected12:15PM EDT; no reusable socket, no interactive login. No training/signals/cancels/migration/submission occurred. Last verified A100 parents RUNNING, saved7039/6516 and logged7180/6810. Local handoff_draft guards pass9 pure-CPU contracts; no remote allocation controller or scheduler test claimed. Restore authenticated access, finish/review/freeze/test separate controller, then submit guarded own-allocation-first continuations. Do not stop A100 while replacement queued.


October20 12:33PM EDT: access restored; guarded production continuations17745549 baseline and17745557 KR submitted, each2H200/32CPU/256GiB/48h. Separate CPU-only exact completed-GPU revalidation17745541 COMPLETED18s exit0; both afterok dependencies fulfilled, queuedResources. Original paired GPU profile17740320 remains the actual readiness evidence; definitive expired-native-dependency rejection preserved. Separate frozen control code in control/ (remote cross-vla-krfm-pi05-h200-control-20261020),14 safety/synthetic CPU checkpoint contracts and scheduler preflights passed. Controller SHA db557049687bc22dd638e1f0a02600e36c074494574aa39865cbcf30e33b6ac4. No pending uncertain submissions. Do not rerun submit.py. ParentA10017719993/17720948 RUNNING and remain canonical until actual corresponding H200 allocation/checkpointed handoff. Updated probe.py includes all new jobs, own H200 checkpoint/provenance/resume-inference/stop/completion receipts, handoff artifacts and control errors. Continue monitoring until both actual resumed training starts or parent already-completed bypasses are verified/registered.
