## Latest verified status

October20 7:08AM EDT: both current pi05 productions are verified training. Baseline17719993 saved1,615updates/51,680samples and logged1,860updates; KR17720948 saved1,077updates/34,464samples and logged1,480updates. Own latest complete production checkpoints now exist for BOTH runs and take precedence for every future continuation, including optimizer/scheduler/per-rank RNG/progress/data cursor under the original30k total budget. Both own GPU readiness and CPU verification receipts remain approved. Do not rerun launch/release helpers or resubmit healthy recorded jobs. See current-production-status.json and the central registry; the initial launch summaries below are historical.

# Pi05 completed-readiness scheduler release

Actual two-A10080 profile17716903 COMPLETED with approved matching source/data/
runtime/hardware and checkpoint/RNG/inference evidence. Slurm rejected its stale
afterok dependency after purging the completed job from live retention. Preserve
the test-only rejection in pi05_baseline-expired-dependency.json.

CPU-only verification17719992 rechecked original GPU receipt SHA256, physical
source/implementation/preparation/recipe hashes, runtime identity,20-update strict
reload/inference flags,15% GPU/node memory headroom and original profile accounting
COMPLETED exit0. It completed18seconds exit0; no new GPU test is claimed.
Production17719993 has afterok17719992 plus unchanged native matching GPU-readiness
gate. Both job IDs were persisted immediately; hardware_candidates.json records
original GPU profile and release verifier separately. Production is PENDINGPriority.

Remote control /scratch/aparame/Dynamic_VLA/cross-vla-krfm-pi05-release-20261020.
No Python source in the active pi05 campaign was edited. KR17716904 is still
profiling; do not release KR until its own full GPU readiness and accounting pass.
The frozen verifier is generic per-recipe, with separate immutable config/receipt
paths. release_from_completed_profile.py is an orchestration receipt for baseline;
reconcile current submissions/pending state before any reuse or adaptation for KR.

KR release is also complete: actual-GPU profile17716904 passed23m09s exit0;
CPU-only revalidation17720947 passed19s exit0, with own exact receipt/hash/runtime
checks and no new GPU test claimed. Production17720948 owns afterok17720947 and
matching native GPU readiness; it is PENDINGPriority. Both completed GPU profiles
are absent from live Slurm retention. Preserve both verified.json files and
rejections; do not re-run orchestration for already-recorded productions.
