# Existing DeltaAI dataset cleanup

User chose the existing DeltaAI dataset over the slow archive transfer and approved excluding all nine invalid training episodes on both servers. The stopped SCP stream wrote a separate tar; the existing dataset was not overwritten.

Existing dataset: `/work/nvme/bivv/aparameshwaran/dynamicvla/huggingface/lerobot/hzxie/DOM`.
New final family: `/work/nvme/bivv/aparameshwaran/dynamicvla/domv-audited-20261012`.
Workflow: `/work/nvme/bivv/aparameshwaran/dynamicvla/setup-existing-data-domv-20261012`.
Source remains the previously verified video-clean snapshot (SHA f991015a229476400e0fc772177af33759d1bc3f23901381e2f83c7faaeab921).

freeze_family.py uses the actual sixteen completed Palmetto audit shard receipts and a failure-only projection of its three short-video records, extending the previous six exclusions. No video files are physically deleted. Only train membership and corresponding frozen hash maps change. Validation/test, fixed anchors, initialization and normalization remain frozen. Final split SHA bbeca518e7f6ee85c39e5863a4d6745c36de53f13b8831595ca36c304ff9950c, train197198episodes/22370201frames, validation2015,test2011.

verify_existing.py checks every retained video, label parquet and phase cache SHA256 against the paired audited Palmetto maps, checks published metadata byte hashes, frozen artifacts and initialization. It does not repeat a full decode audit. Any mismatch blocks launch; selectively repair only failed files against pinned provenance. Verification now runs in the same four-GH200 allocation as readiness and training, using16 CPU threads.

On2026-10-13 the user requested one allocation to avoid successive queue waits. `combined.sbatch` requests four GH200s on one node for48hours, CPU32/RAM384GB. It runs `prepare_combined.py`, then execs the unchanged training launcher without releasing the allocation. `prepare_combined.py` verifies all804900 retained metadata/video/parquet/phase files (at most60minutes), runs exactly one readiness candidate (at most15minutes), and uses coordinate.candidates/select helpers to record real final-split profiling approval. Outer preparation limit80minutes. A failed check exits before optimizer training. The 48hour wall time includes startup work.

`launch_combined.py` stops the waiting old coordinator, cancels pending separate verification3370057, submits `combined.sbatch` once with a persisted intent/receipt, then cancels unstarted held3367506. `production-submission.json` aliases the actual combined receipt so the obsolete coordinator cannot submit additional jobs. No Slurm dependency and no subsequent queue wait between verification/readiness/training. No Palmetto jobs are changed.

On48hour checkpoint requeue, the completed byte-verification receipt is reused only for the same manifest/read-only cache and checked against its saved SHA256. Runtime approval is reused only for the same actual GPU/PyTorch/CUDA configuration and frozen recipe. The unchanged training launcher handles signal forwarding, checkpoints, persistent W&B identity and Slurm requeue. Recipe, model/training source f991015a..., split and normalization remain unchanged. No dataset files are deleted or bulk recopied.

100epochs cap, batch352, LR3e-5, horizon20, BF16 random frames seed42; early stopping h20/action_mse after minimum5epochs/patience3/no>=0.5% relative decrease. Checkpoints every25updates through1000, then30minutes plusepoch/validation/wall limit. W&B remains online for production.

The obsolete archive-transfer continuation remains PAUSED. Do not resume it or copy the archive to the Mac. Read status.json, verified.json, verification-progress.json, coordinate.log and submission receipts before any retry; never duplicate uncertain submissions.
