# DeltaAI four-node KRVLA + velocity + causal phase gate

User requested a separate four-node experiment in addition to the queued two-node DOM+velocity job. Submitted **3382533** on October14,2026 at17:57:25EDT; live scheduler confirmed PENDING/Priority. DOM3381947 remains a separate queued run. No Palmetto jobs were changed.

## Frozen experiment

- Four nodes, four GH200s/node;32CPUs and384GB RAM/node;48hours;regular ghx4 queue;no dependency.
- Named variant `krvp100`: coordinate residual flow, object velocity conditioning, `causal_rules` phase gate.
- Microbatch22/workers4/prefetch2 per GPU, accumulation1/global352, BF16, pinfalse/fusedfalse/random frames.
- LR3e-5,seed42,horizon20,full fine tuning,100epoch cap. Validation h20/action_mse needs0.5%relative improvement;patience3,min5epochs.
- Fresh original pretrained initialization; does not resume cancelled Palmetto phase run or DOM.
- Exact approved Palmetto split:197198train episodes/22370201frames;validation2015/test2011;all nine approved invalid episodes excluded. Original normalization/causal phase cache retained. Reuses804900-file audit; pinned weights rehashed at deployment.
- Source `/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261012-video-clean/dom100`, SHA256 `f991015a229476400e0fc772177af33759d1bc3f23901381e2f83c7faaeab921`. Frozen model and phase-gate implementation match local model/gate files. Additional launch/diagnostic adapters are separately hashed.
- Manifest `5d9f8004d1a7f210790e1d2b1933960b3bb95ada7b70da422278cec391a3e846`; initial recipe `b2d53d1941edfaea02221331376f2034d37222230332cd6f27ecf0edcd8ed972`. Actual successful runtime approval will change the recipe fingerprint.

## Startup and continuation

The same48hour allocation runs a single readiness attempt capped at15minutes plus cleanup: fresh16-rank checkpoint/RNG/uneven-tail/future-input isolation smoke, then fresh200-update full-data process (70400samples;20warmup/180measured). It requires eight checkpoint saves, validation, exact all-rank prediction/RNG restoration and15%GPU/per-nodeRAM headroom. The benchmark explicitly checks residual+velocity+causal-gate config. No DOM benchmark is reused as approval for this variant. Failure exits without production. Success records actual runtime/recipe approval, then starts a fresh production process in the same allocation. Requeues reuse successful readiness, with exact16-rank optimizer/RNG checkpoints.

Checkpoints every25updates through1000,then30minutes,epoch/validation,and20minutes before expiry. Signals reach leaf training workers on every node; continuation uses the original all-rank checkpoint helpers and requeue.

Family: `/work/nvme/bivv/aparameshwaran/dynamicvla/krvp-scale-ddp-20261014`; production `runs/krvp100`; logs `logs/train-3382533.out/.err`. Setup: `/work/nvme/bivv/aparameshwaran/dynamicvla/setup-krvp-4node-20261014`; receipts `status.json`, `production-submission.json`, `scale-check-3382533-16.json` (when allocated).

No actual16-GPU throughput/readiness/production checkpoint/W&B ID exists while queued. Local `evidence/` stores observed submission/deployment/scheduler receipts. `submit.py` is idempotent and blocks blind retries after unknown outcomes. Do not edit frozen worker/controller artifacts in place; prepare a new family for changes. Existing hourly read-only register refresh now tracks both DeltaAI runs independently.

Checked launcher Python/shell syntax, parent artifact/weight/source hashes, scheduler test-only admission, successful real submission and live request resources. Actual GPU checks are scheduled within3382533.
