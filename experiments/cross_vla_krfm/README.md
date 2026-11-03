## Retired duplicate hardware campaign

The user cancelled all six H200 production copies on October20 and requested deletion of their separate run outputs. The six progressed/completed A100 baseline/KR runs are canonical. Do not submit, resume or recreate these H200 copies. Their output folders and production logs are deleted; shared preparation, environment, pinned weights and small audit/readiness receipts remain because the canonical campaign reuses them. See maintenance/training_run_register/deduplication-20261020.json. The launch descriptions below are historical.

# Cross-backbone KRFM on audited DOM

This campaign implements matched baseline/KR pairs for π0.5, SmolVLA and
VLASH on π0.5. It does not reproduce the original models' pretraining, and
training completion alone does not fill DOM success-rate cells in Table 1.
Each trained checkpoint still requires the released 1,800-trial DOM suite.

## Source-backed protocol

Sources inspected October 19, 2026:

- DynamicVLA v2: https://arxiv.org/html/2601.22153v2 (Sec5.1, AppendixB/D).
  Its baseline rule is released pretrained weights, common DOM data, and
  official recipes where available. It does not give a complete per-baseline
  DOM hyperparameter/step table. Its own 32-A100, batch40/GPU, three-stage
  pretraining/midtraining/posttraining schedule is not a downstream baseline
  fine-tuning recipe and is not claimed here.
- π0.5: https://arxiv.org/html/2504.16054; official implementation
  https://github.com/Physical-Intelligence/openpi at
  `215abfb217dbac7d5f1273282331b9b1866c0479`. This campaign uses the official
  Hugging Face/Physical Intelligence PyTorch port in LeRobot0.4.1, not π0.
  Native generic fine-tuning presets differ from OpenPI's LIBERO-specific
  config (which uses batch256 and a different schedule/EMA). DOM is not LIBERO.
- SmolVLA: https://arxiv.org/html/2506.01844;
  https://huggingface.co/docs/lerobot/smolvla. LeRobot0.4.1 commit
  `a5b29d430105f5235eb05bbf2db5a0d747a869d6` supplies native architecture,
  preprocessing, expert-only fine-tuning and optimizer defaults. The documented
  downstream example supplies the20k-step/batch64 budget.
- VLASH: https://arxiv.org/html/2512.01031 (AppendixA/TableV andC);
  https://github.com/mit-han-lab/vlash at
  `22cbabfee0f57874987c75a35a7dac129e695db0`. It is future-state-aware
  training/inference on a backbone, not a separate pretrained VLA.
  Use π0.5, random uniform delay0..3, no action quantization, no LoRA,
  no optional state-conditioning architecture modification. Shared-observation
  optimization is disabled: this is the paper's ordinary random-offset path,
  not its efficiency-study batch expansion. The six-dimensional DOM pose is
  rolled forward using the first six channels of the previous absolute command;
  the action-only gripper channel is not appended to robot state.

| Pair | Effective global batch | AdamW peak LR | Weight decay | Clip norm | Updates | Final LR |
|---|---:|---:|---:|---:|---:|---:|
| π0.5 ± KR |32|2.5e-5|.01|1|30,000|2.5e-6|
| SmolVLA ± KR |64|1e-4|1e-10|10|20,000|2.5e-6|
| VLASH(π0.5) ± KR |32|5e-5|1e-10|1|30,000|2.5e-6|

All use AdamW betas(.9,.95), eps1e-8, cosine decay,1,000 warmup updates,
seed42, two H200s, BF16 and accumulation preserving their effective batch.
No inheritance of the old DynamicVLA batch352/LR3e-5/100-epoch stopping rule.
Native vision/expert architecture and per-model image preparation are retained.
SmolVLA freezes the pretrained VLM/vision branch per native expert-only tuning;
π0.5 and VLASH fine-tune all policy weights without LoRA.

DOM adaptations, which must be disclosed rather than called exact original
paper reproductions: two released DOM cameras; pose6/action7; delta pose
relative to current/rolled-forward robot reference; 20 predicted/executed
25-Hz actions; same audited retained episodes and grouped train/val/test split;
validation on256 deterministic held-out anchors at epoch boundaries/end.
The data contain no gripper observation, so none is invented. The native
single-frame inputs are retained; DynamicVLA's two-time/four-image encoder
construction is not imposed on another backbone.

Normalization uses a frozen, deterministic uniform sample of200k training
anchors, their valid20-action chunks and pose references over delays0..3.
This is an estimated train-only normalization, not the original pretraining
statistics. π0.5/VLASH use1%/99% affine quantiles; SmolVLA uses mean/std.
Each pair shares exactly the same statistics. The source manifest records
this deliberate DOM adaptation. Existing DynamicVLA normalization is not
silently reused across different pretrained embodiments/action conventions.

KR is the existing centered Cartesian coordinate residual: raw target minus
(v_current - training_velocity_mean)*(offset+j)/25 in XYZ, zeros for rotation
and gripper. Model loss is still its native reverse-time flow matching.
Normalization is applied after residualization; inference inverses normalization
and adds exactly the same physical drift once. Current simulator object velocity
is privileged analytic information and must be labeled as such in evaluation.
It is not fed to any neural prefix. No phase gate or neural velocity extension
is included. VLASH uses shifted robot commands but never future images/object
velocity. The paired baseline has KR disabled with everything else identical.

## Execution and gates

Remote root: `/scratch/aparame/Dynamic_VLA/cross-vla-krfm-20261019`.
The parent is `/scratch/aparame/Dynamic_VLA/dom100-audited-20261012`.
Data and initialization remain on Palmetto, not the Mac. Separate virtualenv
and pinned VLASH checkout leave existing training environments untouched.

`setup.sbatch` freezes copied split/hash receipts, train-only statistics and
immutable HF model/tokenizer revisions plus weight hashes. It reuses the
completed dataset audit; every accessed parquet/video is independently checked
against the inherited byte hashes. Its first attempt17668622 exposed the
pose6/action7 mapping mismatch; corrected setup17668675 preserves that record.

Every pair member has its own profiling job, readiness approval, production
job and output. A profile executes20 full effective-batch decoded optimizer
updates on two actual H200s, checks finite gradients and weight change,
valid20x7 physical-coordinate inference, strict checkpoint/optimizer/scheduler
load, exact per-rank RNG restoration, deterministic inference reload and at
least15%GPU/node-process-memory headroom. Eight CPU contract tests cover residual
roundtrip, zero drift, offset timing, untouched non-XYZ channels, padding,
6D state rollforward, rank-shard/resume consistency, native loss-channel conventions and loader RNG isolation.

Production requires both scheduler afterok dependency and matching successful
approval tied to recipe, source, data and hardware. Failed preparation/profile
cannot start production. Production resumes exact optimizer/scheduler and
per-rank RNG from its own checkpoint, with a deterministic consumed-data cursor.
Checkpoints occur every30minutes, epoch/end, and a20-minute walltime signal.
Never copy a profile's optimizer updates into a fresh production run. Fixed
step-budget completion is reported separately from checkpoint epochs.

Prepared baseline configurations are paired with KR configurations. Training
metrics, readiness metrics and eventual simulator success must stay distinct;
no paper result cells are filled until actual evaluation artifacts exist.


## Submitted campaign

Preparation17668675 completed successfully. All eight CPU contracts, real
DOM decoding and metadata-only strict pretrained shape/key checks pass for
all three backbones. These are not GPU readiness approvals.

| Method | Profiling baseline / KR | Production baseline / KR |
|---|---|---|
| π0.5 |17668717 /17668719|17668718 /17668722|
| SmolVLA |17668723 /17668725|17668724 /17668726|
| VLASH(π0.5) |17668727 /17668729|17668728 /17668730|

Every GPU job requests exactly two H200s. Profiles are queued by priority;
production waits on its own successful profile. Submission receipts are in
`submissions.json`. The continuation heartbeat
`finish-paired-vla-kr-profiling-and-launches` checks readiness/launch outcomes
every15minutes and stops after all six production starts are verified. The
separate hourly run register remains read-only.

Native loss-channel conventions are preserved: π0.5 trains the seven real
action channels; pinned SmolVLA and VLASH train32 padded channels. Invalid
future timesteps are masked for every model. Prediction/DOM metrics always
use the restored seven physical channels. Loader RNG and logging setup are
isolated from checkpointed model noise/flow-time RNG. TorchCodec requires
Palmetto's CUDA12 NPP libraries; the Python launcher resolves them before
importing native libraries, including in dependency-queued scripts.
