# A100 80 GB alternatives for paired KRFM on audited DOM

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
seed42, two A100 80 GB GPUs, BF16 and accumulation preserving their effective batch.
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

Remote root: `/scratch/aparame/Dynamic_VLA/cross-vla-krfm-a100-r2-20261019`.
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
updates on two actual A100 80 GB GPUs, checks finite gradients and weight change,
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


## A100 alternatives

Authorized October 19. Each profile and production job uses exactly two A100
80 GB GPUs and 224 GiB node RAM. Global batch sizes, optimizer recipes,
update budgets, audited data and all baseline/KR definitions match the H200
campaign. Microbatch1 with gradient accumulation; own actual A100 readiness
and identity-matching afterok gate required.

Remote root: `/scratch/aparame/Dynamic_VLA/cross-vla-krfm-a100-r2-20261019`.
The original H200 jobs and their files remain unchanged at the original root.
A100 checkpoints/logs/W&B IDs are separate. Frozen preparation receipts and
read-only pretrained assets are reused on Palmetto; no datasets or weights
are copied to the Mac. This is an additional requested campaign, not an
automatic replacement/cancellation of any H200 run.

## Launcher repair, attempt 2
The first six readiness jobs failed before model loading because the venv
inherits PyTorch but does not have a torchrun console script. Their six
dependent production jobs were cancelled by the scheduler. All failure
receipts and logs are preserved in the previous root. This frozen retry uses
venv Python with -m torch.distributed.run, preserving the same native PyTorch
launcher, model/training source, recipes, data and readiness gates. No package
installation or environment modification. H200 jobs/source remain unchanged.
The missing native console entry point for the original H200 campaign was
subsequently restored and passed two-worker startup. No package installation
or change to H200 job IDs, source, scripts or recipes; see the original root
launcher-repair.json receipt.

## Submitted replacement jobs

| Model | Readiness baseline / KR | Training baseline / KR |
|---|---|---|
| pi05 | 17692792 / 17692794 | 17692793 / 17692795 |
| smolvla | 17692796 / 17692798 | 17692797 / 17692799 |
| vlash | 17692800 / 17692802 | 17692801 / 17692803 |

Verified October19 6:35PM EDT: all six profiles PENDING(Priority),
production PENDING(Dependency); no GPU readiness approvals yet. Superseded
first-attempt profile failures and cancelled production IDs remain in
previous-failure.json and the local registry.
