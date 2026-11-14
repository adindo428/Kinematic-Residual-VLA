# Portable family fine-tuning

See [the post-training handoff](../post_training/README.md) for the portable
SmolVLA/VLASH/pi0.5 baseline/KR workflow, unchanged physical metrics and
current object velocity conditioning through existing language inputs.

---

# Architecture-preserving family fine-tuning

The prepared campaign is `scratch/real_robot/family-finetuning-v6/campaign.json`.
It pins `adi2440/kinematic-residual-vla-checkpoints` at revision
`b722d7bfdd31b49fc19d35b8235ae1bf1d298228` and selects the published latest aliases
for SmolVLA, VLASH and pi0.5, each with and without KR. Each weight file is verified
against the repository SHA256 list before loading. Native loading is strict and
checks checkpoint identity/progress against the release index. This is not a
stock `from_pretrained` conversion.

VLASH baseline is a partial epoch-6 snapshot, while its KR pair completed epoch 10.
Pi0.5 baseline completed epoch 10, while its KR pair is a partial epoch-6 snapshot.
SmolVLA's pair both completed epoch 10. Original update counts remain recorded;
these histories do not constitute a matched-pretraining KR ablation.

## Architecture contract

Preserve the native backbone, existing action expert/projections, camera views,
single-frame observation inputs, prompt/state representation, 20-action horizon
and 25 Hz action time. No learned velocity features, additional history/state
channels, new heads, LoRA modules, attention blocks or parameterized KR modules
are introduced. KR is the released analytical coordinate transform, applied before
normalization and restored once. The real-data `velocity_required` flag disables
analytical drift after annotated holding; this is not a learned input or phase head.

Parameter names and shapes are fingerprinted immediately after strict native
loading and checked after training setup, optimizer smoke steps and before saves.
The action sampler's inference-only no-grad wrappers are removed for auxiliary
backpropagation; SmolVLA's Euler update is made out of place to preserve autograd.
The original sampler remains available for fixed-noise numerical parity checks.
None of these changes adds or changes a neural layer.

Training and validation use the checkpoint wrapper's original token/affine
contract consistently. In particular, VLASH's high-level convenience inference
API re-tokenizes state differently from its checkpoint training wrapper. The
campaign uses the wrapper's prepared tokens directly with the unchanged native
sampler, rather than mixing those two preprocessing contracts.

SmolVLA retains its released expert/state-projection fine-tuning mask. For the
large models on this 24 GiB workstation, existing vision/language parameters are
frozen and existing action-expert/projection parameters are trained in FP32 with
BF16 frozen backbones. Unused inherited vocabulary embeddings/heads are frozen.
This changes the trainable set relative to earlier full-model runs, not the
architecture. Each baseline/KR pair gets the same mask and training settings.

## Training and selection

Use the same frozen real dataset, splits, two-frame quality eligibility mask,
task/phase sampling weights and photometric augmentation as v5. Each native model
consumes only its current frame, preserving its original architecture. VLASH
retains uniform 0–3 step training delays and command-based state rollforward;
validation uses delay zero. No future image or object-velocity inputs are used.
The direct v2.1 reader checks video/label lengths and avoids converting or changing
the frozen dataset. Its validation labels/masks were compared with all 1,609
anchors in the original ten-episode evaluation.

New unaugmented, train-only delta-chunk mean/std and quantiles are saved in
`real_statistics.json`; baseline/KR pairs share statistics. SmolVLA keeps mean/std
normalization; VLASH and pi0.5 keep quantile normalization. The neural input/output
dimensions stay unchanged.

The physical objective, horizon weights and validation probe match v5. Increase
reconstruction coverage from two to four examples every fourth effective update.
SmolVLA evaluates those four examples together; large models retain sequential
sampler backward passes to bound memory. Effective flow batch size remains 16.
After the throughput benchmark, SmolVLA uses microbatch 16 with no accumulation;
large models retain microbatch 1 and accumulation 16. Preserve each implementation's native flow-loss padding convention
(7 channels for pi0.5, 32 for SmolVLA/VLASH); physical losses use XYZ, SO(3) and
gripper channels only.

EMA adds no model parameters. Average existing trainable tensors, with target
decay 0.999 and a warmup; compare raw/EMA weights on identical fixed validation
anchors, instruction RNG and three noise seeds. Select by the same scaled physical
score. Keep raw optimizer state separate from EMA-selected inference weights.
Retain best/latest full checkpoints and one recovery file rather than every large
epoch file. Never report the partial recovery file as a completed epoch.

Each model finishes with a before/after evaluation on the original ten validation
episodes, one shared noise seed, and unchanged expert targets. Frozen backbone
weights are shared; only the existing trained tensors are restored for the initial
versus selected comparison. These are target-prediction metrics, not manipulation
success or controller tracking performance.

## Queue and runtime

The current supervisor is `workstation/run_benchmarked_family_campaign.py`.
It adopted the existing, already-benchmarked SmolVLA baseline process without
interrupting training. After that run finishes, it benchmarks all five remaining
checkpoints before starting any further production training job. The former
`run_family_campaign.py` supervisor is retained for provenance.

`workstation/profile_family_performance.py` loads each checkpoint strictly and
benchmarks real forward/backward/optimizer/EMA updates, with the unchanged flow
batch of 16 and four reconstruction examples every fourth update. It tries flow
microbatches 1, 2, 4, 8 and 16, then reconstruction microbatches 1, 2 and 4. It
records finite-gradient checks, allocated/reserved memory, update times and
out-of-memory failures. Selection excludes configurations without GPU-memory
headroom. Loader workers 0, 2 and 4 are measured separately, excluding startup
from their steady-state rate. Benchmark optimizer updates are disposable and
never saved as training weights.

Reports are saved as `benchmarks/FAMILY.json`, including exact checkpoint SHA,
architecture fingerprint and selected settings. The original SmolVLA benchmark
and live verification are imported as its existing report; they were measured
during its active run, not retroactively claimed to precede its original launch.
The remaining five reports must all pass before settings are committed to
`campaign.json` and `benchmark-summary.json` is saved. A failed benchmark blocks
the next training stage. Each future training process also checks that its saved
benchmark and chosen settings match before loading the model for training.

After the benchmark stage, every remaining family still passes the isolated GPU
check for native-sampler parity, gradients, optimizer execution, EMA and architecture
fingerprint. Only then does its production training start. Settings are selected
per checkpoint; larger models are not blindly assigned SmolVLA's microbatch size.

Progress is in `scratch/real_robot/family-finetuning-v6/queue.json` and
`downloads.json`. Each model has separate check/train logs, W&B run and output
directory under `runs/`. The current v5 runner/source/environment are unchanged.

`workstation/family-python.sh` selects isolated Transformers 4.53.3 and native
adapter source, leaving the running Transformers 5.2 environment intact. Official
OpenPI transformer support is pinned separately, with source and installed patch
hashes recorded. Its version-check marker is adjusted from 4.53.2 to the checkpoint
runtime's 4.53.3; model support code is unchanged. Record actual runtime versions
and strict-load/gradient checks rather than claiming the workstation duplicates
the original H200 environment.

## SmolVLA throughput update

`throughput-benchmark.json` records disposable measurements with the same effective
batch, objective and four reconstruction examples every fourth update. Compute
throughput improved from about 25 to 73 samples/s using microbatch 16 and a
four-example reconstruction microbatch. Peak allocated memory was 4.52 GiB.
Four loader workers supplied about 121 samples/s versus 32 without workers.
The live resumed run sustained about 68–70 samples/s with GPU utilization mostly
93–97% in a 12-second sample (brief dips), using about 8 GiB reported total VRAM.
W&B timestamps over the previous updates 38,000–38,997 measured 13.10 samples/s
end to end; the resumed steady rate of about 69 samples/s is approximately 5.27x
faster. `throughput-before-after.json` records this comparison separately from the
compute-only benchmark. The first optimized recovery at update 39,500 was checked:
optimizer and EMA counters both matched 39,500 and the architecture fingerprint
and effective batch remained unchanged.

The SmolVLA performance profile now enables four spawned prefetch workers, an
eight-episode decoder cache per worker and vectorized photometric operations.
The latter matches the previous torchvision brightness/contrast/saturation
operations, preserving per-sample factors shared across cameras/history and the
original ranges. It does not change labels, transforms or augmentation strength.
Validation remains the previous single-anchor fixed-seed protocol.

The baseline resumed at update 39,000, epoch 10, restoring raw model, optimizer,
scheduler, EMA and RNG. Sampler replay verified 32,736 consumed anchors and 32,953
remaining anchors in that epoch. The best checkpoint and initial comparison
weights were retained. `resume_events.json` and `run_config.pre-throughput.json`
preserve this transition; the raw recovery checkpoint is linked as
`throughput-step-39000.pt`. Microbatch regrouping/worker RNG can change stochastic
realizations and floating-point reductions; this is not a bitwise-identical replay.
Future resume checkpoints explicitly record whether an epoch is complete.
SmolVLA+KR gets the same performance profile. Architecture, effective batch,
learning-rate schedule, loss weights and validation selection remain unchanged.

New expert recovery demonstrations still require operator collection; none are
fabricated or added to this campaign. The existing recovery provenance workflow
is documented in `README_ACTION_RETRAINING.md`.

## Pi0.5 loading and sampling compatibility

The 40 GB pi0.5 exports contain both model and unused pretraining optimizer state.
Private writable mmap failed on this 32 GiB host before tensors could load. The
loader now uses documented shared file-backed mapping only for pi0.5, and copies
checkpoint tensors into existing model tensors before training. A regression
test verifies that modifying the loaded model leaves the source checkpoint
unchanged. The mapping option is restored outside the load context. No host
overcommit settings or memory limits are changed.

Pi0.5 activation checkpointing can suppress the prefix KV cache in train mode.
Full-sampler forwards temporarily disable checkpointing flags and restore them
afterward; ordinary flow training still uses activation checkpointing. The
benchmark requires a fixed-noise train-mode sample to match the eval-mode native
reference before timing begins. Reports from before this compatibility fix were
archived and invalidated; no pi0.5 production training used them. Neither fix
changes architecture, parameters, normalization, or the physical loss.
