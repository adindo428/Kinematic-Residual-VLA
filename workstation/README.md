# Linux RTX PRO 6000 workstation

The local **two RTX A6000** computer now has a separate SSD profile for the paper's
**Isaac Sim 4.5.0 / Isaac Lab 2.2.1** stack. Source `workstation/env.local.sh` and use
`bash workstation/dom-paper45.sh {fetch|prepare|pilot|full|report}`. This profile
pins upstream revision `a27a06d2ca74d0e987a5e552e01013073e93cfd8`, before the 5.1
migration, and uses GPU 0 for simulation and GPU 1 for inference. Environments,
downloads, assets, caches, temporary files and results live under
`scratch/dom45` on the SSD. Runtime checks and package manifests are in its
`checks` directory. `setup-paper45.sh` records the dependency installation recipe;
`isaaclab-paper45.sh` selects the simulator Python explicitly. Set `KRVLA_RESULTS`
to a fresh SSD directory when retrying an interrupted pilot. The full suite
requires a verified nine-trial pilot and produces the released DynamicVLA row
against Table 1; it does not evaluate the paper's other policy baselines.

### Current paper-stack run

The October 10 run uses `scratch/dom45/results/dom-paper45-a6000-20261010-02`.
The pilot completed all nine trials with three official successes and no
infrastructure failures. The full run requests 90 configurations × 20 trials,
and writes the Table 1 comparison CSV, Markdown, and PNG only after all 1,800
trials pass accounting checks. Check current status before claiming completion:

```bash
bash workstation/dom-paper45.sh status
systemctl --user status krvla-dom-paper45-20261010
cat scratch/dom45/results/dom-paper45-a6000-20261010-02/suite_status.json
tail -f scratch/dom45/logs/dom-full.log
```

The service continues after this chat or terminal closes; the computer must
remain powered on and awake. A first startup hit a port held by the finished
pilot simulator; its logs are preserved under `failed-startups`. The wrapper now
executes simulator Python directly, and shutdown removes any remaining process
group children. Completed dimensions can be resumed with `full --resume-verified`;
partial dimension folders are refused and must be preserved separately before a
fresh retry. Do not start a second evaluator against the same result directory.

### Full DOM and KRVLA + Velocity training

On October 11 the user selected **all DOM episodes with the two training camera
streams (wrist and opposite)** and stopped side-camera downloads. Training
`krv100` remains queued after baseline evaluation. The
dataset revision is `bbf6baa15f7253746d271a6e958e037afa537b1e`, with 207,306 episodes
and 23,552,217 frames. The dataset destination is
`scratch/dom45/cache/huggingface/lerobot/hzxie/DOM`; the SQLite inventory and
download status are in `scratch/dom45/training/dom-full`. The resumable service
is `krvla-dom-full-download-20261010.service`. It inventories small directories
to avoid the Hub's recursive-root timeout, honors rate-limit resets, and verifies
Git/LFS hashes. Required coverage is 621,926 files / 193.74 GB, including all
metadata, parquets, and both training views; side-camera files are excluded from
downloads and completion gates. Already downloaded side-camera files remain on
SSD. Pinned resolver GET requests avoid the SDK's extra per-file HEAD request;
HTTP 429 reset times are still respected. Optional authenticated downloads use
`bash workstation/hf-ssd.sh`, which stores the token on SSD. Never paste tokens
into chat.

#### Import the existing Palmetto DOM cache

`workstation/palmetto-dom.sh` authenticates to Clemson's dedicated transfer node
`hpcdtn01.rcd.clemson.edu`. Its public host key was verified against Clemson's
[published fingerprint](https://docs.rcd.clemson.edu/palmetto/transfer/scp/).
Use a local terminal on this PC, without `sudo`:

```bash
bash workstation/palmetto-dom.sh connect
```

Complete your password/Duo prompts in that terminal. The default account is
`aparame`, and source is `/scratch/aparame/.cache/huggingface/lerobot/hzxie/DOM`.
The SSH connection socket and verified host key live on SSD in
`scratch/dom45/ssh`. The queued `krvla-dom-palmetto-transfer-20261011.service`
waits for that authenticated connection, checks the account/source metadata,
then stops the Hub downloader to prevent simultaneous writers. It uses resumable
rsync to import only the pinned inventory's unverified files; verified files
already on SSD are retained. It dereferences Hub snapshot symlinks and does not
delete files or modify Palmetto. Partial transfer data and logs stay on SSD.
The displayed checksum counters update during offline verification; use the
rsync log for transfer speed and copied bytes. The first connected source
inspection found 201,280 opposite-camera and 201,233 wrist-camera files out of
207,306 published episodes, and no side-camera tree at this path. This cache
therefore saves substantial downloading but does not establish full release
coverage; missing files still need the Hub unless another complete source is
identified.

After copying, offline verification checks the expected Git/LFS hashes. Files
from a different revision are not accepted. If the Palmetto cache lacks files
(including root metadata), the Hub downloader resumes only the missing/mismatched
required files. The side-camera stream is not required.

The user also authorized completing this same two-camera cache on Palmetto.
`krvla-dom-palmetto-backfill-20261011.service` waits for local checksum completion,
copies missing or wrong-sized required files back to the same account/path,
then verifies every required remote file against the pinned hashes. Equal-sized
mismatches are repaired and verification repeats. It does not delete files.
Remote verification manifests/reports live under the dataset's
`.krvla-verification/<revision>` directory; local progress is in
`training/dom-full/palmetto_backfill_status.json` and
`logs/dom-palmetto-backfill.log` under `scratch/dom45`. Reconnect with the command
above if the SSH master expires. After the upload completed on October 11, the
user requested local profiling proceed independently of checksum verification
running entirely on Palmetto. Remote verification continues in parallel and is
not a local profiling/training prerequisite.

Use `bash workstation/palmetto-dom.sh status` or
`bash workstation/krv100-ssd.sh` for status, and
`scratch/dom45/logs/dom-palmetto-transfer.log` for rsync progress. Different
accounts/source paths require setting `PALMETTO_DOM_USER`/`PALMETTO_DOM_SOURCE`
in the transfer service environment as well as the login command.

The PC initially authenticated as `omkarmayekar555`. The user selected their own
account instead. Log in locally using the SSD credential file; never commit or
paste the API key:

```bash
bash workstation/wandb-ssd.sh login
bash workstation/wandb-ssd.sh check
bash workstation/krv100-ssd.sh
```

`krvla-krv100-after-eval-20261010.service` waits for **all** of: the verified full
1,800-trial baseline, checksum-verified all-episode two-camera DOM on SSD,
and the user's SSD W&B login. It then verifies W&B
identity and write access with a labeled readiness
run, clones the policy environment into `scratch/dom45/envs/train`, prepares a
new full-data family with grouped train/validation/test splits and training-only
velocity/drift statistics, and profiles real two-GPU training. Production starts
only after backward/optimizer, uneven-tail, full inference, strict checkpoint,
per-rank RNG, and 15% memory-headroom checks pass. Any other readiness failure
stops the workflow for inspection. No training GPU work runs during the baseline.

On October 11, profiling encountered six published header-only MP4s that match
the official hashes but cannot decode. The user approved excluding their six
episodes (662 frames) from training, retaining 207,300 episodes / 23,551,555
frames across the grouped splits. `approved_exclusions.json` in the training
root records exact IDs/files/hashes; preparation and the coordinator require
exactly these exclusions. Raw SSD and Palmetto dataset files remain unchanged.
Prior family/profiling/source/request/stage logs are preserved under
`archives/20261011T205207Z-before-six-approved-exclusions` in the training root.
The revised family uses fresh splits and training-only statistics. Immutable
episode phase caches may be reused only after prior artifact/parquet/phase
hashes and dataset/configuration checks; old profiling approvals are not reused.
Fresh profiling then releases automatic training after readiness checks.
`bash workstation/krv100-ssd.sh` displays the exclusions and current stage.

The October 12 readiness recovery retains the completed family and all 25
profiling measurements. `readiness_krv100.py` limits the eight-episode readiness
fixture's worker metadata to those episodes, shares the leader's checkpoint
path/fingerprint, and aligns BatchNorm buffers only for canonical checkpoint
comparison. Both GPUs' actual identities remain checked independently; they
report slightly different usable VRAM and must not derive separate shared
checkpoint keys. The full repaired check passed. `train_krv100.py` uses the
shared readiness record and the measured approval to launch the frozen
numerical training function. These adapters leave the numerical source and
family unchanged. `profiling/controller_resume.json` records the repair;
resuming cached measurements requires unchanged data, worker, hardware and
measurement protocol plus valid result hashes. Completed W&B write readiness
and GPU readiness are reused when their identities/provenance match. The status
command reports the number of cached profiling trials reused. Original errors
and diagnostic attempts are preserved in the SSD training root's `diagnostics`
and stage logs. No confirmation is needed between successful profiling and
automatic training.

W&B 0.30 moved the run-ID helper used by the frozen training implementation.
The workstation training launcher supplies a compatibility alias to the
installed SDK's `wandb.sdk.lib.runid.generate_id`; this preserves the numerical
source, data family and completed profiling approval. Status shows training
startup separately from logged optimizer updates, samples, loss and rate, plus
the latest saved checkpoint counters. An active launch can still be loading its
workers; verify increasing optimizer counters before claiming progress.
The production launcher now saves through the existing atomic checkpoint and
per-rank RNG writer every five minutes at completed optimizer boundaries.
This limits unsaved work after loader failures while preserving numerical
training settings. As of October 12 at 09:58 EDT, training stopped at update
85 because of three additional published short videos (last logged update 80,
saved checkpoint zero). The full camera sample-count audit and fresh-download
evidence are in `diagnostics/video-frame-counts.json` and
`diagnostics/published-short-videos.json`. Additional exclusion approval is
granted for all three: nine excluded episodes / 984 frames total, retaining
207,297 episodes / 23,551,233 frames. The coordinator reuses prior records,
hashes, phase caches and weights while updating grouped splits/statistics.
It reconfirms the selected runtime twice and reruns complete readiness before
automatic training; the full prior search and both timelines stay archived
with their original provenance. The revised family starts from released weights
in a new W&B run. `krv100-ssd.sh` also shows current preparation progress.

Worker startup uses `training_metadata.py` in both profiling and production.
After the frozen constructor computes `base.stats`, the adapter removes its
unused per-episode statistics table from metadata before spawning workers.
Both aggregate statistics dictionaries and all episode/task/index/hash/phase
inputs are retained. Source inspection confirms workers do not read the removed
table. The profiling search/controller records adapter hashes and production
checks them against its approval. This avoids copying 207,306 unused statistics
records into each of 16 workers. It leaves the frozen numerical source intact.

### One-epoch training and dependent Table 1 evaluation

The current user budget is one completed epoch, followed automatically by a
closed-loop evaluation of that final KRVLA+Velocity checkpoint. The SSD
`training_control.json` defines the stop boundary separately from the frozen
recipe, preserving the existing LR schedule, optimizer, RNG and sampler cursor.
The trainer finishes held-out validation, saves final/epoch-001 checkpoints,
and stops before an epoch-two update. `epoch1_table1_request.json` defines the
independent Table 1 job; its service waits without allocating GPU memory.

```bash
watch -n 10 bash workstation/krv100-ssd.sh
systemctl --user status krvla-epoch1-table1-20261013.service
tail -F scratch/dom45/logs/krvla-epoch1-table1-dependency.log
```

After successful training, a nine-trial pilot gates the same 1800 published DOM
trials as the baseline under Isaac Sim 4.5.0 / Isaac Lab 2.2.1. GPU 0 simulates;
GPU 1 runs the trained model with streaming enabled. Current object pose and
velocity in robot coordinates are additional policy inputs, documented in the
protocol. The frozen policy source, checkpoint weights/normalizers, published
scene configurations, seeds and manipulation success/time scoring are checked.

Results are in
`scratch/dom45/results/krvla-velocity-epoch001-paper45-a6000-20261013-01`:
`table1_comparison.md/.csv/.png`, `summary.json`, and
`table1_baseline_comparison.md/.csv` against the completed local DOM baseline.
Per-dimension logs, videos and verified trial records remain under `full/`.
The public Table 1 benchmark is separate from held-out training loss validation.

### Mandatory throughput profiling before training

`workstation/profile_krv100.py` launches isolated trials through
`third_party/DynamicVLA/scripts/profile_dom_distributed.py`. It first sweeps
per-GPU microbatches 2, 4, 8, 16, 32, 44, 88, 176 using real decoded tensors
already on the GPU. It then compares the two fastest safe compute configurations
on the actual full training split with 0, 1, 2, 4, 8 loader workers **per GPU**.
It also compares prefetch depth, pinned transfers and fused AdamW. The three
fastest end-to-end candidates receive two longer confirmations: two warmup
updates and ten measured optimizer updates each.

Selection uses mean confirmed **samples/second**, with at least 15% measured
GPU and host memory headroom. Effective batch 352 is preserved through gradient
accumulation. Each trial starts from the released weights and discards its
updates. End-to-end trials vary deterministic uniform frame order to reduce
repeated-video file-cache bias. Both ranks report GPU utilization, memory bandwidth utilization, peak
VRAM, CPU usage, parent/worker RAM, loader wait and synchronized update latency.
Unexpected model/data/correctness errors stop the search. Only capacity/loader
failures and explicit watchdog timeouts skip a candidate. The selected runtime
must also pass the decoded optimizer, uneven-tail, inference, reload and RNG
checks. A separate instrumented run captures CPU/CUDA timelines for both ranks;
trace instrumentation is excluded from throughput selection.

Artifacts stay on the SSD under
`scratch/dom45/training/krv100-full-20261010/profiling/`:

- `profiling_status.json`: current search stage and candidate.
- `comparison.csv`: per-trial settings, speed, utilization and peak memory.
- `profiling_report.md`: confirmed candidates and the selected runtime.
- `selected_runtime.json`: frozen approval checked by the production launcher.
- `trials/<id>/`: individual measurements/logs and selected rank timeline traces,
  viewable in Perfetto.

`bash workstation/krv100-ssd.sh` shows profiling progress. The coordinator runs
profiling automatically after baseline/data/login gates pass; production cannot
start without its approval. Cached trials are tied to source, dataset and
hardware identities. GPU optimization has not been measured until this stage
runs. Short-run throughput excludes validation/checkpoint I/O and may differ
from full-epoch throughput.

The source snapshot and request are under `scratch/dom45/training/krv100-full-20261010`.
The recipe retains effective batch 352, 20-step prediction/execution, full fine
tuning, uniform frames, seed 42, BF16, fixed LR `3e-5`, a 100-epoch cap, and
validation early stopping after at least five epochs (three validations without
a 0.5% relative improvement). Microbatch is measured on the two A6000s. This
workstation support requires its own decoded checks; historical cluster profiles
do not approve it. Full-data preparation and training may take substantial time.
Training checkpoints will be in `family/runs/krv100`; W&B uses the user's selected
entity and project `Kinematic-Residual-VLA`. Simulator evaluation of the trained
model is deferred until training finishes.

The following sections describe the separate Sim 5.1 / Blackwell profile.

Start here on the new lab computer. This profile targets **RTX PRO 6000 Blackwell** on Linux x86_64, using CUDA 12.8 policy wheels. It also works with the older RTX 6000 Ada when its driver supports that CUDA build; the actual card name, architecture and memory are recorded by the checks. No GPU performance claim is made until the real checkpoint and camera pilot run there.

The code, model configuration, checkpoint and metrics do not need a faster-GPU-specific change. The policy keeps Torch 2.7.1, TorchVision 0.22.1 and TorchCodec 0.5. CUDA **12.8** replaces the L40S policy's 12.6 wheel variant because PyTorch's [Blackwell support starts with the 12.8 builds](https://pytorch.org/blog/pytorch-2-7/). See the [official 2.7.1 install commands](https://pytorch.org/get-started/previous-versions/#v271).

## 1. Clone and configure VS Code

```bash
git clone https://github.com/adindo428/Kinematic-Residual-VLA.git
cd Kinematic-Residual-VLA
cp workstation/env.example.sh workstation/env.local.sh
code krvla-linux.code-workspace
```

Edit `workstation/env.local.sh` to match your Miniconda installation and the storage disk. `KRVLA_WORK` holds assets, caches, weights and results **outside the checkout**. Do not store credentials in the example file. Select the `krvla-policy` interpreter in VS Code after installation. Tasks use the paths in `env.local.sh`, so they do not depend on the editor's selected Python.

For the next coding chat, supply [HANDOFF.md](HANDOFF.md) as context. It records the model families, metric definitions, failed diagnostics, checkpoint locations and next debugging steps. Repository files carry this context across devices; a new conversation should read them before making changes.

## 2. Driver and system packages

Run `nvidia-smi` and confirm the exact GPU and driver. For this declared Sim 5.1 setup, use a compatible NVIDIA production driver. NVIDIA lists Linux **580.65.06** as a tested version and includes RTX PRO 6000 in its [Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html). A later driver must still be checked for simulator compatibility. The installer does not change drivers.

The simulator pip route requires Python 3.11 and glibc ≥2.35 according to [NVIDIA](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html); Ubuntu 22.04/24.04 are the intended workstation systems. For an older distribution use NVIDIA's binary installation route and an appropriate simulator launcher. These apt commands are for Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg libmemcached-dev zlib1g-dev pkg-config build-essential cmake
```

Install Miniconda if needed and initialize its shell so `conda activate` and `conda run` are available in the VS Code terminal.

## 3. Install separate policy and simulator environments

```bash
source workstation/env.local.sh
conda create -n krvla-policy python=3.10 -y
conda activate krvla-policy
# Update POLICY_PYTHON in env.local.sh if this differs from its example path.
export POLICY_PYTHON="$(command -v python)"
bash workstation/setup_policy.sh

conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
bash workstation/setup_sim.sh
```

The simulator installer pins **Sim 5.1.0 / Lab v2.3.2**, with its own CUDA 12.8 Torch 2.7.0 environment, and installs no RL training frameworks. It refuses a different/edited existing Isaac Lab checkout. Simulator package downloads and first startup can take time. Review NVIDIA's runtime license prompt during first startup.

Sim 5.1 is retained to match our current diagnostic stack. NVIDIA now labels that release unsupported; future version migration must be evaluated as a separate configuration. The paper used **Sim 4.5 / Lab 2.2.1**, so the new workstation configuration is not a verified Table 1 reproduction. See [upstream clarification](https://github.com/hzxie/DynamicVLA/issues/29).

Return to the repo root, reload the final environment paths and check the actual simulator:

```bash
source workstation/env.local.sh
"$ISAACLAB" -p "$KRVLA_REPO/workstation/sim_probe.py" \
  --headless --enable_cameras --device cuda:0 --output "$KRVLA_WORK/checks/simulator.json"
"$POLICY_PYTHON" "$KRVLA_REPO/workstation/preflight.py" \
  --output "$KRVLA_WORK/checks/policy.json"
```

Policy preflight runs CUDA matrix and TorchVision kernels, decodes a tiny video through TorchCodec, checks pinned versions and checks package dependencies. The simulator probe launches the actual runtime and steps physics; its scope excludes benchmark assets/camera correctness. The DOM pilot below supplies that check. Neither is a policy-accuracy test.

## 4. Download and run the released DOM baseline

```bash
bash workstation/dom.sh fetch
```

Extract the official assets using the layout in [Local DOM evaluation](../evaluation/official_dom/README.md#2-download-source-weights-and-assets). `ASSETS` must contain `DOM-Test.zip`, `test-envs.txt`, `tests/`, `scenes/` with textures, and `objects/`. Use the official archive downloads there; model/source downloads alone do not include scenes. Allow online access on first model load for any upstream backbone configuration/tokenizer resources.

```bash
bash workstation/dom.sh prepare
bash workstation/dom.sh pilot --dry-run
bash workstation/dom.sh pilot
```

The nine-trial pilot records logs, videos, checkpoint integrity, finite action chunks, warmed inference time, policy peak VRAM and sampled device memory. Inspect actual finger contact/object lift and the EE finishing pose. A pilot verifies completed trial accounting even when manipulation fails; it is not evidence of benchmark accuracy.

Results: `$RESULTS/pilot/`. The workspace tasks expose **policy check**, **simulator check**, **DOM prepare**, **DOM pilot command preview**, and **DOM pilot**. No task automatically starts all 1,800 trials.

If the pilot behaves correctly and you decide to collect a full table:

```bash
bash workstation/dom.sh full
# Continue complete dimensions only; preserve partial failed outputs.
bash workstation/dom.sh full --resume-verified
```

Do not append to or overwrite an interrupted pilot. Select a new `RESULTS` value and prepare it again. Full results are `$RESULTS/full/00_CR/` through `08_DR/`; final tables appear in `$RESULTS/`. The underlying official checkpoint/source and streaming settings remain unchanged.

## 5. Continue the grasp diagnostics locally

These optional diagnostics copy the baseline into three separate variants: original, corrected 5.1 gripper, corrected gripper plus action-index alignment. They retain the released checkpoint and official task scoring, and record fingers, object/EE state and selected action indices. All variants use the same seeded diagnostic instrumentation. Faster gripper and queue alignment remain **unproven fixes**.

After `dom.sh prepare`, use new paths separate from the official baseline:

```bash
export DIAG_STAGE="$KRVLA_WORK/diagnostic-01"
export DIAG_RESULTS="$KRVLA_WORK/results/grasp-diagnostic-01"
"$POLICY_PYTHON" workstation/diagnostic.py prepare --base "$STAGE" \
  --official-root "$RESULTS" --stage "$DIAG_STAGE" --root "$DIAG_RESULTS"
"$POLICY_PYTHON" workstation/diagnostic.py run --stage "$DIAG_STAGE" \
  --root "$DIAG_RESULTS" --assets "$ASSETS" --isaaclab "$ISAACLAB" \
  --policy-python "$POLICY_PYTHON"
```

Three configurations × two repeats × three arms = **18 manipulation trials**. Results include `diagnostic_summary.json`, `comparison.csv`, and each variant's `telemetry/`, `client_trace.jsonl`, logs and videos. `--labels original` runs only that arm. Completed directories are preserved and cannot be overwritten. The portable driver shares the selected GPU and uses ports 28000/28001; `--port` changes them. The older Palmetto driver remains available for its two-GPU setup.

## Tracking and training

For our four 75-epoch tracking models, follow [tracking75](../experiments/tracking75/README.md) using the same `POLICY_PYTHON` and `ISAACLAB` paths. Copy their trained checkpoints and verification markers from Palmetto first. Do not load them through the DOM100 fork. Tracking pauses simulation during inference, so its physics-time score should not depend on inference speed in the same way as the streamed DOM manipulation benchmark.

For ongoing DOM100 coding/training, follow [the training guide](../docs/TRAINING.md) and [DOM100](../third_party/DynamicVLA/DOM100.md). Leave the matched training recipe unchanged during this environment migration. Start with the documented single-GPU smoke/readiness workflow before choosing a full training batch; this workstation has not yet been profiled for training.

## Validation scope

The preparation was performed on the Mac, using CPU tests, command previews and shell/Python syntax checks. CUDA wheel installation, Linux rendering and RTX PRO inference remain to be verified on the lab machine. If a check fails, preserve its JSON/log and fix the environment before interpreting evaluation scores.

The [contact-friction follow-up](../evaluation/official_dom/diagnostics/friction_20261009/README.md) adds optional `--finger-friction 1 2` conditions based on the corrected/aligned gripper, with native material readback. Use its separate comparison commands to investigate object slip without changing joint friction.
