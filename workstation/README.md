# Linux RTX PRO 6000 workstation

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
