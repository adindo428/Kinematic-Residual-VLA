# Local evaluation: released DynamicVLA on DOM manipulation

Use this workflow on the lab Linux PC to test the public DOM-trained baseline **without retraining**. It runs the real manipulation tests and compares with DynamicVLA's Table 1. The tracking occupancy metric is not used here.

**Reproduction limitation confirmed October 9, 2026:** the author states in [upstream issue #29](https://github.com/hzxie/DynamicVLA/issues/29) that Table 1 used Linux, Isaac Sim 4.5.0 / Isaac Lab 2.2.1, and separate RTX A6000 GPUs for simulation and inference. The current upstream README's 5.1.0 / 2.3.2 setup has not reproduced those numbers according to that clarification. This launcher pins the newer upstream source and shares one GPU; its scores are results for that configuration, not a verified reproduction of the paper environment.

The newer Franka configuration limits finger velocity to 0.05 m/s rather than the reference 0.2 m/s and changes joint friction. [Open PR #30](https://github.com/hzxie/DynamicVLA/pull/30) proposes a calibrated fix, validated with scripted grasps but not full learned-policy evaluation. Keep any fix in a separate diagnostic source/result directory. Verify actual finger motion, grasp timing, and observation/action indices before interpreting failed manipulation trials as a checkpoint-quality result. A pilot passing the checks below confirms loading and trial accounting; it does not establish physical grasp correctness or reproduction of Table 1.

## 1. Install two environments

The pinned upstream code recommends Python 3.10 for the policy and Python 3.11 with **Isaac Sim 5.1.0 / Isaac Lab 2.3.2** for simulation. Install the NVIDIA driver and Isaac runtime using the [Isaac Lab 2.3.2 installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/index.html). Use Linux; these commands target an NVIDIA workstation, not this Mac.

From the repository root:

```bash
export KRVLA_REPO="$PWD"
conda create -n krvla-policy python=3.10 -y
conda activate krvla-policy
sudo apt-get update
sudo apt-get install -y ffmpeg libmemcached-dev zlib1g-dev pkg-config build-essential
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 torchcodec==0.5 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "$KRVLA_REPO/evaluation/official_dom/requirements-policy.txt" \
  -c "$KRVLA_REPO/evaluation/official_dom/requirements-constraints.txt"
export POLICY_PYTHON="$(command -v python)"
export ISAACLAB="$HOME/IsaacLab/isaaclab.sh"  # change to your actual installation
```

The Torch/TorchVision/TorchCodec versions match the Palmetto policy environment. `pylibmc` is an upstream dependency; its system headers are installed above. This policy environment is also usable for tracking inference. Do not install policy requirements into the Isaac environment.

After installing Isaac Lab, activate its Python 3.11 environment and install the extra simulator packages using that environment's launcher:

```bash
"$ISAACLAB" -p -m pip install shapely==2.1.2 pyzmq==26.4.0 h5py
```

Return to `conda activate krvla-policy` before running the commands below. The Isaac launcher must still select its own Python 3.11 environment when invoked. If your Isaac Lab installation uses the active conda environment instead, follow its guide to select the simulation environment explicitly; confirm `"$ISAACLAB" -p -c 'import sys; print(sys.executable, sys.version)'` shows the simulator Python. Set `POLICY_PYTHON` to the policy environment's absolute Python path, so policy execution remains separate even when the simulation environment is active.

If you installed Isaac Lab in a conda environment, a small wrapper makes the separation explicit. Replace `env_isaaclab` if you chose another environment name:

```bash
mkdir -p "$HOME/krvla-evaluation/bin"
cat > "$HOME/krvla-evaluation/bin/isaaclab-local" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
exec conda run --no-capture-output -n env_isaaclab bash "$HOME/IsaacLab/isaaclab.sh" "$@"
SH
chmod +x "$HOME/krvla-evaluation/bin/isaaclab-local"
export ISAACLAB="$HOME/krvla-evaluation/bin/isaaclab-local"
"$ISAACLAB" -p -c 'import sys; print(sys.executable, sys.version)'
```

Use this wrapper for the simulation package installations too. Keep `POLICY_PYTHON` pointing to `krvla-policy/bin/python`.

## 2. Download source, weights, and assets

```bash
export WORK="$HOME/krvla-evaluation"
export STAGE="$WORK/official"
export ASSETS="$WORK/assets"
export RESULTS="$WORK/results/dom-local-01"
export RUNNER="$KRVLA_REPO/evaluation/official_dom/scripts/local.py"
mkdir -p "$WORK" "$ASSETS"
"$POLICY_PYTHON" "$RUNNER" fetch --stage "$STAGE"
```

`fetch` obtains upstream commit `434dc7105cf6d559018d15af5fecbcdad44f036e` and checkpoint `hzxie/dynamic-vla-DOM` revision `c9609d94c5940eb42436a31927d7e80ae454279e`. It verifies the released weight SHA256 `e3da8b1e0326d7a3ee6873f7fa17cdf7015201251901be5107377cb851ce3af8`. Existing upstream source at a different commit is refused. Downloading the checkpoint does not train it.

Download the [DOM Test archive](https://gateway.infinitescript.com/?f=DOM-Test) and [DOM 3D Objects archive](https://gateway.infinitescript.com/?f=DOM-3D-Objects). Extract their **contents** into this layout; avoid extra nested directory levels:

```text
$ASSETS/
  DOM-Test.zip           # keep the original test archive for byte verification
  test-envs.txt          # 90 test configuration entries
  tests/*.json
  scenes/*.usd
  scenes/textures/...
  objects/apple/apple01.usd
  objects/<other object directories>/...
```

The test archive includes benchmark scenes. If required scene dependencies are missing, obtain [DOM 3D Scenes](https://gateway.infinitescript.com/?f=DOM-3D-Scenes) and retain the complete textures directory. Do not edit the JSON tests or modify USD texture references for this reproduction. The simulator also needs the Franka robot assets used by Isaac Lab; ensure its asset server is reachable, or configure the same local mirror in your Isaac installation. Missing assets are infrastructure failures.

## 3. Prepare and inspect commands

```bash
"$POLICY_PYTHON" "$RUNNER" prepare \
  --stage "$STAGE" --assets "$ASSETS" --root "$RESULTS" --gpu-label "RTX 3080"
"$POLICY_PYTHON" "$RUNNER" run \
  --stage "$STAGE" --assets "$ASSETS" --root "$RESULTS" \
  --isaaclab "$ISAACLAB" --policy-python "$POLICY_PYTHON" --mode pilot --dry-run
```

Preparation checks the 90 released tests, dimensions, checkpoint configuration, required scene/object files, and byte equality with `DOM-Test.zip`. It writes hashes, dependencies, test lists and `protocol.json`. It does not prove camera/physics/GPU operation. The dry run prints executable argument lists without launching the GPU processes.

## 4. Run the nine-trial pilot

```bash
"$POLICY_PYTHON" "$RUNNER" run \
  --stage "$STAGE" --assets "$ASSETS" --root "$RESULTS" \
  --isaaclab "$ISAACLAB" --policy-python "$POLICY_PYTHON" --mode pilot
```

This checks exact checkpoint loading, finite 20-step action chunks, and one complete trial in each dimension. A manipulation failure can be a valid trial; a crashed client, missing trial, or no-action outcome cannot pass verification. `pilot_verified.json` is required before the full suite.

On the **RTX 3080**, inspect `policy_probe.json`, `pilot/gpu_memory.csv`, `pilot/server.log`, and `pilot/client.log`. The probe records policy peak allocated/reserved VRAM. The CSV samples whole-device memory every five seconds while simulation and policy run together; it can miss short peaks. Check for out-of-memory errors and view pilot videos before proceeding. The 3080 has less VRAM than the L40S; this local launcher has CPU validation but has not yet been run on your lab GPU. See [Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).

## 5. Run the complete benchmark

```bash
"$POLICY_PYTHON" "$RUNNER" run \
  --stage "$STAGE" --assets "$ASSETS" --root "$RESULTS" \
  --isaaclab "$ISAACLAB" --policy-python "$POLICY_PYTHON" --mode full
```

Dimensions run serially on one GPU: CR, DA, LS, VU, SR, MP, VG, MG, DR. Each has ten configurations and twenty trials per configuration. To execute just CR after the pilot, add `--dimension CR`. To continue after whole dimensions finished, add `--resume-verified`; finished dimensions are reverified and skipped. Partially written directories are refused rather than silently appended. Preserve failed directories, use a fresh result root for a clean restart, and rerun the pilot there. Default ports are 26000/26001; use `--port` if occupied. Ctrl-C terminates both child process groups.

Final outputs:

```text
$RESULTS/
  protocol.json, preflight.json, policy_probe.json, pilot_verified.json
  pilot/                 # logs, runtime, commands, trial results, videos, memory samples
  full/00_CR/ ... full/08_DR/
  summary.json
  table1_comparison.csv
  table1_comparison.md
  table1_comparison.png
```

After independently completing all dimensions, generate the table with:

```bash
"$POLICY_PYTHON" "$RUNNER" report --root "$RESULTS"
```

Success is instructed manipulation completion. Time is simulation step count × actual step duration, averaged across successful and failed trials. The observer only adds durable result recording; actions, reset behavior, task termination and continuous inference/streaming remain upstream. The paper's overall reference is **47.06% success, 8.53 s**. The upstream reset seed behavior is retained; repeat independence is not claimed. GPU speed can change streaming outcomes, so a different GPU need not reproduce the paper exactly.

## Palmetto reference

`scripts/run_suite.py`, `scripts/submit.py` and `slurm/` preserve the L40S deployment used for the remote evaluation. They contain Palmetto paths. The workstation entry point is **`scripts/local.py`**. See [cluster notes](../../docs/CLUSTERS.md).
