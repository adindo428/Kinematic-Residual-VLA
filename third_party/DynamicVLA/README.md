<img src="https://www.infinitescript.com/projects/DynamicVLA/DynamicVLA-Logo.webp" height="100px" align="right">

# DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation

[Haozhe Xie](https://haozhexie.com), Beichen Wen, Jiarui Zheng, [Zhaoxi Chen](https://frozenburning.github.io/), [Fangzhou Hong](https://hongfz16.github.io/), [Haiwen Diao](https://paranioar.github.io/), [Ziwei Liu](https://liuziwei7.github.io/)

S-Lab, Nanyang Technological University

[![codefactor badge](https://www.codefactor.io/repository/github/hzxie/DynamicVLA/badge)](https://www.codefactor.io/repository/github/hzxie/DynamicVLA)
![Counter](https://api.infinitescript.com/badgen/count?name=hzxie/DynamicVLA)
[![arXiv](https://img.shields.io/badge/arXiv-2601.22153-b31b1b.svg)](https://arxiv.org/abs/2601.22153)
[![YouTube](https://img.shields.io/badge/Spotlight%20Video-%23FF0000.svg?logo=YouTube&logoColor=white)](https://youtu.be/NmJnHcI04_Q)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-DOM%20Dataset-orange)](https://huggingface.co/datasets/hzxie/DOM)

![Teaser](https://github.com/user-attachments/assets/ffc2071a-c4b8-4ebf-9a41-870de65bb3da)

## Changelog🔥

- [2026/04/26] Released training and testing code.
- [2026/01/26] Repository created.

## Cite this work📝

```
@article{xie2026dynamicvla,
  title     = {DynamicVLA: A Vision-Language-Action Model for 
               Dynamic Object Manipulation},
  author    = {Xie, Haozhe and 
               Wen, Beichen and 
               Zheng, Jiarui and 
               Chen, Zhaoxi and 
               Hong, Fangzhou and 
               Diao, Haiwen and 
               Liu, Ziwei},
  journal   = {arXiv preprint arXiv:2601.22153},
  year      = {2026}
}
```

## Dataset and Pretrained Models 🛢️

### DOM Dataset

- [DOM Training Set](https://huggingface.co/datasets/hzxie/DOM) – for training DynamicVLA
- [DOM Testing Set](https://gateway.infinitescript.com/?f=DOM-Test) – for benchmarking; includes test configurations and a subset of 3D scenes
- [DOM 3D Objects](https://gateway.infinitescript.com/?f=DOM-3D-Objects) – assets for data generation and benchmarking
- [DOM 3D Scenes](https://gateway.infinitescript.com/?f=DOM-3D-Scenes) – full scene assets for data generation

### Pretrained Models

- [DynamicVLA  (trained on DOM)](https://huggingface.co/hzxie/dynamic-vla-DOM)


## Installation 📥

We recommend using **conda** to create two separate environments:

- one for **model training & inference**
- one for **Isaac Lab simulation & evaluation**

### PyTorch Environment

- Install **Python 3.10** and **PyTorch 2.7.1** *(Other versions should work, but are not fully tested)*
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Isaac Lab Environment

- Install **Python 3.10** *(Other versions should work, but are not fully tested)*
- Install **Isaac Sim 4.5.0** and **Isaac Lab 2.2.1**
  Follow the official guide: https://isaac-sim.github.io/IsaacLab/v2.2.0/source/setup/installation/index.html
- Install additional dependencies:

```bash
pip install shapely pyzmq h5py
```

## Benchmarking Your Policy 🏅

### Prepare scenes and objects

Download [DOM Testing Set](#dom-dataset) (including a subset of DOM 3D scenes) and [DOM 3D Objects](#dom-dataset).

```
PROJECT_ROOT/
├── objects/           # Put DOM 3D Objects here
├── scenes/            # Put DOM 3D Scenes here
|   └── textures       # Put the textures of 3D scenes here
|   └── *.usd          # Put the USD files of 3D scenes here
├── tests/             # Put DOM Testing Set here
|   └── *.json
|── test-envs.txt      # The list of test environments (included in DOM Testing Set)
├── datasets/          # Generated simulated datasets will be stored here
└── dynamic-vla/       # git clone https://github.com/hzxie/DynamicVLA dynamic-vla
    └── runs           # Create folder for evaluation and checkpoints output
```

### Run Policy Evaluation Server

> ⚠️ This step requires the **Isaac Lab environment**

From the `PROJECT_ROOT/dynamic-vla` directory, run:

```bash
python3 simulations/evaluate.py \
    --scene_dir ../scenes \
    --output_dir ../output/evaluation \
    --env_cfg ../test-envs.txt \
    --enable_cameras --headless -n 20 --save
```

**Arguments:**

- `test-envs.txt` are provided by [DOM Testing Set](#dom-dataset)
- `-n 20`: run 20 trials per environment
- `--save`: save evaluation videos to `output_dir`
- `--headless`: run without GUI
- `--enable_cameras`: enable visual observations

### Run Policy Inference

> ⚠️ This step requires the **PyTorch environment**

From the `PROJECT_ROOT/dynamic-vla` directory, run:

```bash
python3 scripts/inference.py \
    -p /path/to/vla-checkpoint \
    -r euler -d -s
```

**Arguments:**

- `-p`: path to the trained model checkpoint
- `-r euler`: use Euler angles for rotation representation
- `-d`: enable **delta actions** *(actions are relative to current state)*
- `-s`: enable **contiguous inference** *(if supported by the model)*

## Simulated Dataset Generation 🧪

> ⚠️ This step requires the **Isaac Lab environment**

### IsaacSim Simulation

From the `PROJECT_ROOT/dynamic-vla` directory, you can generate synthetic data using:

```bash
python3 simulations/simulate.py \  
        --headless  --enable_cameras  --seed 42  --save  --task place
```

Example configuration file: `simulations/configs/sim_cfg.yaml`

**Arguments:**

-   `--task` : task type to simulate. Options: `pick`, `place`, `long-horizon`.
-   `--robot`: robot type used in simulation _(default: `franka`, also supports `piper`)_.
-   `--headless`: run simulation without GUI.
-   `--enable_cameras`: include visual observations in the output dataset.
-   `--debug`: enable debug mode and render trajectories as `.mp4` videos.
-   `--seed`: random seed for simulation _(automatically increments for each run if specified)_.
-   `-n`, `--n_simulations`: number of simulation episodes to generate *(default: `10,000`)*.
-   `--save`: save generated simulation data in HDF5 format.

### Trajectory Replay

After data generation, convert the trajectories into a format compatible with VLA training:

```bash
python3 scripts/translate_dataset_seq.py \
        --dataset_dir ../datasets --output_dir ../datasets-tr \
        --enable_cameras --headless --save
```

**Arguments:**

-   `--dataset_dir`: directory containing the raw simulation datasets.
-   `--output_dir`: directory to store the processed trajectories.
-   `--enable_cameras`: include visual observations in the output dataset.
-   `--headless`: run simulation without GUI.
-   `--save`: save generated simulation data in HDF5 format.

### Convert LeRobot Dataset

We provide a script to convert the generated `.h5` files into the LeRobot dataset *(v2.1 format)*, using **Euler angles** as the rotation representation:

```bash
python3 scripts/create_lerobot_dataset.py \  
        --dataset_dir ../datasets-tr --repo hzxie/DOM --rotation euler
```

This will create lerobot dataset using all the hdf5 datasets in the default output directory.

## Training 👩🏽‍💻

> ⚠️ This step requires the **PyTorch environment**

From the `PROJECT_ROOT/dynamic-vla` directory, run:

```bash
torchrun --nnodes=1  --nproc_per_node=8  --standalone run.py \  
  -c configs/dynamicvla.yaml \  
  -p /path/to/pretrained/model
  -d hzxie/DOM
```

**Arguments:**

-   `--nnodes`: number of compute nodes (machines) used for distributed training
-   `--nproc_per_node`: number of GPUs per node
-   `-c`: path to the training config file
-   `-p`: path to the pretrained model checkpoint *(optional)*
-   `-d`: name of the LeRobot dataset *(v2.1 format)*

### Residual Speed-OOD Training

This fork includes a proactive residual experiment that trains on low-to-mid
speed DOM episodes and evaluates on faster ones. The split is resolved from the
configured candidate range by mean object speed:

```text
train: mean speed <= 0.12 m/s
test:  mean speed >= 0.13 m/s
```

```bash
python run.py \
  --cfg configs/residual_dynamicvla_dom_cartesian_speed_ood_0p12_0p13.yaml \
  --exp residual_dynamicvla_dom_cartesian_speed_ood_0p12_0p13 \
  --ckpt hzxie/dynamic-vla-DOM
```

Use the same command with `--prefetch-only` to resolve and warm the speed split
before starting a GPU allocation.

Then run the offline high-speed comparison:

```bash
python scripts/proactive_residual_offline_compare.py \
  --cfg configs/residual_dynamicvla_dom_cartesian_speed_ood_0p12_0p13.yaml \
  --baseline-checkpoint hzxie/dynamic-vla-DOM \
  --residual-checkpoint runs/checkpoints/residual_dynamicvla_dom_cartesian_speed_ood_0p12_0p13 \
  --output runs/proactive_vla/offline_compare/speed_ood_0p12_0p13_report.json \
  --max-batches 0
```

See `PROACTIVE_RESIDUAL_MVP.md` for the full residual workflow.

### Full-Finetune Speed-OOD Slurm Training

For the current MVP full-parameter baseline-then-residual run, submit from the
DynamicVLA project root so the relative Slurm log path lands in `runs/slurm_logs`:

```bash
cd /scratch/aparame/Dynamic_VLA/third_party/DynamicVLA
mkdir -p runs/slurm_logs
sbatch slurm/train_full_finetune_speed_ood_pair_h100.sbatch
# or
sbatch slurm/train_full_finetune_speed_ood_pair_h200.sbatch
sbatch slurm/train_full_finetune_speed_ood_pair_a100.sbatch
```

These launchers call `scripts/train_full_finetune_speed_ood_pair.sh`, which runs
`dynamicvla_dom_full_finetune_speed_ood_0p12_0p13` first and then
`residual_dynamicvla_dom_full_finetune_speed_ood_0p12_0p13`. By default
`AUTO_RESUME=1`: before each stage, the shell script checks
`runs/checkpoints/<exp>/config.json` and `model.safetensors`; if both exist, it
passes that checkpoint directory to `run.py --ckpt` and resumes from the saved
epoch. Set `AUTO_RESUME=0` only when you intentionally want to ignore an existing
checkpoint.

The H100 launcher is tuned for the observed low memory use by defaulting to
`DVLA_BATCH_SIZE=32` and `DVLA_GRAD_ACCUM_STEPS=8`, preserving the old effective
batch size while halving the number of batches per epoch. The A100 launcher keeps
`16/16`; the 2-GPU H200 launcher uses `64/16`. Override these at submit time if
needed:

```bash
DVLA_BATCH_SIZE=24 DVLA_GRAD_ACCUM_STEPS=11 \
  sbatch slurm/train_full_finetune_speed_ood_pair_h100.sbatch
```

The full-finetune Slurm wrappers also set `RUN_PREFETCH=0` by default because the
CPU preprocessing job should already have warmed the DOM cache. Use
`RUN_PREFETCH=1` on submit only when you intentionally want the GPU job to verify
and fill missing cache entries.

### Full-Finetune CPU Eval-Split Prefetch and IID/OOD Evaluation

After full-finetune training, warm the held-out high-speed OOD evaluation split
without requesting a GPU:

```bash
cd /scratch/aparame/Dynamic_VLA/third_party/DynamicVLA
sbatch slurm/postprocess_full_finetune_eval_splits.sbatch
```

The wrapper defaults to `EVAL_SPLITS=ood`, where `ood` maps to the configured
`test` speed split. To warm both the IID low-speed split and OOD high-speed split:

```bash
EVAL_SPLITS="iid ood" sbatch slurm/postprocess_full_finetune_eval_splits.sbatch
```

Once the cache is warm, run the combined baseline-vs-residual offline evaluation
on IID and OOD splits:

```bash
sbatch slurm/evaluate_full_finetune_iid_ood_h100.sbatch
```

This writes a combined report and plots to:

```text
runs/proactive_vla/offline_compare/full_finetune_iid_ood_report.json
runs/proactive_vla/offline_compare/iid_ood_metric_summary.png
```

Use `MAX_BATCHES`, `EVAL_BATCH_SIZE`, `EVAL_SPLITS`, `BASELINE_CKPT`,
`RESIDUAL_CKPT`, or `EVAL_OUTPUT` at submit time to override the defaults.

### Scratch Speed-OOD DDP Training

For a leakage-safe baseline-vs-residual comparison, train both VLAs from scratch
on the materialized low-speed DOM split and keep the high-speed split untouched
until post-training evaluation. The fixed split is stored in
`configs/dom_speed_ood_0p12_0p13_split.json`:

```text
train: mean object speed <= 0.12 m/s
test:  mean object speed >= 0.13 m/s
```

The scratch configs set `TRAIN.EVAL_TEST_DURING_TRAIN: false`, so `run.py`
resolves, prefetches, and loads only the train episodes during optimization. Do
not pass a DOM pretrained `--ckpt`; these runs intentionally do not initialize
from `hzxie/dynamic-vla-DOM`. Auto-resume only reuses local scratch
checkpoints from the same experiment.

Submit one sequential 8-GPU job on H100s or A100s from the DynamicVLA root:

```bash
cd /scratch/aparame/Dynamic_VLA/third_party/DynamicVLA
mkdir -p runs/slurm_logs
sbatch slurm/train_scratch_speed_ood_pair_h100.sbatch
# or
sbatch slurm/train_scratch_speed_ood_pair_a100.sbatch
```

Scratch launchers use the same `AUTO_RESUME=1` behavior as full-finetune runs,
but only after a local scratch checkpoint exists. The H100 scratch wrapper uses
`DVLA_BATCH_SIZE=256` and `DVLA_GRAD_ACCUM_STEPS=2`; the A100 wrapper keeps
`128/4`. Both preserve the scratch configs' effective batch size while reducing
batches per epoch on H100.

The job runs:

1. `dynamicvla_dom_scratch_speed_ood_0p12_0p13` with residual flow disabled.
2. `residual_dynamicvla_dom_scratch_speed_ood_0p12_0p13` with kinematic residual flow enabled.
3. A post-training offline comparison on the held-out high-speed split.

Set `RUN_PREFETCH=0`, `RUN_TRAIN=0`, or `RUN_OFFLINE_COMPARE=0` at submit time to
skip any of those phases. If you submit from inside `slurm/`, Slurm's relative
output path writes logs under `slurm/runs/slurm_logs`; submitting from the project
root writes them under `runs/slurm_logs`.

The final comparison report is written to:

```text
runs/proactive_vla/offline_compare/speed_ood_0p12_0p13_scratch_full_report.json
```

### Automatic Paired-Velocity GPU Tuning

Use the submission wrapper to select an A100, H100, or H200 allocation with four
through eight GPUs:

```bash
cd /scratch/aparame/Dynamic_VLA/third_party/DynamicVLA
bash scripts/submit_paired_velocity_pair.sh h200 4
bash scripts/submit_paired_velocity_pair.sh h100 8
bash scripts/submit_paired_velocity_pair.sh a100 6
```

Additional arguments are forwarded to `sbatch`, for example
`bash scripts/submit_paired_velocity_pair.sh h200 4 --partition work1`.
The job verifies the hardware actually allocated by Slurm, launches one DDP
process per GPU, and chooses the following high-throughput per-GPU batch:

```text
A100 below 70 GiB: 16
A100 at least 70 GiB: 32
H100:                 32
H200:                 64
```

Gradient accumulation is calculated automatically to stay as close as possible
to `DVLA_TARGET_GLOBAL_BATCH_SIZE=1024`. `NPROC_PER_NODE`,
`DVLA_BATCH_SIZE`, or `DVLA_GRAD_ACCUM_STEPS` can still be set explicitly to
override an automatic choice. The legacy H100-named sbatch file remains the
underlying generic launcher; direct `sbatch --gres=gpu:<type>:<count>` submission
also uses runtime auto-tuning.

### Residual-Only Paired-Velocity Resume

To skip baseline training and resume only the paired-velocity residual VLA from
its local rolling checkpoint, submit:

```bash
cd /scratch/aparame/Dynamic_VLA/third_party/DynamicVLA
RESUME_RESIDUAL_ONLY=1 RUN_PREFETCH=0 \
  sbatch slurm/train_paired_velocity_pair_h100.sbatch
```

`RESUME_RESIDUAL_ONLY=1` requires `config.json`, `model.safetensors`, and
`training_state.pt` in `runs/checkpoints/residual_dynamicvla_dom_paired_velocity`
(or in the directory supplied with `RESIDUAL_CKPT`). It skips baseline training
and passes the residual checkpoint directory to `run.py --ckpt`, restoring the
saved epoch, optimizer, and learning-rate scheduler. This mode does not require
`AUTO_RESUME=1`. Set `RUN_OBSERVED_EVAL=0 RUN_PAIRED_SWEEP=0` as well when the
allocation should perform training only.

### Checkpoint Evaluation

> ⚠️ This step requires the **PyTorch environment**

During training, you can automatically evaluate checkpoints using the following script:

```bash
python3 scripts/eval_checkpoints.py \
        -r euler -d -s -p "*fvit*46k*" \
        --ckpt_dir ./runs/checkpoints/ \  
```

**Arguments:**

-   `--ckpt_dir`: directory containing the checkpoints to be evaluated.
-   `-p`: pattern used to match checkpoint filenames *(supports wildcard patterns)*.
-   `--host`: host address of the evaluation server *(default: `127.0.0.1`)*.
-   `--img_port`: port used for the image stream on the evaluation server *(default: `3186`)*.
-   `--act_port`: port used for the action stream on the evaluation server *(default: `3188`)*.

## License 🗒️

This project is licensed under  [NTU S-Lab License 1.0](https://github.com/hzxie/DynamicVLA/blob/master/LICENSE). Redistribution and use should follow this license.
