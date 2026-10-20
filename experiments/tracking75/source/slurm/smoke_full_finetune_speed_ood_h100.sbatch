#!/usr/bin/env bash
#SBATCH --job-name=dvla-speed-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --mem=64GB
#SBATCH --output=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.out
#SBATCH --error=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.err

set -euo pipefail
cd /home/aparame/Research/Dynamic_VLA/third_party/DynamicVLA
source scripts/palmetto_env.sh
python scripts/rebuild_dom_speed_split.py
python scripts/check_train_readiness.py

smoke_root="${DVLA_OUTPUT_ROOT}/smoke/${SLURM_JOB_ID:?}"
mkdir -p "${smoke_root}"
python scripts/make_speed_smoke_split.py \
  --output "${smoke_root}/split.json" --train-episodes 8
python - "${smoke_root}" <<'PY'
import sys
from pathlib import Path

import yaml

smoke_root = Path(sys.argv[1])
config = yaml.safe_load(Path("configs/dynamicvla_dom_full_finetune_speed_ood_0p10_0p18.yaml").read_text())
config["DATASET"]["EPISODE_SPLIT_FILE"] = str(smoke_root / "split.json")
config["WANDB"]["ENABLED"] = False
config["WANDB"]["LOG_CODE"] = False
config["DIR"]["OUTPUT"] = str(smoke_root)
config["TRAIN"]["N_EPOCHS"] = 1
config["TRAIN"]["BATCH_SIZE"] = 1
config["TRAIN"]["GRAD_ACCUM_STEPS"] = 1
config["TRAIN"]["CKPT_SAVE_FREQ"]["EPOCH"] = 1
(smoke_root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
PY
export DVLA_OUTPUT_ROOT="${smoke_root}"
export DVLA_N_WORKERS=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
python run.py --cfg "${smoke_root}/config.yaml" --exp "speed-smoke-${SLURM_JOB_ID}"
