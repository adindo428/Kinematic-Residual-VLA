#!/usr/bin/env bash
set -euo pipefail
module load ffmpeg/6.1.1
cd /work/nvme/bivv/aparameshwaran/dynamicvla/pi05-comparison-20261021
export PYTHONPATH="$PWD/pinned:$PWD/source:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/work/nvme/bivv/aparameshwaran/dynamicvla/native-krvla-dom/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GLOO_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1
env/bin/python -c 'import torch; assert torch.cuda.device_count()==2; print(torch.__version__,[torch.cuda.get_device_name(i) for i in range(2)])'
sha256sum checkpoint.pt > checkpoint-verified.sha256
grep -q '^09c0466b8809bc8e5f0dc248da135bcb186c64bb0759557def0bfdaee9119e6c ' checkpoint-verified.sha256
for variant in control cpu_tokens cpu_tokens_prefetch workers12_prefetch autocast_cache no_recompute control_repeat; do
  # A candidate OOM or equivalence failure is evidence, not a reason to skip the controls.
  set +e
  timeout --signal=TERM --kill-after=30s 14m env/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 benchmark.py "$variant" > "$variant.log" 2>&1
  status=$?
  set -e
  printf '{"variant":"%s","exit_code":%s}\n' "$variant" "$status" > "$variant-exit.json"
  if [[ "$variant" == control && "$status" != 0 ]]; then exit "$status"; fi
done
env/bin/python - <<'PY'
import json,pathlib,datetime
p=pathlib.Path('.')
(p/'sweep-complete.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exits':[json.loads(f.read_text()) for f in p.glob('*-exit.json')],'results':[json.loads(f.read_text()) for f in p.glob('*-result.json')],'production_changed':False},indent=2))
PY
