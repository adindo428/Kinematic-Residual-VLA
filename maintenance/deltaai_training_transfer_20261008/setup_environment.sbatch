#!/bin/bash
#SBATCH --account=bivv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=krvla-env
#SBATCH --output=/work/nvme/bivv/aparameshwaran/dynamicvla/setup-20261008/environment-%j.log
set -euo pipefail
module load python/miniforge3_pytorch/2.12.0
ROOT=/work/nvme/bivv/aparameshwaran/dynamicvla
export HF_HOME=$ROOT/huggingface
export XDG_CACHE_HOME=$ROOT/cache
export CONDA_PKGS_DIRS=$ROOT/cache/conda-pkgs
ENV=$ROOT/venv-krvla-dom-20261008
python -m venv --system-site-packages "$ENV"
source "$ENV/bin/activate"
NATIVE=$ROOT/native-krvla-dom
if [[ ! -f "$NATIVE/include/libmemcached/memcached.h" ]]; then
  /sw/user/python/miniforge3-pytorch-2.12.0/bin/conda create --yes --prefix "$NATIVE" --override-channels -c conda-forge libmemcached
fi
export LIBMEMCACHED=$NATIVE
if [[ ! -f "$NATIVE/lib/libavutil.so.60" ]]; then
  /sw/user/python/miniforge3-pytorch-2.12.0/bin/conda install --yes --prefix "$NATIVE" --override-channels -c conda-forge 'ffmpeg=8'
fi
export LD_LIBRARY_PATH="$NATIVE/lib:${LD_LIBRARY_PATH:-}"
export LDFLAGS="-Wl,-rpath,$NATIVE/lib ${LDFLAGS:-}"
python -m pip install --no-deps pylibmc==1.6.3
printf 'torch==2.12.0+cu130\ntorchvision==0.27.0+cu130\n' > "$ROOT/setup-20261008/torch-constraints.txt"
python -m pip install --no-deps lerobot==0.3.3 torchcodec==0.16.0 av==18.1.0 easydict==1.13 draccus==0.10.0 num2words==0.5.14 jsonlines==4.0.0 deepdiff==8.6.2 gymnasium==0.29.1 imageio==2.38.0 imageio-ffmpeg==0.6.0
python -m pip install --no-deps pyserial==3.5 termcolor==3.3.0 rerun-sdk==0.22.1 pynput==1.8.2 python-xlib==0.33 evdev==2.0.0 toml==0.10.2
python -m pip install -c "$ROOT/setup-20261008/torch-constraints.txt" transformers==5.2.0 docopt typing-inspect mergedeep pyyaml-include==1.4.1 orderly-set Farama-Notifications
python -m pip freeze > "$ROOT/setup-20261008/environment-freeze.txt"
python - <<'PY'
import json, platform, importlib, sys, subprocess
from pathlib import Path
import torch
assert platform.machine()=='aarch64', platform.machine()
assert torch.cuda.is_available(), 'GPU unavailable'
report={'machine':platform.machine(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'imports':{}}
source=Path('/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261008T152033Z')
for tree in ('dom100','tracking'):
    modules=['torchcodec.decoders','utils.datasets','core.train','policies.dynamicvla.modeling_dynamicvla']
    if tree=='dom100': modules.append('core.dom_training')
    # Each checkout registers its own policy class with LeRobot. Production
    # uses one checkout per process, so verify them in separate processes.
    code='import importlib; '+ '; '.join('importlib.import_module('+repr(m)+')' for m in modules)
    subprocess.run([sys.executable,'-c',code],cwd=source/tree,check=True)
    for module in modules:
        report['imports'][tree+'/'+module]='ok'
x=torch.randn(64,64,device='cuda'); y=x@x; torch.cuda.synchronize()
from torchcodec.decoders import VideoDecoder
video=next(Path('/work/nvme/bivv/aparameshwaran/dynamicvla/huggingface/lerobot/adi2440/dom-franka-tracking/videos').rglob('*.mp4'))
decoder=VideoDecoder(str(video),seek_mode='approximate',num_ffmpeg_threads=1)
frame=decoder.get_frame_at(0)
report['video_check']={'path':str(video),'frame_shape':list(frame.data.shape)}
Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-20261008/environment-report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
PY
