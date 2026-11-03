#!/usr/bin/env python3
"""Exercise actual CUDA kernels and decoder libraries in the policy environment."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


def run_checks():
    import torch
    import torchvision
    from torchcodec.decoders import VideoDecoder
    if platform.system() != 'Linux' or sys.version_info[:2] != (3, 10):
        raise RuntimeError('Expected Linux and a dedicated Python 3.10 policy environment')
    expected = {'torch': '2.7.1', 'torchvision': '0.22.1', 'torchcodec': '0.5',
                'transformers': '5.2.0', 'lerobot': '0.3.3', 'timm': '1.0.27'}
    versions = {key: importlib.metadata.version(key) for key in expected}
    for key, value in expected.items():
        if versions[key].split('+')[0] != value:
            raise RuntimeError(f'{key}: expected {value}, got {versions[key]}')
    if not torch.cuda.is_available():
        raise RuntimeError('No visible CUDA GPU')
    capability = torch.cuda.get_device_capability(0)
    cuda_version = tuple(map(int, (torch.version.cuda or '0.0').split('.')[:2]))
    if capability[0] >= 12 and cuda_version < (12, 8):
        raise RuntimeError('Blackwell requires the cu128 policy build; rerun setup_policy.sh')
    x = torch.randn(64, 64, device='cuda')
    if not torch.isfinite(x @ x.T).all().item():
        raise RuntimeError('Non-finite CUDA matrix result')
    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]], device='cuda')
    kept = torchvision.ops.nms(boxes, torch.tensor([.9, .8], device='cuda'), .5)
    if kept.tolist() != [0]:
        raise RuntimeError('TorchVision CUDA operator failed')
    torch.cuda.synchronize()
    # A real tiny decode catches missing FFmpeg shared libraries and ABI mismatches.
    with tempfile.TemporaryDirectory(prefix='krvla-decoder-') as directory:
        path = Path(directory)/'probe.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                        'color=c=red:s=32x32:r=5:d=1', '-c:v', 'mpeg4', str(path)], check=True)
        decoder = VideoDecoder(str(path), device='cpu')
        if len(decoder) < 1 or tuple(decoder[0].shape) != (3, 32, 32):
            raise RuntimeError('TorchCodec decode failed')
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    return {'validated': True, 'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'dependencies': versions,
            'gpu': torch.cuda.get_device_name(0), 'compute_capability': list(capability),
            'torch_cuda': torch.version.cuda, 'compiled_architectures': torch.cuda.get_arch_list(),
            'device_total_bytes': torch.cuda.get_device_properties(0).total_memory,
            'nvidia_smi': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                                   '--format=csv,noheader'], text=True),
            'cuda_matrix_and_torchvision_ops': True, 'cpu_video_decode': True,
            'scope': 'Policy runtime only; does not validate Isaac rendering or manipulation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_checks()
    except Exception as error:
        report = {'validated': False, 'error': str(error), 'traceback': traceback.format_exc()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if not report['validated']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
