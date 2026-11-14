"""GPU-allocation entry point; version checks and latency gates precede scoring."""
import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path('/scratch/aparame/Dynamic_VLA/pi05-kr-l40-lowclutter-20261111')


def main():
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURMD_NODENAME')
    assert 'login' not in os.uname().nodename
    d = json.loads((ROOT/'deployment.json').read_text())
    assert (ROOT/'initialized.json').exists()
    # This entry uses the remote policy environment. Simulator metadata is
    # checked by its own Python before starting any Isaac application.
    import subprocess
    simpy = ROOT/'envs/sim45/bin/python'
    probe = subprocess.check_output([str(simpy), '-c',
        "import importlib.metadata as m,json; print(json.dumps({'isaacsim':m.version('isaacsim')}))"], text=True)
    version = json.loads(probe.strip().splitlines()[-1])['isaacsim']
    assert version == '4.5.0', f'Refusing simulator {version}'
    assert (ROOT/'IsaacLab-2.2.1/VERSION').read_text().strip() == '2.2.1'
    import torch
    assert torch.cuda.device_count() == 1
    gpu = torch.cuda.get_device_name(0)
    assert 'L40' in gpu and 'L40S' not in gpu
    assert torch.cuda.get_device_properties(0).total_memory >= 44*1024**3
    (ROOT/'gpu_runtime.json').write_text(json.dumps({'job': os.environ['SLURM_JOB_ID'],
        'host': os.uname().nodename, 'gpu': gpu, 'simulator': version, 'isaaclab': '2.2.1',
        'policy_torch': torch.__version__, 'simulation_and_policy_share_one_gpu': True}, indent=2)+'\n')
    os.environ['DOM_RESIDENT_ADAPTER_ROOT'] = d['control']
    os.environ['ISAACLAB_ROOT'] = str(ROOT/'IsaacLab-2.2.1')
    work = Path(d['work'])
    os.environ['KRVLA_KIT_ARGS'] = f'--portable --portable-root={work}/kit --/persistent/isaac/asset_root/cloud={work}/assets/isaac --/persistent/isaac/asset_root/default={work}/assets/isaac --/persistent/isaac/asset_root/nvidia={work}/assets/isaac'
    import run_resident_adapter_table1 as runner
    runner.main()


if __name__ == '__main__': main()
