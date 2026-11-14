"""Run inside the CPU allocation after transferring the exact local runtime."""
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile

ROOT = Path('/scratch/aparame/Dynamic_VLA/pi05-kr-l40-lowclutter-20261111')


def main():
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURMD_NODENAME')
    assert 'login' not in os.uname().nodename
    deployment = json.loads((ROOT/'deployment.json').read_text())
    (ROOT/'workstation/isaaclab-paper45.sh').chmod(0o755)
    env = ROOT/'envs/sim45'
    if not (env/'bin/python').exists():
        env.mkdir(parents=True, exist_ok=True)
        with tarfile.open(ROOT/'sim45.tar.gz') as tar: tar.extractall(env)
    old_lab = '/media/ailab-students/Linux_SSD/projects/Dynamic_VLA/scratch/dom45/IsaacLab-2.2.1'
    for p in (env/'lib/python3.10/site-packages').glob('__editable__*'):
        if p.suffix not in ['.py', '.pth']: continue
        text = p.read_text(); p.write_text(text.replace(old_lab, str(ROOT/'IsaacLab-2.2.1')))
    work = Path(deployment['work'])
    for sub in ['objects', 'scenes']:
        dest = work/'assets'/sub
        if not dest.exists(): dest.symlink_to(Path('/scratch/aparame/Dynamic_VLA/dom_benchmark')/sub)
    checkpoint = ROOT/'inference_checkpoint.pt'; s = checkpoint.stat()
    control = Path(deployment['control']); plan = json.loads((control/'plan.json').read_text())
    entry = plan['models'][0]['entry']
    entry.update(checkpoint_bytes=s.st_size, weight_bytes=s.st_size,
                 checkpoint_mtime_ns=s.st_mtime_ns, weight_mtime_ns=s.st_mtime_ns)
    (control/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    catalog_path = Path(deployment['low'])/'catalog.json'; catalog = json.loads(catalog_path.read_text())
    catalog['models'][plan['models'][0]['key']] = entry
    catalog_path.write_text(json.dumps(catalog, indent=2)+'\n')
    (ROOT/'initialized.json').write_text(json.dumps({'complete': True,
        'cpu_job': os.environ['SLURM_JOB_ID'], 'host': os.uname().nodename,
        'simulator_expected': '4.5.0', 'isaaclab_expected': '2.2.1'}, indent=2)+'\n')


if __name__ == '__main__': main()
