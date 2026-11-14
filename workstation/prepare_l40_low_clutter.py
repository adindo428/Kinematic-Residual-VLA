"""Stage an exact Sim-4.5 deployment with mandatory unscored latency gates."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO/'scratch/dom45/evaluations/pi05-kr-l40-20261111'
STAGE = LOCAL/'stage'
REMOTE = Path('/scratch/aparame/Dynamic_VLA/pi05-kr-l40-lowclutter-20261111')
WORK = REMOTE/'scratch/dom45'
LOW = REPO/'scratch/dom45/evaluations/low-clutter-full-table1-20261102'
RLOW = WORK/'evaluations/low-clutter-full-table1-20261102'
CONTROL = RLOW/'l40_suite'
KEY = 'pi05_kr_epoch3_partial4_l40'


def read(p): return json.loads(Path(p).read_text())
def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(x, indent=2)+'\n')
def target(p): return STAGE/Path(p).relative_to(REMOTE)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def copytree(src, dest):
    shutil.copytree(src, dest, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc', '.pytest_cache'))


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    copytree(REPO/'workstation', STAGE/'workstation')
    copytree(REPO/'evaluation/official_dom/scripts', STAGE/'evaluation/official_dom/scripts')
    copytree(REPO/'scratch/dom45/official/source', target(WORK/'official/source'))
    copytree(REPO/'scratch/dom45/IsaacLab-2.2.1', STAGE/'IsaacLab-2.2.1')
    copytree(REPO/'scratch/dom45/assets/isaac', target(WORK/'assets/isaac'))
    copytree(LOW/'configs', target(RLOW/'configs'))
    (STAGE/'workstation/isaaclab-paper45.sh').write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        f'export ISAACLAB_ROOT={REMOTE}/IsaacLab-2.2.1\n'
        f'export CONDA_PREFIX={REMOTE}/envs/sim45\n'
        'unset EXP_PATH ISAAC_PATH ISAACSIM_PATH CARB_APP_PATH PYTHONHOME\n'
        'if [[ "${1:-}" == -p || "${1:-}" == --python ]]; then shift; fi\n'
        'exec "$CONDA_PREFIX/bin/python" "$@"\n')
    (STAGE/'evaluate.sbatch').write_text(f'''#!/usr/bin/env bash
#SBATCH --account=omniverse_dndawso_manual_users
#SBATCH --partition=omniverse
#SBATCH --qos=omniverse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --constraint=gpu_l40
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=pi05kr-eval-sim45-L40
#SBATCH --output={REMOTE}/gpu-%j.log
set -euo pipefail
test -n "${{SLURM_JOB_ID:-}}" && test -n "${{SLURMD_NODENAME:-}}"
cd {REMOTE}
export APPTAINERENV_SLURM_JOB_ID="$SLURM_JOB_ID"
export APPTAINERENV_SLURMD_NODENAME="$SLURMD_NODENAME"
exec /usr/bin/apptainer exec --nv --cleanenv \\
 --bind /scratch/aparame:/scratch/aparame,/home/aparame:/home/aparame \\
 /scratch/aparame/isaaclab.sif env -u EXP_PATH -u ISAAC_PATH -u ISAACSIM_PATH -u CARB_APP_PATH -u PYTHONHOME \\
 PYTHONPATH={REMOTE}/workstation:{REMOTE}/evaluation/official_dom/scripts \\
 LD_LIBRARY_PATH=/.singularity.d/libs:{REMOTE}/envs/sim45/lib:/home/aparame/.conda/envs/dynamicvla/lib \\
 OMNI_KIT_ACCEPT_EULA=YES HF_HOME=/scratch/aparame/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 \\
 /scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/env/bin/python -u {REMOTE}/workstation/run_l40_low_clutter.py
''')
    source_code = REPO/'scratch/dom45/evaluations/multivla20k-20261029/model_code/pi05_kr'
    code = REMOTE/'model_code/pi05_kr'; copytree(source_code, target(code))
    write(target(code/'checkpoints.local.json'), read(LOCAL/'remote_checkpoint_references.json'))
    for name in ['sim45.tar.gz', 'inference_checkpoint.pt']:
        p = STAGE/name
        if not p.exists(): p.symlink_to(LOCAL/name)
    receipt = read(LOCAL/'inference_checkpoint_receipt.json')
    entry = dict(read(LOW/'pi05-kr-epoch003-partial-next-20261111/plan.json')['models'][0]['entry'])
    entry.update(checkpoint=str(REMOTE/'inference_checkpoint.pt'), weight_path=str(REMOTE/'inference_checkpoint.pt'),
        checkpoint_sha256=receipt['inference_checkpoint_sha256'], sha256=receipt['inference_checkpoint_sha256'],
        checkpoint_bytes=receipt['bytes'], weight_bytes=receipt['bytes'], code_root=str(code),
        recipe=str(code/'recipes/pi05_kr.json'), original_checkpoint_sha256=receipt['source_checkpoint_sha256'],
        label='pi0.5 + KR (3.208 epochs; L40 qualified deployment)', backend='Palmetto single L40')
    protocol = read(LOW/'protocol.json')
    protocol['tests'] = [{**t, 'path': str(RLOW/'configs'/Path(t['path']).name)} for t in protocol['tests']]
    write(target(RLOW/'protocol.json'), protocol)
    lists = {dim: [str(RLOW/'configs'/Path(s).name) for s in (LOW/'test_lists'/(dim+'.txt')).read_text().splitlines() if s]
             for dim in ['CR', 'DA', 'LS', 'VU', 'SR', 'MP', 'VG', 'MG', 'DR']}
    for dim, paths in lists.items():
        p = target(RLOW/'test_lists'/(dim+'.txt')); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('\n'.join(paths)+'\n')
    tests = {t['name']: t for t in protocol['tests']}
    jobs = []
    pilot_scenes = [lists[dim][0] for dim in ['CR', 'LS', 'SR']]
    for case, scene in enumerate(pilot_scenes):
        for repeat in range(2):
            key = f'pilot-{case}-{repeat}'
            jobs.append(dict(id=key, model=KEY, scene=scene, scene_sha256=tests[Path(scene).stem]['config_sha256'],
                repeats=1, repeat_offset=0, folder=str(CONTROL/'blocks'/key), dimension=['CR','LS','SR'][case], scored=False))
    for pass_index, offset in [(1, 0), (2, 10)]:
        for dim, paths in lists.items():
            for scene in paths:
                key = f'pass{pass_index}-{dim}-{Path(scene).stem}'
                jobs.append(dict(id=key, model=KEY, scene=scene, scene_sha256=tests[Path(scene).stem]['config_sha256'],
                    repeats=10, repeat_offset=offset, folder=str(CONTROL/'blocks'/key), dimension=dim,
                    scored=True, pass_index=pass_index))
    for j in jobs:
        p = target(Path(j['folder'])/'case.txt'); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(j['scene']+'\n')
    # The node has one physical GPU; simulation and policy share it. This is
    # explicitly qualified against A6000 evidence before any scored trial.
    for filename in ['run_resident_adapter_table1.py', 'resident_adapter_table1.py']:
        p = STAGE/'workstation'/filename; text = p.read_text()
        text = text.replace("CUDA_VISIBLE_DEVICES='1'", "CUDA_VISIBLE_DEVICES='0'")
        text = text.replace("int(line.split(',')[0]) == 1", "int(line.split(',')[0]) == 0")
        p.write_text(text)
    p = STAGE/'workstation/resident_adapter_table1_client.py'; text = p.read_text()
    text = text.replace("        pool.qualify_all(args.output)",
        "        pool.qualify_all(args.output)\n        from l40_latency_gate import check_dummy, check_pilots\n        check_dummy(plan, pool.profile, args.output)")
    text = text.replace("write(args.output/'pilot_verification.json', verify_pilots(plan)); qualified_scene = True",
        "write(args.output/'pilot_verification.json', verify_pilots(plan)); check_pilots(plan, args.output); qualified_scene = True")
    p.write_text(text)
    sys.path.insert(0, str(REPO/'workstation'))
    from capture_table1_runtime import stats
    from monitor_table1_latency import log_samples
    records = [json.loads(s) for s in (LOW/'resident_adapters-20261108/aggregates/pi05_kr/trial_results.jsonl').read_text().splitlines()]
    references = {}
    for scene in pilot_scenes:
        folders = sorted({r['source'] for r in records if r['env_name'] == Path(scene).stem})
        samples = [s for folder in folders for s in log_samples(Path(folder)/'client.log')]
        stale = [s['stale'] for s in samples if s['stale'] is not None]
        assert len(samples) >= 30 and stale
        references[scene] = {**stats([s['ms'] for s in samples]),
            'expired_chunk_fraction': sum(v >= 20 for v in stale)/len(stale), 'sources': folders}
    previous = read(LOW/'resident_adapters-20261108/archives/before-vlash-baseline-20261110T223511/execution/resident_profile.json')
    dummy = previous['paired_trials'][-1]['models']['pi05_kr']['single']
    ref = {'dummy': dummy, 'cases': references,
           'scope': 'Same PI05 architecture, preprocessing and shapes; A6000 reference checkpoint is epoch 6.453, not identical epoch-3.208 weights. Task references use the same frozen scenarios.'}
    catalog = dict(models={KEY: entry}, order=[KEY], expected_gpu='L40',
        python='/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/env/bin/python',
        official_source=str(WORK/'official/source'), vlash_source='/scratch/aparame/Dynamic_VLA/table1-multivla-20261026/vlash_source',
        active_resident_adapter_root=str(CONTROL))
    write(target(RLOW/'catalog.json'), catalog)
    copy = target(RLOW/'table1_comparison.md'); copy.write_text('# Palmetto qualified L40 evaluation\n\n| Method | Status | CR | DA | LS | VU | SR | MP | VG | MG | DR | Success | Time |\n|---|---|'+'---:|'*11+'\n')
    plan = dict(models=[dict(key=KEY, entry=entry, catalog={k: catalog[k] for k in ['official_source','vlash_source']})],
        catalog={k: catalog[k] for k in ['official_source','vlash_source']}, jobs=jobs,
        success_history={KEY: []}, a6000_reference=ref, new_scored_trials=1800,
        protocol_sha256=sha(target(RLOW/'protocol.json')), minimum_latency_retention=.9)
    plan['files_sha256'] = {str(REMOTE/p.relative_to(STAGE)): sha(p)
        for folder in [STAGE/'workstation', target(code)] for p in folder.rglob('*')
        if p.is_file() and p.suffix in ['.py', '.json']}
    write(target(CONTROL/'plan.json'), plan)
    write(target(CONTROL/'existing.json'), {'records': {KEY: []}, 'sources': []})
    write(target(CONTROL/'request.json'), {'enabled': True, 'notify_on_completion': False})
    write(target(CONTROL/'checkpoint_policy.json'), {'minimum_completed_epochs': 10, 'held_models': {},
        'authorized_exceptions': {KEY: {'user_authorized': True, 'checkpoint_sha256': entry['checkpoint_sha256']}}})
    write(STAGE/'deployment.json', {'root': str(REMOTE), 'work': str(WORK), 'low': str(RLOW),
        'control': str(CONTROL), 'entry': entry, 'expected_sim': '4.5.0', 'expected_lab': '2.2.1',
        'a6000_reference': ref, 'local_source_protocol_sha256': sha(LOW/'protocol.json')})
    print('Staged pinned deployment; 6 excluded pilots; 1,800 trials only after both latency gates pass.')


if __name__ == '__main__': main()
