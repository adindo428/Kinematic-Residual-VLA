"""Stage matched old-20k distance-gated cross-VLA continuations and missing data."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'scratch/dom45/training/original20k-distance-arai-20261113'
OLD = ROOT/'scratch/dom45/training/low-clutter-vcg-arai-20261112'
LOCAL = ROOT/'scratch/dom45/training/krv-distance-gate-old20k-20261112/family'
REMOTE = '/scratch/arai/Dynamic_VLA/original20k-distance-20261113'
BASE = '/scratch/arai/Dynamic_VLA/low-clutter-velocity-gated-20261112'

def read(path): return json.loads(path.read_text())
def write(path,value): path.write_text(json.dumps(value,indent=2)+'\n')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    assert not (STAGE/'run-plan.json').exists()
    for name in ['source','recipes','runtimes','scripts']:
        shutil.copytree(OLD/name,STAGE/name,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    for name in ['env.sh','node.sh','supervise.py']:
        (STAGE/name).write_text((OLD/name).read_text().replace(BASE,REMOTE))
    for name in ['split.json','parquet_hashes.json','video_hashes.json']:
        shutil.copyfile(LOCAL/name,STAGE/name)
    manifest=read(LOCAL/'manifest.json'); split=read(STAGE/'split.json')['splits']
    write(STAGE/'conditioning-statistics.json',dict(mean=manifest['velocity_mean'],std=manifest['velocity_std'],
        frames=manifest['train_frames'],source_split_sha256=manifest['split_sha256']))
    # Keep the parent checkpoints' state/action normalization and analytic mean.
    data=read(STAGE/'base-data-provenance.json')
    data.update(dataset_root=BASE+'/dataset',split_sha256=manifest['split_sha256'],
        data_archive=REMOTE+'/dataset-overlay.tar',data_archive_index=REMOTE+'/dataset-tar-index.json',
        data_archives=dict(base=BASE+'/dataset.tar',overlay=REMOTE+'/dataset-overlay.tar'),
        phase_gate='distance_threshold; current observed EE/object positions only',distance_gate_radius_m=.05,
        conditioning_statistics='Original 20k train-only velocity statistics; parent physical normalization/analytic centering preserved')
    write(STAGE/'data-provenance.json',data)
    source=STAGE/'source'
    text=(source/'storage.py').read_text().replace('        self.index=json.loads',
        '        self.archives={k:Path(v) for k,v in manifest.get("data_archives",{}).items()}\n        self.index=json.loads')
    text=text.replace("        with self.archive.open('rb') as stream:","        archive=self.archives[entry['archive']] if 'archive' in entry else self.archive\n        with archive.open('rb') as stream:")
    (source/'storage.py').write_text(text)
    # Share the exact torch gate with the local DynamicVLA implementation.
    shutil.copyfile(LOCAL.parent/'source/proactive_vla/distance_gate.py',source/'distance_gate.py')
    text=(source/'data.py').read_text().replace('from phase_gate import replay_prefix_states, gate_from_state, PhaseGateConfig',
                                            'from distance_gate import current_distance_gate')
    text=text.replace('        self.gate_config = PhaseGateConfig()','')
    a=text.index('            compact, phases = replay_prefix_states(');b=text.index('            decoders = {}',a)
    text=text[:a]+text[b:]
    text=text.replace("'phase_gate': torch.from_numpy(gate_from_state(values['_gate_state'][frame], self.horizon+delay, .04)[delay:]),\n                'phase_id': int(values['_phase_id'][frame])",'''"gate_ee_position": torch.from_numpy(values['observation.state'][frame,:3].copy()),
                "gate_object_position": torch.from_numpy(values['observation.environment_state'][frame,:3].copy())''')
    (source/'data.py').write_text(text)
    text=(source/'policy.py').read_text().replace("recipe.get('phase_gate') == 'causal_rules'", "recipe.get('phase_gate') == 'distance_threshold'")
    text=text.replace("        gate = b['phase_gate']",'''        from distance_gate import current_distance_gate
        gate = current_distance_gate(b['gate_ee_position'], b['gate_object_position'],
                                    self.recipe['distance_gate_radius_m'], 20, 1, device)''')
    (source/'policy.py').write_text(text)
    text=(source/'velocity_checks.py').read_text().replace("    a['phase_gate'] = b['phase_gate'] = torch.zeros(count, 20, 1)",
        "    a['gate_object_position'] = b['gate_object_position'] = batch['gate_ee_position'].clone()")
    text=text.replace("    full = {**b, 'phase_gate': torch.ones(count, 20, 1)}", "    full = {**b, 'gate_object_position': b['gate_ee_position'] + torch.tensor([.2,0.,0.])}")
    (source/'velocity_checks.py').write_text(text)
    text=(source/'train.py').read_text().replace("'phase_gate':'causal_rules'","'phase_gate':recipe['phase_gate'],'distance_gate_radius_m':.05")
    (source/'train.py').write_text(text)
    models=['smolvla_kr','pi05_kr','vlash_kr']
    for model in models:
        path=STAGE/'recipes'/(model+'.json');recipe=read(path)
        recipe.update(phase_gate='distance_threshold',distance_gate_radius_m=.05,
            training_anchors=manifest['train_frames'],subset_sha256=manifest['split_sha256'],replica=False,
            budget_authorization='2026-11-13: user requested original local 20k split, 5cm gate and resubmission, four A100 GPUs each')
        recipe['steps']=recipe['parent_updates']+recipe['min_epochs']*(recipe['training_anchors']//recipe['global_batch'])
        recipe['profiling_reference']['scope']='Historical A100 measurement; original20k/distance-gate/source differ, bounded in-job validation required'
        write(path,recipe)
        path=STAGE/'scripts'/(model+'.sbatch');path.write_text(path.read_text().replace(BASE,REMOTE).replace('lc20k-vcg-','old20k-dg-'))
    plan=read(OLD/'run-plan.json');plan.update(dataset_split_sha256=manifest['split_sha256'],
        training_frames=manifest['train_frames'],kr='5cm current observation distance gate, multiplying centered drift in targets and reconstruction',
        distance_gate_radius_m=.05,local_reference=str(LOCAL.parent),remote_root=REMOTE,
        prior_attempt=BASE,initializer='Same own trained parent checkpoints and Adam/schedule as prior arai submission',
        comparability='Exact local 20k membership and gate; backbones, prior training, optimizers and physical normalization differ by model')
    # Files already on arai retain their immutable original archive offsets.
    base_index=read(STAGE/'base-dataset-index.json');index={};missing=[]
    for ep in sorted(set(sum(split.values(),[]))):
        names=[f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet']
        names += [f'videos/chunk-{ep//1000:03d}/{cam}/episode_{ep:06d}.mp4' for cam in ['observation.images.opst_cam','observation.images.wrist_cam']]
        for name in names:
            if name in base_index:index[name]={**base_index[name],'archive':'base'}
            else:missing.append(name)
    dataset=Path(manifest['dataset_root']);archive=STAGE/'dataset-overlay.tar'
    with tarfile.open(archive,'w',format=tarfile.USTAR_FORMAT) as tar:
        for i,name in enumerate(missing):
            path=dataset/name;size=path.stat().st_size;info=tarfile.TarInfo(name);info.size=size;info.mode=0o644
            offset=tar.offset+512
            with path.open('rb') as f:tar.addfile(info,f)
            index[name]=dict(offset=offset,size=size,archive='overlay')
            if i%3000==0:print('Packed missing files',i+1,'/',len(missing),flush=True)
    write(STAGE/'dataset-tar-index.json',index)
    receipt=dict(complete=True,files=len(index),overlay_files=len(missing),reused_base_files=len(index)-len(missing),
        bytes=archive.stat().st_size,archive_sha256=sha(archive),split_sha256=sha(STAGE/'split.json'),storage='layered indexed tar',
        base_archive_sha256='43c9d1bfe92d90862a4431d68e28f37a04cd080d8f49549cac03ded06ec49a90')
    write(STAGE/'transfer-receipt.json',receipt);write(STAGE/'run-plan.json',plan)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
