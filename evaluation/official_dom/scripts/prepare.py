"""Freeze official assets, weights and source; prepare the unmodified DOM tests."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import zipfile

DIMENSIONS = {'1-1':'CR','1-2':'DA','1-3':'LS','2-1':'VU','2-2':'SR',
              '2-3':'MP','3-1':'VG','3-2':'MG','3-3':'DR'}
PAPER = dict(CR=60.5, DA=38.5, LS=40.5, VU=51.5, SR=48., MP=33.5,
             VG=59.5, MG=65., DR=26.5, Success=47.06, Time=8.53)

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def prepare(stage,root,assets,gpu_label="L40S",source_commit='434dc7105cf6d559018d15af5fecbcdad44f036e'):
    stage,root,assets=map(Path,(stage,root,assets))
    root.mkdir(parents=True,exist_ok=True)
    source=stage/'source';ckpt=stage/'checkpoint'
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    assert commit==source_commit
    assert not subprocess.check_output(['git','-C',str(source),'diff','--name-only'],text=True).strip()
    cfg=json.loads((ckpt/'config.json').read_text())
    assert cfg['type']=='dynamicvla' and cfg['delta_timestamps']['observation']==[-2,0]
    assert cfg['chunk_size']==cfg['n_action_steps']==20 and cfg['use_delta_action']
    names=[Path(line).name for line in (assets/'test-envs.txt').read_text().splitlines() if line.strip()]
    assert len(names)==len(set(names))==90
    assert Counter(name[:3] for name in names)==Counter({k:10 for k in DIMENSIONS})
    # Verify that no previous local experiment modified the released test JSONs.
    with zipfile.ZipFile(assets/'DOM-Test.zip') as archive:
        members={Path(name).name:name for name in archive.namelist() if name.endswith('.json')}
        for name in names:
            assert (assets/'tests'/name).read_bytes()==archive.read(members[name]),name
    required={};tests=[];scenes=set()
    for name in names:
        path=assets/'tests'/name;d=json.loads(path.read_text())
        assert '_franka_' in name
        scene=assets/'scenes'/Path(d['scene']['house']['spawn']['usd_path']).name
        assert scene.is_file(),scene
        scenes.add(str(scene));required[str(scene)]=sha256(scene)
        for v in d['scene'].values():
            if not isinstance(v,dict) or v.get('class_type')!='isaaclab.assets.rigid_object.rigid_object:RigidObject':continue
            if 'usd_path' not in v['spawn']:continue
            usd=Path(v['spawn']['usd_path']);p=assets/'objects'/usd.parent.name/usd.name
            assert p.is_file(),p
            required[str(p)]=sha256(p)
        tests.append({'name':Path(name).stem,'path':str(path),'dimension':DIMENSIONS[name[:3]],
                      'config_sha256':sha256(path),'seed':d['seed'],'horizon_seconds':d['episode_length_s']})
    lists=root/'test_lists';lists.mkdir(exist_ok=True)
    for index,dimension in enumerate(DIMENSIONS.values()):
        selected=[t['path'] for t in tests if t['dimension']==dimension]
        (lists/f'{index:02d}_{dimension}.txt').write_text('\n'.join(selected)+'\n')
    pilot=[next(t['path'] for t in tests if t['dimension']==d) for d in DIMENSIONS.values()]
    (lists/'pilot.txt').write_text('\n'.join(pilot)+'\n')
    (lists/'all.txt').write_text('\n'.join(t['path'] for t in tests)+'\n')
    deps={name:importlib.metadata.version(name) for name in ('torch','transformers','lerobot','timm','huggingface-hub')}
    protocol={'benchmark':'Official DOM manipulation benchmark','checkpoint_repo':'hzxie/dynamic-vla-DOM',
        'checkpoint_revision':'c9609d94c5940eb42436a31927d7e80ae454279e','checkpoint':str(ckpt),
        'checkpoint_config_sha256':sha256(ckpt/'config.json'),'checkpoint_weights_sha256':sha256(ckpt/'model.safetensors'),
        'official_repository':'https://github.com/hzxie/DynamicVLA','official_source_commit':commit,
        'requested_runtime':({'isaacsim':'4.5.0','isaaclab':'2.2.1'}
                             if commit=='a27a06d2ca74d0e987a5e552e01013073e93cfd8'
                             else {'isaacsim':'5.1.0','isaaclab':'2.3.2'}),
        'paper':'https://arxiv.org/abs/2601.22153','paper_version':'2601.22153v2',
        'paper_reference':PAPER,'requested_gpu':gpu_label,'gpu_label':gpu_label,'expected_trials':1800,'trials_per_configuration':20,
        'configurations':90,'dimension_order':list(DIMENSIONS.values()),'configurations_per_dimension':10,
        'unique_scene_usd_files_in_release':len(scenes),'tests':tests,'dependencies':deps,
        'test_jsons_identical_to_official_archive':True,'asset_sha256':required,
        'rotation':'euler','delta_actions':True,'continuous_inference_and_latency_aware_streaming':True,
        'observation_offsets':[-2,0],'physics_time_step_s':.04,
        'success_definition':'Official simulator termination: instructed pick/place completed, rather than a tracking-distance criterion.',
        'time_definition':'Simulation time from motion onset to termination, including successful completion and failed trials. Recorded as simulator step count times actual step_dt.',
        'seed_policy':'Unmodified official reset and instruction randomization; released per-configuration seed retained.',
        'recording':'Thin observer wraps simulator calls to record status, step count and time after each trial; policy, physics, actions and termination logic remain upstream code.',
        'limitations':['Online inference latency can change outcomes; compare the recorded GPU/runtime configuration with the paper.',
                       'No claim that the released weights must exactly reproduce the paper; report all per-dimension deviations.',
                       'Twenty repeats retain the upstream reset seed behavior; repeat independence is not assumed.']}
    write_json(root/'protocol.json',protocol)
    write_json(root/'preflight.json',{'validated':True,'tests':90,'asset_paths':len(required),'policy_dependencies':deps})
    (root/'README.md').write_text('Official DynamicVLA DOM manipulation evaluation\n\n'
        'Released hzxie/dynamic-vla-DOM checkpoint; no fine-tuning or retraining. '
        'The official 90 test configurations are unchanged: 20 trials each, 1800 total. '
        f'Requested GPU: {gpu_label}. Pilot: one trial from each of nine dimensions; full run starts only after validation.\n\n'
        'Results: full/00_CR through full/08_DR contain trial_results.jsonl, verification.json, logs and videos. '
        'Final table1_comparison.csv, table1_comparison.md, table1_comparison.png and summary.json appear here after all trials verify.\n\n'
        'Success is actual manipulation completion. Time includes failed trials, using simulation time. '
        'Reference: Table 1, arXiv 2601.22153v2: 47.06% overall success, 8.53 s mean time. '
        'Streaming remains enabled. See protocol.json for hardware, exact revisions and provenance.\n')
    print(json.dumps({'prepared':True,'source_commit':commit,'tests':90,'scenes':len(scenes),'root':str(root)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,type=Path);p.add_argument('--root',required=True,type=Path);p.add_argument('--assets',required=True,type=Path)
    a=p.parse_args();prepare(a.stage,a.root,a.assets)
