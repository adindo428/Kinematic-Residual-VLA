"""Four frozen checkpoint arms, original DOM physical configurations, 900 cases each."""
import copy, inspect, json
from pathlib import Path
import run_krv_profile_ablation as base
ROOT=base.low.WORK/'evaluations/krv-full-dom-ablation-20261115'
base.ROOT=ROOT

def render():
    p=base.low.read(ROOT/'plan.json'); scores={m['key']:[0,0] for m in p['models']}
    for j in p['jobs']:
        f=Path(j['folder'])/'trial_results.jsonl'
        if f.exists():
            for line in f.read_text().splitlines():
                r=json.loads(line);scores[j['model']][0]+=bool(r['success']);scores[j['model']][1]+=1
    lines=['# Full-scale DOM KRA / VPD / VI ablation','','900 matched cases per row: original 90 DOM configurations, repeats 1–10. Fresh trials; prior low-clutter scores excluded.','','| Model | KRA | VPD | VI | Success / completed | SR (%) |','|---|:---:|:---:|:---:|---:|---:|']
    for m in p['models']:
        s,n=scores[m['key']];rate=f'{100*s/n:.2f}' if n else 'pending'
        lines.append(f"| {m['label']} | {'✓' if m['kr'] else '✗'} | {'✓' if m['vpd'] else '✗'} | {'✓' if m['vi'] else '✗'} | {s}/{n} | {rate} |")
    lines+=['','DOM: DeltaGPU continuation latest update184997, complete-data stage epoch0, cursor13381632; no validation-best exists yet. KRV: full-split validation-best epoch3/update196707. Each retains trained inputs and normalization. DOM has no neural object velocity; comparison does not isolate KR training alone. VPD/VI are inference-only.','', 'Target manuscript: tables/velocity_innovation_ablation.tex. Manuscript values await verified completion.']
    (ROOT/'live_results.md').write_text('\n'.join(lines)+'\n');return scores

def prepare():
    if (ROOT/'plan.json').exists():return base.low.read(ROOT/'plan.json')
    old=base.low.read(base.low.WORK/'evaluations/krv-cr50-checkpoint-ablation-20261114/active-plan.json')
    models=[copy.deepcopy(m) for m in old['models'] if m['key'].startswith('full_')]
    dom=copy.deepcopy(models[0]);dom.update(key='dom',label='DOM DeltaGPU update184997',kr=False,vpd=False,vi=False,gated=False,checkpoint=str(ROOT/'checkpoints/dom'))
    # Compatible frozen KRV source supports the same raw-action DOM architecture.
    dom['weight_sha256']=base.low.sha(Path(dom['checkpoint'])/'model.safetensors');dom['config_sha256']=base.low.sha(Path(dom['checkpoint'])/'config.json')
    models.append(dom)
    protocol=base.low.read(base.low.WORK/'evaluations/krv-full-original-dom-20261106/protocol.json')
    jobs=[];prompts={}
    for t in protocol['tests']:
        assert base.low.sha(t['path'])==t['config_sha256']
        cfg=base.low.read(t['path'])
        for m in models:
            folder=ROOT/'blocks'/f"{t['name']}-{m['key']}";folder.mkdir(parents=True,exist_ok=True);(folder/'case.txt').write_text(t['path']+'\n')
            jobs.append(dict(id=folder.name,model=m['key'],scene=t['path'],scene_sha256=t['config_sha256'],repeats=10,repeat_offset=0,folder=str(folder),dimension=t['dimension'],scored=True,seed=t['seed'],vi=m['vi']))
    p=dict(models=models,jobs=jobs,tests=protocol['tests'],saved_prompts=prompts,files_sha256={},protocol=dict(trials_per_arm=900,total_trials=3600,inference_only=True,matched_training_claim=False,original_scene_physics_prompts_and_execution_unchanged=True),source_pointer=base.low.read(ROOT/'checkpoints/dom/source-pointer.json'))
    files=list((base.low.REPO/'workstation').glob('krv_profile*.py'))+[Path(__file__)]+list(Path(models[0]['source']).rglob('*.py'))
    p['files_sha256']={str(f.resolve()):base.low.sha(f) for f in files}
    assert len(jobs)==360 and len(protocol['tests'])==90
    base.low.write(ROOT/'plan.json',p);base.low.write(ROOT/'protocol.json',p['protocol']);render();return p

def launch(plan):
    pending=copy.deepcopy(plan)
    pending['jobs']=[j for j in plan['jobs'] if not (Path(j['folder'])/'resident_client_completion.json').exists()]
    base.low.write(ROOT/'active-plan.json',pending)
    runner=base.runner;runner.ROOT=ROOT;runner.status=base.status;runner.verify=lambda _:base.verify(plan)
    code=inspect.getsource(runner.launch).replace("'closed-loop-pilot'","'krv-full-dom-ablation'").replace('closed_loop_pilot_server.py','krv_full_dom_server.py').replace('closed_loop_pilot_client.py','krv_profile_ablation_client.py').replace('total=75',"total=len(plan['jobs'])").replace('completed=75',"completed=len(plan['jobs'])")
    code=code.replace("'--plan', str(ROOT/'plan.json')","'--plan', str(ROOT/'active-plan.json')")
    exec(compile(code,__file__,'exec'),runner.__dict__);runner.launch(pending)

if __name__=='__main__':
    base.prepare=prepare;base.render=render;base.launch=launch;base.main()
