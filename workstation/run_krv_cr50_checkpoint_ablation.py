"""Finish the fixed first50CR trials, then evaluate full-split KRV on the same cases."""
import copy,inspect,json
from pathlib import Path
import run_krv_profile_ablation as base

ROOT=base.low.WORK/'evaluations/krv-cr50-checkpoint-ablation-20261114'
PARENT=base.low.WORK/'evaluations/krv-frozen-kra-vpd-vi-20261114'
base.ROOT=ROOT


def render():
    plan=base.low.read(ROOT/'plan.json');scores={m['key']:[0,0] for m in plan['models']}
    for job in plan['jobs']:
        path=Path(job['folder'])/'trial_results.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                scores[job['model']][0]+=bool(r['success']);scores[job['model']][1]+=1
    lines=['# KRV checkpoint ablation — low-clutter CR','','50 matched trials per row: the first5CR configurations in the frozen queue, repeats1–10. DOMV is archived separately. Each checkpoint retains its trained gate, velocity conditioning and normalization. VPD and VI are inference-only interventions.','','| Checkpoint | KRA | VPD | VI | Successes / completed | SR (%) |','|---|:---:|:---:|:---:|---:|---:|']
    for m in plan['models']:
        s,n=scores[m['key']];assert n<=50;rate=f'{100*s/n:.2f}' if n else 'pending';lines.append(f"| {m['label']} | ✓ | {'✓' if m['vpd'] else '✗'} | {'✓' if m['vi'] else '✗'} | {s}/{n} | {rate} |")
    lines+=['','Local20k: epoch3.943/update26959, trained5cm distance gate and two-velocity history. Full-split: validation-best epoch3/update196707, trained constant gate and current velocity. Cross-checkpoint differences include dataset size, budget, gate and conditioning; compare inference interventions within each checkpoint.','',f'Run details: [protocol]({ROOT}/protocol.json). DOMV historical result:22/50(44%); excluded from the main ablation.']
    text='\n'.join(lines)+'\n';(ROOT/'live_results.md').write_text(text);(PARENT/'live_results.md').write_text(text)
    return scores


def launch(full):
    pending=copy.deepcopy(full);pending['jobs']=[j for j in full['jobs'] if not (Path(j['folder'])/'trial_results.jsonl').exists()]
    keys={j['model'] for j in pending['jobs']};pending['models']=[m for m in full['models'] if m['key'] in keys]
    assert all(j['dimension']=='CR' for j in pending['jobs'])
    local_remaining=[j for j in pending['jobs'] if not j['model'].startswith('full_')]
    full_remaining=[j for j in pending['jobs'] if j['model'].startswith('full_')]
    pending['jobs']=local_remaining+full_remaining
    base.low.write(ROOT/'active-plan.json',pending)
    runner=base.runner;runner.ROOT=ROOT;runner.status=base.status
    runner.verify=lambda _:base.verify(base.low.read(ROOT/'plan.json'))
    text=inspect.getsource(runner.launch).replace("'closed-loop-pilot'","'krv-cr50-ablation'").replace('closed_loop_pilot_server.py','krv_profile_ablation_server.py').replace('closed_loop_pilot_client.py','krv_profile_ablation_client.py').replace('total=75',"total=len(plan['jobs'])").replace('completed=75',"completed=len(plan['jobs'])")
    text=text.replace("'--plan', str(ROOT/'plan.json')", "'--plan', str(ROOT/'active-plan.json')")
    exec(compile(text,str(Path(__file__).resolve()),'exec'),runner.__dict__)
    runner.launch(pending)

if __name__=='__main__':
    base.render=render;base.launch=launch;base.main()
