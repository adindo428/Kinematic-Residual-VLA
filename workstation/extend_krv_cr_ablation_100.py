"""After verified first50 finish, extend allsix KRV settings to100 matched CR trials."""
import copy,fcntl,json,os,signal,time
import psutil
from pathlib import Path
import run_krv_cr50_checkpoint_ablation as study

ROOT=study.ROOT
read=study.base.low.read
write=study.base.low.write
STOP=False

def stop(*_):
    global STOP
    STOP=True;study.base.runner.STOP.set()

def render():
    plan=read(ROOT/'plan.json');scores={m['key']:[0,0] for m in plan['models']};target=plan['protocol']['trials_per_arm']
    for job in plan['jobs']:
        p=Path(job['folder'])/'trial_results.jsonl'
        if p.exists():
            for line in p.read_text().splitlines():
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                scores[job['model']][0]+=bool(r['success']);scores[job['model']][1]+=1
    lines=['# KRV checkpoint ablation — low-clutter CR','',f'{target} matched trials per row: same five fixed CR configurations, repeats1–20. DOMV is archived separately. Checkpoint-specific trained inputs, normalization and gates are retained. VPD/VI are inference-only interventions.','','| Checkpoint | KRA | VPD | VI | Successes / completed | SR (%) |','|---|:---:|:---:|:---:|---:|---:|']
    for m in plan['models']:
        s,n=scores[m['key']];assert n<=target;rate=f'{100*s/n:.2f}' if n else 'pending'
        lines.append(f"| {m['label']} | ✓ | {'✓' if m['vpd'] else '✗'} | {'✓' if m['vi'] else '✗'} | {s}/{n} | {rate} |")
    lines+=['','A: original20k checkpoint, epoch3.943/update26959, trained5cm distance gate and two-velocity history. B: full203151episode checkpoint, best epoch3/update196707, trained constant gate and current velocity. Compare inference changes within each checkpoint; A/B differ in more than dataset size.','',f'[Checkpoint guide]({ROOT}/README.md) · [Earlier full-data VI comparison]({ROOT}/previous-full-data-velocity-ablation.md)']
    text='\n'.join(lines)+'\n';(ROOT/'live_results.md').write_text(text);(study.PARENT/'live_results.md').write_text(text)
    return scores

def main():
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (ROOT/'extend100.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous_pid=int((ROOT/'runner.pid').read_text())
        while not STOP:
            completion=ROOT/'completion.json'
            if completion.exists() and read(completion).get('verified'):break
            if not psutil.pid_exists(previous_pid):raise RuntimeError('The first50 coordinator exited before verified completion; inspect its logs before continuing')
            write(ROOT/'extension-status.json',{'stage':'waiting_for_first50_completion','target_trials_per_arm':100,'additional_trials':300,'current_run_uninterrupted':True})
            time.sleep(5)
        if STOP:return
        with (study.base.low.ROOT/'queue.lock').open('a') as gpu,(ROOT/'runner.lock').open('a') as parent:
            # Acquire this study's ownership before the shared GPU lock, as the main runner does.
            fcntl.flock(parent,fcntl.LOCK_EX)
            while not STOP:
                try:fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(5)
            if STOP:return
            assert read(ROOT/'completion.json')['verified']
            old=read(ROOT/'plan.json');assert old['protocol']['trials_per_arm']==50
            plan=read(ROOT/'plan-100-pending.json')
            assert plan['models']==old['models'] and len(plan['tests'])==5
            for name in ('plan.json','protocol.json','completion.json','status.json','active-plan.json'):
                path=ROOT/name
                if path.exists():path.rename(ROOT/(name.removesuffix('.json')+'-first50.json'))
            (ROOT/'execution').rename(ROOT/'execution-first50')
            write(ROOT/'plan.json',plan);write(ROOT/'protocol.json',plan['protocol'])
            study.base.render=render;study.render=render
            write(ROOT/'runner.pid',os.getpid())
            write(ROOT/'extension-status.json',{'stage':'running_additional50','target_trials_per_arm':100,'additional_trials':300})
            study.launch(plan)
            write(ROOT/'extension-status.json',{'stage':'complete','trials_per_arm':100,'total_trials':600})

if __name__=='__main__':
    try:main()
    except Exception as error:
        write(ROOT/'extension-status.json',{'stage':'failed','error':str(error)});raise
