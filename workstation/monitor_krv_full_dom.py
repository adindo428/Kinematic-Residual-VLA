"""Passive periodic closed-loop health snapshots; never changes evaluator behavior."""
import argparse,json,re,time,statistics
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'scratch/dom45/evaluations/krv-full-dom-ablation-20261115'
OUT=ROOT/'runtime_metrics'
LAT=re.compile(r'Inference Time: ([\d.]+)s; Sleep Time: ([\d.]+)s')

def distribution(v):
    v=sorted(v)
    return dict(samples=len(v),p50=v[len(v)//2] if v else None,p95=v[min(len(v)-1,int(.95*len(v)))] if v else None,max=max(v) if v else None)

def snapshot():
    plan=json.loads((ROOT/'plan.json').read_text());rows=[]
    for m in plan['models']:
        latency=[];sleeps=[];step=[];age=[];steps=fresh=superseded=warnings=trials=0
        for j in plan['jobs']:
            if j['model']!=m['key']:continue
            folder=Path(j['folder'])
            for f in [folder/'client.log',folder/'server.log']:
                if f.exists():
                    text=f.read_text(errors='replace');warnings+=text.count('output queue is full')
                    for a,b in LAT.findall(text):latency.append(float(a)*1000);sleeps.append(float(b)*1000)
            f=folder/'loop_health.jsonl'
            if f.exists():
                for line in f.read_text().splitlines():
                    try:r=json.loads(line)
                    except json.JSONDecodeError:continue
                    trials+=1;steps+=r['steps'];fresh+=r['fresh_actions'];superseded+=r['superseded_messages'];step+=r['step_ms'];age+=r['chunk_age_steps']
        rows.append(dict(model=m['key'],episodes=trials,inference_ms=distribution(latency),intentional_sleep_ms=distribution(sleeps),step_compute_ms=distribution(step),chunk_age_steps=distribution(age),steps=steps,fresh_actions=fresh,action_coverage=fresh/steps if steps else None,steps_reusing_previous_action=steps-fresh,superseded_messages=superseded,output_queue_full_warnings=warnings))
    report=dict(utc=datetime.now(timezone.utc).isoformat(),models=rows,notes=['Latest-only transport deliberately replaces observations/actions; superseded messages are not proof of network loss.','Chunk age is simulation steps since chunk observation, not end-to-end wall latency.','Inference logs exclude intentional alignment sleep; available only when native policy logs them.','Reports are passive; no automatic parameter or scoring changes.'])
    OUT.mkdir(exist_ok=True)
    (OUT/'latest.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'history.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
    lines=['# Closed-loop performance monitoring','',report['utc'],'','Updated every 120 seconds. Units: milliseconds unless specified.','','| Arm | Episodes | Inference p50 / p95 | Step compute p95 | Fresh action coverage | Superseded messages | Queue-full warnings |','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        fmt=lambda v:f'{v:.2f}' if v is not None else 'pending'
        lines.append(f"| {r['model']} | {r['episodes']} | {fmt(r['inference_ms']['p50'])} / {fmt(r['inference_ms']['p95'])} | {fmt(r['step_compute_ms']['p95'])} | {fmt(100*r['action_coverage'])+'%' if r['action_coverage'] is not None else 'pending'} | {r['superseded_messages']} | {r['output_queue_full_warnings']} |")
    lines+=['']+report['notes'];(OUT/'performance.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--once',action='store_true');args=a.parse_args()
    while True:
        snapshot()
        if args.once:break
        time.sleep(120)
