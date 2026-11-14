"""Read allocated-node results and publish explicit pending/provisional W&B rows."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import palmetto_allocated_transport as transport

REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO/'scratch/dom45/evaluations/pi05-kr-l40-20261111'
LOW = LOCAL.parent/'low-clutter-full-table1-20261102'
REMOTE = '/scratch/aparame/Dynamic_VLA/pi05-kr-l40-lowclutter-20261111'
KEY = 'pi05_kr_epoch3_partial4_l40'
DIMS = ['CR', 'DA', 'LS', 'VU', 'SR', 'MP', 'VG', 'MG', 'DR']


def read(p): return json.loads(Path(p).read_text())
def write(p, d):
    p = Path(p); tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(d, indent=2)+'\n'); tmp.replace(p)


def remote_snapshot():
    script = f'''import json
from pathlib import Path
r=Path({REMOTE!r}); c=r/'scratch/dom45/evaluations/low-clutter-full-table1-20261102/l40_suite'
out={{}}
for key,path in {{'job':r/'job.json','runtime':r/'gpu_runtime.json','status':c/'execution/status.json',
 'dummy':c/'execution/l40_dummy_qualification.json','streaming':c/'execution/l40_streaming_qualification.json',
 'qualified':c/'execution/l40_scoring_authorized.json','completion':c/'completion.json','failure':c/'failure.json',
 'live':c/'live_results.json'}}.items():
 if path.exists():
  try:out[key]=json.loads(path.read_text())
  except (OSError,ValueError):pass
if 'failure' not in out and (r/'deployment_failure.json').exists():
 out['failure']=json.loads((r/'deployment_failure.json').read_text())
print(json.dumps(out))
'''
    return json.loads(transport.run(['python3', '-'], input_text=script, timeout=90))


def table(snapshot):
    lines = [s for s in (LOW/'table1_comparison.md').read_text().splitlines() if s.startswith('|')]
    columns = [s.strip() for s in lines[0].strip('|').split('|')]
    values = [[s.strip() for s in row.strip('|').split('|')] for row in lines[2:]]
    records = snapshot.get('live', {}).get('records', {}).get(KEY, [])
    qualified = snapshot.get('qualified', {}).get('passed', False)
    if records: assert qualified
    cells = []
    for dimension in DIMS:
        group = [r for r in records if r['dimension'] == dimension]
        n = len(group)
        cells.append(f'{100*sum(r["success"] for r in group)/n:.2f}'+(f' (n={n})' if n != 200 else '') if n else 'pending')
    state = snapshot.get('status', {}).get('stage', 'awaiting L40 allocation / Sim 4.5 preflight')
    if snapshot.get('failure'): state = 'Held: runtime or latency qualification failed'
    if snapshot.get('completion', {}).get('complete'): state = 'Complete'
    overall = f'{100*sum(r["success"] for r in records)/1800:.2f}' if len(records) == 1800 else 'pending'
    duration = f'{sum(r["task_time_s"] for r in records)/1800:.2f}' if len(records) == 1800 else 'pending'
    values.append(['pi0.5 + KR (3.208 epochs; Palmetto L40)', state, *cells, overall, duration])
    return columns, values, records


def main():
    import wandb
    run_record = read(LOW/'wandb_l40_followup_table.json')
    settings = wandb.Settings(disable_git=True, disable_code=True, save_code=False,
        x_disable_meta=True, x_disable_stats=True, console='off')
    run = wandb.init(entity='aparame-clemson-university', project='Kinematic-Residual-VLA',
        id=run_record['id'], resume='allow', name='Table1-low-clutter-updated-L40-followup-20261111',
        job_type='evaluation', settings=settings)
    previous = None
    try:
        while True:
            try:
                snapshot = remote_snapshot()
                if (LOCAL/'staging_failure.json').exists():
                    snapshot['failure'] = read(LOCAL/'staging_failure.json')
                write(LOCAL/'remote_status.json', snapshot)
                columns, values, records = table(snapshot)
                signature = hashlib.sha256(json.dumps(values).encode()).hexdigest()
                metrics = {'L40/scored_trials': len(records),
                    'L40/qualified': int(snapshot.get('qualified', {}).get('passed', False))}
                if signature != previous:
                    metrics['Table1/results'] = wandb.Table(columns=columns, data=values); previous = signature
                for gate in ['dummy', 'streaming']:
                    if gate in snapshot: run.summary['L40/'+gate+'_qualification'] = snapshot[gate]
                if 'runtime' in snapshot: run.summary['L40/runtime'] = snapshot['runtime']
                if 'job' in snapshot: run.summary['L40/slurm_job'] = snapshot['job']
                if snapshot.get('qualified', {}).get('passed'):
                    pending = LOW/'pi05-kr-epoch003-partial-next-20261111/queue_status.json'
                    if pending.exists() and read(pending).get('state') == 'waiting_for_vlash_comparison':
                        subprocess.run(['systemctl','--user','stop','krvla-pi05-kr-epoch3-queued-20261111.service'], check=True, timeout=30)
                        write(pending, {'state': 'delegated_to_qualified_l40', 'wandb': run.url})
                run.log(metrics)
                if snapshot.get('completion', {}).get('complete'):
                    assert len(records) == 1800
                    assert len({(r['env_name'],r['trial']) for r in records}) == 1800
                    assert Counter(r['env_name'] for r in records) == Counter({t['name']:20 for t in read(LOW/'protocol.json')['tests']})
                    write(LOCAL/'completed_records.json', records)
                    run.summary['L40/success_percent'] = 100*sum(r['success'] for r in records)/1800
                    write(LOCAL/'completion.json', {'complete': True, 'trials':1800, 'wandb':run.url})
                    break
                if snapshot.get('failure'):
                    run.summary['L40/held_reason'] = snapshot['failure'].get('error', 'Qualification failed')
                    write(LOCAL/'held.json', {'reason': snapshot['failure'], 'local_fallback_retained': True,
                                             'wandb': run.url})
                    break
                write(LOCAL/'publisher_status.json', {'state':'publishing', 'wandb':run.url,
                    'updated_utc':datetime.now(timezone.utc).isoformat(), 'remote_scored_trials':len(records)})
            except Exception as error:
                write(LOCAL/'publisher_error.json', {'error':str(error), 'updated_utc':datetime.now(timezone.utc).isoformat()})
            time.sleep(60)
    finally: run.finish()


if __name__ == '__main__': main()
