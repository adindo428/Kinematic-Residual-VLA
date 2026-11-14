"""Wait for the training queue, then repeat the paired ten-episode comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    args=parser.parse_args();project=Path(__file__).resolve().parents[1]
    output=args.runs/'final-validation-10'
    status=args.runs/'final-evaluation-status.json'
    def update(**value):status.write_text(json.dumps(value,indent=2)+'\n')
    update(state='waiting_for_training',output=str(output))
    for _ in range(48*60):
        queue=args.runs/'queue.json'
        record=json.loads(queue.read_text()) if queue.exists() else {}
        if record.get('state')=='failed':
            update(state='training_failed',queue=record);return
        if record.get('state')=='completed':break
        time.sleep(60)
    else:
        update(state='wait_timeout');return
    update(state='evaluating',output=str(output))
    command=[str(project/'workstation/finetune-python.sh'),str(project/'workstation/evaluate_real_action_error.py'),
             '--runs',str(args.runs),'--output',str(output),'--episodes','10','--seed','12345']
    result=subprocess.run(command,cwd=project)
    if result.returncode:
        update(state='evaluation_failed',returncode=result.returncode);return
    old=json.loads((project/'scratch/real_robot/action-evaluation/validation-10-seed12345/summary.json').read_text())
    new=json.loads((output/'summary.json').read_text())
    assert old['episodes']==new['episodes'] and old['dataset']==new['dataset']
    comparison={}
    for family in ['dom','domv','krv100']:
        comparison[family]={}
        for scope in ['first_action','full_horizon']:
            a,b=old['models'][family][scope],new['models'][family][scope]
            assert a['valid_targets']==b['valid_targets']
            comparison[family][scope]={key:dict(before=a[key],after=b[key],change_percent=100*(b[key]/a[key]-1))
                for key in a if key!='valid_targets' and a[key]!=0}
    (output/'before_after.json').write_text(json.dumps(comparison,indent=2)+'\n')
    update(state='completed',comparison=str(output/'before_after.json'))


if __name__=='__main__':main()
