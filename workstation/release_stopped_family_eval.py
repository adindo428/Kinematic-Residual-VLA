"""Verify saved family snapshots after a confirmed user-requested training stop."""
import argparse,hashlib,json
from pathlib import Path
import torch


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--family',default='smolvla_baseline');a=p.parse_args();root=a.root.resolve();run=root/'runs'/a.family
    stop=read(root/'evaluation-training-stop.json')
    if not stop.get('user_requested') or not stop.get('training_process_exited'):raise ValueError('A confirmed user-requested stop is required')
    stats=read(root/'real_statistics.json');config=read(run/'run_config.json');best_pointer=read(run/'best_checkpoint.json');records={}
    for selection,path in [('best',Path(best_pointer['path']).resolve()),('last',run/'latest.pt')]:
        if path.parent!=run:raise ValueError('Checkpoint outside selected run')
        snapshot=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if snapshot['architecture_signature']!=config['architecture_signature'] or snapshot['statistics']!=stats:raise ValueError('Saved architecture or normalization mismatch')
        if selection=='best':
            if snapshot['epoch']!=best_pointer['epoch'] or snapshot['selected_weights']!=best_pointer['weights']:raise ValueError('Best checkpoint metadata mismatch')
        elif not snapshot.get('completed_epoch'):raise ValueError('Last snapshot is not a completed epoch')
        records[selection]=dict(path=str(path),epoch=snapshot['epoch'],weights=snapshot['selected_weights'],sha256=digest(path),step=snapshot['step'])
        del snapshot
    pointer=read(run/'latest_checkpoint.json')
    write(run/'latest_checkpoint.before-evaluation-release.json',pointer)
    write(run/'latest_checkpoint.json',dict(records['last'],completed_epoch=True))
    release=dict(**stop,snapshots_verified=True,checkpoints=records,final_before_after_evaluation_completed=False,
        note='User stopped training early. Saved checkpoints are verified for inference; automatic final before/after evaluation was not run.')
    write(run/'evaluation_release.json',release)
    previous=read(run/'status.json');write(run/'status.before-evaluation-release.json',previous)
    write(run/'status.json',dict(family=a.family,state='stopped',step=previous.get('step'),epoch=records['last']['epoch'],
        reason=stop['reason'],wandb_url=previous.get('wandb_url'),evaluation_ready=True,best_epoch=records['best']['epoch'],last_completed_epoch=records['last']['epoch']))
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
