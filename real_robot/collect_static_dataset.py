"""Resumable static collection in whole-session train/validation/test blocks."""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .stream_upload import save,successes

PROJECT=Path(__file__).resolve().parents[1]
PREPROCESSING=PROJECT/'scratch/real_robot/training_releases/KRVLA-5fd5ef3e4256-extrapolated-80ms-grasp-v1/meta/image_preprocessing.json'


def blocks(total=100,size=None):
    size=total if size is None else size
    if total<10 or total%10 or size<1:raise ValueError('Use a total divisible by 10 (at least 10) and a positive block size')
    result=[]
    for split,count in [('train',total*8//10),('validation',total//10),('test',total//10)]:
        while count:
            n=min(size,count);result.append(dict(split=split,episodes=n,segments=[]));count-=n
    return result


def merge_split_blocks(state,size):
    """Combine session targets, preserving all existing segment/split records."""
    totals={split:sum(b['episodes'] for b in state['blocks'] if b['split']==split)
            for split in ['train','validation','test']}
    if max(totals.values())>size:raise ValueError('Use a block size large enough for each remaining split when migrating an existing campaign')
    old=copy.deepcopy(state['blocks'])
    state['blocks']=[dict(split=split,episodes=total,
        segments=[s for b in old if b['split']==split for s in b['segments']]) for split,total in totals.items() if total]
    state.setdefault('plan_history',[]).append(dict(reason='remove artificial recording block cap',blocks=old))
    state['config']['block_size']=size
    state['current_block']=None


def collector_command(config,block,count,session):
    return [sys.executable,'-u','-m','real_robot.collect','--task',config['task'],'--static-object',
        '--episodes',str(count),'--split',block['split'],'--output',str(session),
        '--viewer',config['viewer'],'--keyboard-source','terminal','--default-success',
        '--robot-max-relative-target','10','--image-preprocessing',config['image_preprocessing'],
        '--hub-repo',config['repo'],'--stream-upload','--upload-batch-episodes',str(config['upload_batch_episodes']),
        *(['--public'] if config['public'] else [])]


def reconcile_segment(segment,run=subprocess.run):
    session=Path(segment['session']);recording=session/'recording'
    summary=recording/'summary.json'
    if not summary.exists():
        if not (recording/'manifest.json').exists():
            segment.update(state='completed',episodes=0,reason='startup ended before recording')
            return 0
        raise ValueError('Session has no shutdown summary; inspect it before resuming: '+str(session))
    raw=json.loads(summary.read_text())
    if raw.get('error') or raw.get('cleanup_errors'):raise ValueError('Collection error needs review: '+str(session))
    state_path=session/'stream-upload/status.json'
    status=json.loads(state_path.read_text()) if state_path.exists() else {}
    expected={row['episode_id'] for row in successes(recording)}
    def accounted(s):
        uploaded=set(s.get('uploaded_episode_ids',[]));quarantined={int(i) for i in s.get('quarantined_episode_ids',{})}
        return not uploaded&quarantined and uploaded|quarantined==expected
    if status.get('state')!='completed' or not accounted(status):
        result=run([sys.executable,'-u','-m','real_robot.stream_upload','--session',str(session)],cwd=PROJECT)
        if result.returncode:raise ValueError('Pending processing/upload needs review: '+str(session))
        status=json.loads(state_path.read_text())
    if status.get('state')!='completed' or not accounted(status):
        raise ValueError('Saved and uploaded episode counts differ: '+str(session))
    count=len(status.get('uploaded_episode_ids',[]))
    segment.update(state='completed',episodes=count,saved_successes=len(expected),
                   quarantined_episode_ids=status.get('quarantined_episode_ids',{}),last_receipt=status.get('last_receipt'))
    return count


def run_plan(root,state,run=subprocess.run):
    """Resume counts from durable receipts; an early Q pauses the campaign."""
    path=root/'campaign.json';config=state['config']
    for index,block in enumerate(state['blocks']):
        for segment in block['segments']:
            if segment.get('state')!='completed':reconcile_segment(segment,run)
        count=sum(s.get('episodes',0) for s in block['segments'])
        if count>block['episodes']:raise ValueError('Block episode count exceeds its frozen target')
        if count==block['episodes']:continue
        wanted=block['episodes']-count
        identity=hashlib.sha256(str(root).encode()).hexdigest()[:10]
        session=PROJECT/'scratch/real_robot/collections'/f'static-{config["task"]}-{block["split"]}-auto-{identity}-{index:02d}-{len(block["segments"]):02d}'
        segment=dict(session=str(session),state='pending',requested=wanted)
        block['segments'].append(segment);state.update(state='collecting',current_block=index);save(path,state)
        print(f'Next: {config["task"]} {block["split"]}, {wanted} successes. New placements for held-out sessions. SPACE starts each attempt.',flush=True)
        result=run(collector_command(config,block,wanted,session),cwd=PROJECT)
        added=reconcile_segment(segment,run);save(path,state)
        if added!=wanted:
            excluded=len(segment.get('quarantined_episode_ids',{}))
            reason=(f'{excluded} saved episodes were excluded by alignment checks; rerun the same command to collect the remaining valid episodes'
                    if excluded else 'Session ended early; rerun the same command to collect the remainder')
            state.update(state='paused',reason=reason)
            save(path,state);print(state['reason'],flush=True);return 0 if not result.returncode else 1
    state.update(state='completed',current_block=None);save(path,state)
    totals={split:sum(s['episodes'] for b in state['blocks'] if b['split']==split for s in b['segments'])
            for split in ['train','validation','test']}
    print('Collection and uploads complete: '+json.dumps(totals),flush=True)
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True,choices=['striped_ball','marker','orange'])
    p.add_argument('--episodes',type=int,default=100);p.add_argument('--block-size',type=int,help='Optional recording cap; default collects each split in one session')
    p.add_argument('--root',type=Path)
    p.add_argument('--prepare-only',action='store_true',help='Save/review the split plan without recording or uploading')
    p.add_argument('--hub-repo',default='ailab-Clemson/KRVLA-dataset')
    p.add_argument('--private',action='store_true',help='For a different private destination; default is the existing public dataset')
    p.add_argument('--image-preprocessing',type=Path,default=PREPROCESSING)
    p.add_argument('--viewer',choices=['opencv','rerun'],default='rerun')
    p.add_argument('--upload-batch-episodes',type=int,default=5)
    args=p.parse_args(argv)
    if args.block_size is None:args.block_size=args.episodes
    try:plan=blocks(args.episodes,args.block_size)
    except ValueError as exc:p.error(str(exc))
    if args.upload_batch_episodes<1:p.error('Upload batch size must be positive')
    preprocessing=args.image_preprocessing.resolve()
    if not preprocessing.is_file():p.error('Saved image preprocessing is missing: '+str(preprocessing))
    root=(args.root or PROJECT/'scratch/real_robot/static-collection-campaigns'/f'{args.task}-{args.episodes}-v1').resolve()
    root.mkdir(parents=True,exist_ok=True)
    config=dict(task=args.task,total=args.episodes,block_size=args.block_size,repo=args.hub_repo,
        public=not args.private,image_preprocessing=str(preprocessing),viewer=args.viewer,
        upload_batch_episodes=args.upload_batch_episodes)
    with (root/'campaign.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'campaign.json'
        if path.exists():
            state=json.loads(path.read_text())
            if state.get('superseded_by'):
                p.error('This partial campaign was consolidated into '+state['superseded_by']+
                        '; resume the original --task '+args.task+' --episodes 100 campaign instead')
            if {k:v for k,v in state['config'].items() if k!='block_size'}!={k:v for k,v in config.items() if k!='block_size'}:
                p.error('Resume configuration differs from the saved campaign; use its original options or a new --root')
            if state['config']['block_size']!=config['block_size']:
                try:merge_split_blocks(state,config['block_size'])
                except ValueError as exc:p.error(str(exc))
                save(path,state)
        else:
            state=dict(schema='krvla_static_collection_campaign_v1',config=config,blocks=plan,state='prepared')
            save(path,state)
        if args.prepare_only:
            print(json.dumps(dict(plan=str(path),state=state['state'],
                blocks=[dict(split=b['split'],episodes=b['episodes']) for b in state['blocks']]),indent=2))
            return
        try:code=run_plan(root,state)
        except (Exception,KeyboardInterrupt) as exc:
            state.update(state='paused',reason=f'{type(exc).__name__}: {exc}');save(path,state)
            print('Campaign retained: '+state['reason']+' | '+str(path),flush=True);code=1
    raise SystemExit(code)


if __name__=='__main__':main()
