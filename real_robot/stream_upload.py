"""Process/upload immutable completed-episode snapshots beside live collection.

No cameras, serial ports, robot control, or partial episode uploads. Each batch
and upload receipt is durable; rerunning the worker retries pending work.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import uuid


class SnapshotNotReady(ValueError):
    pass


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def complete_rows(path):
    """Only newline-terminated records are committed by the live log writer."""
    data=Path(path).read_bytes()
    return [json.loads(line) for line in data.splitlines(keepends=True) if line.endswith(b'\n') and line.strip()]


def successes(recording):
    result=[]
    for path in sorted((recording/'episodes').glob('episode_*.json')):
        row=json.loads(path.read_text())
        if row.get('outcome')=='success' and row.get('end_monotonic_s') is not None:
            result.append(row)
    return result


def snapshot(recording,episodes,destination,*,finished=False):
    """Freeze selected attempts and their closed raw archives using hard links."""
    recording=Path(recording).resolve();destination=Path(destination)
    if destination.exists():
        saved=json.loads((destination/'snapshot.json').read_text())
        if saved['episode_ids']!=[row['episode_id'] for row in episodes]:raise ValueError('Snapshot episode selection changed')
        return destination
    begin=min(e['start_monotonic_s'] for e in episodes)
    end=max(e['end_monotonic_s'] for e in episodes)
    if not finished and time.monotonic()<end+1:
        raise SnapshotNotReady('Waiting for completed-episode archive flush')
    streams={}
    fields={'robot.jsonl':'host_read_end_monotonic_s','observations.jsonl':'host_recorded_monotonic_s',
            'wrist/frames.jsonl':'available_monotonic_s'}
    try:
        for name,key in fields.items():
            rows=complete_rows(recording/name)
            if not rows or (not finished and max(row[key] for row in rows)<end-.1):
                raise SnapshotNotReady('Waiting for final raw archives: '+name)
            if name=='robot.jsonl' and not finished and max(row[key] for row in rows)<end:
                raise SnapshotNotReady('Waiting for robot log to pass episode end')
            streams[name]=[row for row in rows if begin-.3<=row[key]<=end]
    except FileNotFoundError as exc:
        raise SnapshotNotReady('Waiting for raw logs') from exc
    files=set()
    for row in streams['observations.jsonl']:
        files.update((row.get('rgbd') or {}).get('files',{}).values())
    files.update(row['file'] for row in streams['wrist/frames.jsonl'])
    for name in files:
        path=(recording/name).resolve()
        if Path(name).is_absolute() or '..' in Path(name).parts or not path.is_relative_to(recording) or not path.is_file():
            raise SnapshotNotReady('Waiting for raw archive '+name)
    stage=destination.with_name(destination.name+'.pending-'+uuid.uuid4().hex)
    stage.mkdir(parents=True)
    try:
        manifest=json.loads((recording/'manifest.json').read_text())
        manifest['stream_snapshot_original_session']=str(recording)
        save(stage/'manifest.json',manifest)
        for name,rows in streams.items():
            path=stage/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows))
        for row in episodes:save(stage/'episodes'/f'episode_{row["episode_id"]:06d}.json',row)
        for name in files:
            target=stage/name;target.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(recording/name,target)
            except OSError:shutil.copy2(recording/name,target)
        save(stage/'snapshot.json',dict(original_session=str(recording),episode_ids=[e['episode_id'] for e in episodes]))
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return destination


def process_job(root,job,config,*,finished=False):
    """Reuse an encoded batch after a network failure; append retries deduplicate."""
    from .dom_dataset import export_sessions,validate_dataset
    from .dataset_quality import process as audit
    from .hub_upload import upload_dataset
    work=root/'jobs'/job['id'];work.mkdir(parents=True,exist_ok=True)
    source=snapshot(root.parent/'recording',job['episodes'],work/'snapshot',finished=finished)
    export=Path(job['export']) if job.get('export') else None
    if export is None:
        # A prior export may already contain valid episodes even when some
        # attempts were rejected. Retrying should not encode the valid ones again.
        completed=sorted(work.glob('attempt-*/dom/export_report.json'))
        completed=[p for p in completed if json.loads(p.read_text()).get('complete')]
        if completed:export=completed[-1].parent
        else:
            index=1
            while (work/f'attempt-{index:03d}').exists():index+=1
            attempt=work/f'attempt-{index:03d}';attempt.mkdir()
            export=attempt/'dom'
            try:export_sessions([source],export,image_preprocessing=config.get('image_preprocessing'))
            except ValueError:
                failure=export.with_name(export.name+'.partial')/'failure.json'
                report=json.loads(failure.read_text()) if failure.exists() else {}
                attempts=report.get('attempts',[])
                if len(attempts)!=len(job['episodes']) or any(a.get('exported') for a in attempts):raise
                quarantine={str(int(Path(a['source']).stem.split('_')[-1])):dict(reason=a['reason'],source=a['source']) for a in attempts}
                if set(quarantine)!={str(e['episode_id']) for e in job['episodes']}:raise
                job.update(stage='quarantined',quarantined_episode_ids=quarantine,exported_episode_ids=[])
                save(work/'job.json',job);return None
        report=json.loads((export/'export_report.json').read_text())
        accepted=[int(Path(a['source']).stem.split('_')[-1]) for a in report['attempts'] if a.get('exported')]
        quarantine={str(int(Path(a['source']).stem.split('_')[-1])):dict(reason=a['reason'],source=a['source'])
                    for a in report['attempts'] if not a.get('exported')}
        requested={e['episode_id'] for e in job['episodes']}
        if (set(accepted)&{int(k) for k in quarantine} or set(accepted)|{int(k) for k in quarantine}!=requested
                or len(accepted)!=report['episodes']):raise ValueError('Exported/rejected episode accounting differs from selected batch')
        job.update(exported_episode_ids=accepted,quarantined_episode_ids=quarantine)
        job['export']=str(export.resolve());save(work/'job.json',job)
    validate_dataset(export,decode_videos=False)
    if not (export.parent/'quality-audit/quality_audit.json').exists():
        audit_path=export.parent/'quality-audit'
        if audit_path.exists():audit_path=export.parent/('quality-audit-'+uuid.uuid4().hex[:8])
        audit(export,audit_path,repair=True)
    job['stage']='uploading';save(work/'job.json',job)
    receipt=upload_dataset(export,config['repo'],config['public'],append=True)
    job.update(stage='uploaded',receipt=receipt);save(work/'job.json',job)
    return receipt


def run(session,*,once=False):
    session=Path(session).resolve();root=session/'stream-upload'
    config=json.loads((root/'config.json').read_text())
    with (root/'worker.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(state='watching',uploaded_episode_ids=[],quarantined_episode_ids={},jobs=[])
        if (root/'status.json').exists():state.update(json.loads((root/'status.json').read_text()))
        state.update(state='watching',error=None,pid=os.getpid());save(root/'status.json',state)
        while True:
            recording=session/'recording';finished=(recording/'summary.json').exists() or (root/'FINISH').exists()
            available=successes(recording)
            pending=[row for row in available if row['episode_id'] not in state['uploaded_episode_ids']
                     and str(row['episode_id']) not in state['quarantined_episode_ids']]
            prior=root/'jobs'/str(state.get('current_job'))/'job.json'
            previous=json.loads(prior.read_text()) if prior.is_file() else None
            retry=previous is not None and previous.get('stage') not in ('uploaded','quarantined')
            if pending and (retry or len(pending)>=config['batch_episodes'] or finished or once):
                chosen=previous['episodes'] if retry else pending[:config['batch_episodes']]
                job_id='batch-'+'-'.join(f'{e["episode_id"]:06d}' for e in chosen)
                path=root/'jobs'/job_id/'job.json'
                job=json.loads(path.read_text()) if path.exists() else dict(id=job_id,episodes=chosen,stage='pending')
                save(path,job)
                state.update(state='processing',current_job=job_id);save(root/'status.json',state)
                try:receipt=process_job(root,job,config,finished=finished)
                except SnapshotNotReady as exc:
                    if finished:
                        state.update(state='failed',error=str(exc));save(root/'status.json',state);return 1
                    state.update(state='waiting_for_archives',error=str(exc));save(root/'status.json',state)
                    time.sleep(1);continue
                except Exception as exc:
                    state.update(state='failed',error=f'{type(exc).__name__}: {exc}')
                    save(root/'status.json',state)
                    print('Background upload paused; raw collection is retained. Retry: '+
                          f'{__import__("sys").executable} -m real_robot.stream_upload --session {session}',flush=True)
                    return 1
                accepted=job.get('exported_episode_ids',[e['episode_id'] for e in chosen])
                state['uploaded_episode_ids']=sorted(set(state['uploaded_episode_ids'])|set(accepted))
                state['quarantined_episode_ids'].update(job.get('quarantined_episode_ids',{}))
                if job_id not in state['jobs']:state['jobs'].append(job_id)
                state.update(state='watching',error=None)
                if receipt is not None:state['last_receipt']=receipt
                save(root/'status.json',state)
                print(f'Background upload: {len(state["uploaded_episode_ids"])} uploaded, {len(state["quarantined_episode_ids"])} set aside for alignment review | {config["repo"]}',flush=True)
                continue
            if finished or once:
                if finished and config.get('sync_training_on_finish') and state.get('last_receipt'):
                    receipt=state['last_receipt']
                    if state.get('training_commit_synced')!=receipt['commit']:
                        try:
                            from .training_release import sync
                            sync(config['repo'],receipt['dataset_path'],receipt['commit'])
                        except Exception as exc:
                            state.update(state='failed',error=f'Training branch refresh: {exc}')
                            save(root/'status.json',state);return 1
                        state['training_commit_synced']=receipt['commit']
                state.update(state='completed' if finished else 'watching',current_job=None)
                save(root/'status.json',state);return 0
            time.sleep(1)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True);p.add_argument('--once',action='store_true')
    p.add_argument('--repo',help='Initialize a worker beside an already running local collection')
    p.add_argument('--public',action='store_true');p.add_argument('--batch-episodes',type=int,default=5)
    p.add_argument('--image-preprocessing',type=Path)
    args=p.parse_args(argv)
    config_path=args.session.resolve()/'stream-upload/config.json'
    if not config_path.exists():
        if not args.repo:p.error('No worker configuration; supply --repo to attach to an existing collection')
        if args.batch_episodes<1:p.error('Batch size must be positive')
        if not (args.session/'recording/manifest.json').is_file():p.error('No collection recording manifest at --session')
        from .hub_upload import authenticated_repo,verify_upload_access,find_published_base
        api,repo=authenticated_repo(args.repo);info=verify_upload_access(api,repo,args.public)
        base=find_published_base(api,repo,info)
        preprocessing=args.image_preprocessing or (base/'meta/image_preprocessing.json' if base else None)
        if preprocessing is None or not preprocessing.is_file():p.error('Supply the existing dataset --image-preprocessing file')
        save(config_path,dict(repo=repo,public=args.public,batch_episodes=args.batch_episodes,
             image_preprocessing=str(preprocessing.resolve()),sync_training_on_finish=repo=='ailab-Clemson/KRVLA-dataset'))
    elif args.repo and json.loads(config_path.read_text())['repo']!=args.repo:
        p.error('Destination differs from the saved worker configuration')
    os.nice(10)
    raise SystemExit(run(args.session,once=args.once))


if __name__=='__main__':main()
