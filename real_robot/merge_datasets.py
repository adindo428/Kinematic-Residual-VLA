"""Merge validated DOM exports without re-encoding their video streams."""
import argparse
import copy
import math
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

from .dom_dataset import (CAMERAS,DATA_PATH,VIDEO_PATH,json_write,jsonl_write,jsonl_read,
                          numeric_stats,validate_dataset,write_parquet,sha256)


def same_source_episode(first,first_ep,second,second_ep):
    """A repeated source may be skipped only if its contents are unchanged."""
    def without_index(row):return {k:v for k,v in row.items() if k!='episode_index'}
    for key in ('sources','episodes','cameras'):
        if without_index(first[key][first_ep])!=without_index(second[key][second_ep]):
            raise ValueError('Repeated source episode has changed metadata')
    def rows(item,ep):
        values=pq.read_table(item['root']/DATA_PATH.format(episode_chunk=ep//1000,episode_index=ep)).to_pylist()
        return [dict({k:v for k,v in row.items() if k not in ('index','episode_index','task_index')},
                     task=item['tasks'][row['task_index']]['task']) for row in values]
    if rows(first,first_ep)!=rows(second,second_ep):raise ValueError('Repeated source episode has changed numeric data')
    paths=[lambda ep:f'quality/episode_{ep:06d}.jsonl']
    for camera in CAMERAS:
        paths.append(lambda ep,camera=camera:VIDEO_PATH.format(episode_chunk=ep//1000,episode_index=ep,video_key='observation.images.'+camera))
    for path in paths:
        if sha256(first['root']/path(first_ep))!=sha256(second['root']/path(second_ep)):
            raise ValueError('Repeated source episode has changed video or quality data')


def merge_datasets(datasets, output, *, append=False,link_immutable=False):
    datasets=[Path(p).resolve() for p in datasets]
    output=Path(output).resolve()
    if output.exists():raise ValueError('Merged output already exists; use a new version')
    inputs=[];seen={};base_info=None;base_preprocessing=None;entries=[]
    for number,root in enumerate(datasets):
        validate_dataset(root)
        import json
        info=json.loads((root/'meta/info.json').read_text())
        prep=json.loads((root/'meta/image_preprocessing.json').read_text()) if (root/'meta/image_preprocessing.json').exists() else None
        # Selection provenance can differ; pixel transforms must be identical.
        from .image_preprocessing import geometry
        geometry_key=None if prep is None else [geometry(prep,role) for role in ('front','side','wrist')]
        if base_info is None:
            base_info=info;base_preprocessing=prep;base_geometry=geometry_key
        elif any(info[k]!=base_info[k] for k in ('features','fps','robot_type')) or geometry_key!=base_geometry:
            raise ValueError('Dataset schemas or image transforms differ')
        item=dict(root=root,info=info,episodes=jsonl_read(root/'meta/episodes.jsonl'),
                  tasks=jsonl_read(root/'meta/tasks.jsonl'),stats=jsonl_read(root/'meta/episodes_stats.jsonl'),
                  sources=jsonl_read(root/'meta/sources.jsonl'),cameras=jsonl_read(root/'meta/camera.jsonl'),
                  report=json.loads((root/'export_report.json').read_text()),
                  hashes=json.loads((root/'meta/raw_hashes.json').read_text()))
        inputs.append(item)
        for source in item['sources']:
            identity=source['source_episode']
            if identity in seen:
                if not append:raise ValueError('Repeated source episode; refusing duplicate data: '+identity)
                previous,old=seen[identity]
                same_source_episode(inputs[previous],old,item,source['episode_index'])
                continue
            seen[identity]=(number,source['episode_index'])
            entries.append((source['split'],number,source['episode_index']))
    if not entries:raise ValueError('No episodes to merge')
    order={'train':0,'validation':1,'test':2}
    # Published IDs must remain a prefix, even when the new block uses a split
    # that would otherwise sort ahead of an existing validation/test episode.
    entries.sort(key=(lambda e:(e[1],e[2])) if append else (lambda e:(order[e[0]],e[1],e[2])))
    staging=output.with_name(output.name+'.partial')
    staging.mkdir(parents=True,exist_ok=False)
    metadata=[];statistics=[];sources=[];cameras=[];tasks=[];splits={k:[] for k in order};mapping={};total=0
    if append:tasks=[t['task'] for t in inputs[0]['tasks']]
    def retain(source,destination):
        if link_immutable:
            try:os.link(source,destination);return
            except OSError:pass
        shutil.copy2(source,destination)
    try:
        for split,number,old in entries:
            item=inputs[number];root=item['root'];ep=len(metadata);mapping[number,old]=ep
            rows=pq.read_table(root/DATA_PATH.format(episode_chunk=old//1000,episode_index=old)).to_pylist()
            for row in rows:
                task=item['tasks'][row['task_index']]['task']
                if task not in tasks:tasks.append(task)
                row.update(episode_index=ep,index=total+row['frame_index'],task_index=tasks.index(task))
            destination=staging/DATA_PATH.format(episode_chunk=ep//1000,episode_index=ep)
            if append and number==0:
                destination.parent.mkdir(parents=True,exist_ok=True)
                retain(root/DATA_PATH.format(episode_chunk=old//1000,episode_index=old),destination)
            else:write_parquet(rows,destination)
            stat=copy.deepcopy(item['stats'][old]);stat['episode_index']=ep
            if not (append and number==0):
                for key in rows[0]:stat['stats'][key]=numeric_stats([r[key] for r in rows])
            for camera in CAMERAS:
                key='observation.images.'+camera
                oldpath=root/VIDEO_PATH.format(episode_chunk=old//1000,episode_index=old,video_key=key)
                newpath=staging/VIDEO_PATH.format(episode_chunk=ep//1000,episode_index=ep,video_key=key)
                newpath.parent.mkdir(parents=True,exist_ok=True);retain(oldpath,newpath)
            quality=staging/f'quality/episode_{ep:06d}.jsonl';quality.parent.mkdir(exist_ok=True)
            retain(root/f'quality/episode_{old:06d}.jsonl',quality)
            episode=copy.deepcopy(item['episodes'][old]);episode['episode_index']=ep
            source=copy.deepcopy(item['sources'][old]);source['episode_index']=ep
            camera=copy.deepcopy(item['cameras'][old]);camera['episode_index']=ep
            metadata.append(episode);statistics.append(stat);sources.append(source);cameras.append(camera)
            splits[split].append(ep);total+=len(rows)
        info=copy.deepcopy(base_info)
        info.update(total_episodes=len(metadata),total_frames=total,total_tasks=len(tasks),
                    total_videos=len(metadata)*len(CAMERAS),total_chunks=math.ceil(len(metadata)/1000),
                    splits={name:f'{ids[0]}:{ids[-1]+1}' for name,ids in splits.items() if ids})
        json_write(staging/'meta/info.json',info)
        jsonl_write(staging/'meta/episodes.jsonl',metadata)
        jsonl_write(staging/'meta/episodes_stats.jsonl',statistics)
        jsonl_write(staging/'meta/sources.jsonl',sources)
        jsonl_write(staging/'meta/camera.jsonl',cameras)
        jsonl_write(staging/'meta/tasks.jsonl',[dict(task_index=i,task=t) for i,t in enumerate(tasks)])
        json_write(staging/'meta/splits.json',splits)
        if base_preprocessing is not None:
            prep=copy.deepcopy(base_preprocessing)
            prep['source_sessions']=sorted({s['raw_session'] for s in sources})
            json_write(staging/'meta/image_preprocessing.json',prep)
        from lerobot.common.datasets.compute_stats import aggregate_stats
        train=[{k:{n:np.asarray(v) for n,v in values.items()} for k,values in statistics[ep]['stats'].items()} for ep in splits['train']]
        if train:
            agg=aggregate_stats(train)
            json_write(staging/'meta/train_stats.json',{k:{n:v.tolist() for n,v in values.items()} for k,values in agg.items()})
        hashes={};audit=[];audited=set()
        for number,item in enumerate(inputs):
            for path,value in item['hashes'].items():
                if path in hashes and hashes[path]!=value:raise ValueError('Raw source hash conflict')
                hashes[path]=value
            for attempt in item['report']['attempts']:
                attempt=copy.deepcopy(attempt)
                if append and attempt['source'] in audited:continue
                audited.add(attempt['source'])
                if attempt.get('exported'):attempt['episode_index']=mapping[number,attempt['episode_index']]
                audit.append(attempt)
        json_write(staging/'meta/raw_hashes.json',hashes)
        report=copy.deepcopy(inputs[0]['report'])
        report.update(complete=True,episodes=len(metadata),frames=total,attempts=audit,
                      merged_exports=[str(p) for p in datasets],videos_reencoded=False,append=append)
        json_write(staging/'export_report.json',report)
        validate_dataset(staging)
        staging.rename(output)
        return report
    except BaseException as exc:
        json_write(staging/'failure.json',dict(error=f'{type(exc).__name__}: {exc}'))
        raise


def main(argv=None):
    import json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    report=merge_datasets(a.datasets,a.output)
    print(json.dumps({k:report[k] for k in ['episodes','frames','videos_reencoded']}))


if __name__=='__main__':main()
