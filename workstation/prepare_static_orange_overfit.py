"""Freeze exactly the 100 existing static-orange episodes for an isolated run."""
import argparse,copy,hashlib,json,os,shutil
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from real_robot.dom_dataset import json_write,jsonl_write,jsonl_read,write_parquet
from real_robot.calibration_acceptance import dataset_acceptance


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['all100','original'],required=True);a=p.parse_args();output=a.output.resolve()
    output.mkdir(parents=True,exist_ok=False);source=PROJECT/'scratch/real_robot/finetuning-runs/static-continuation-constant-20261112/dataset'
    data=output/'dataset';(data/'meta').mkdir(parents=True)
    old={k:jsonl_read(source/f'meta/{k}.jsonl') for k in ['episodes','sources','camera','episodes_stats']}
    ids=[r['episode_index'] for r in old['sources'] if r.get('static_object_collection') and json.loads(old['episodes'][r['episode_index']]['tasks'][0])['objects']=='rolling orange']
    if len(ids)!=100:raise ValueError('Expected exactly 100 static-orange source episodes')
    original=json.loads((source/'meta/splits.json').read_text());quality=json.loads((source/'meta/quality_policy.json').read_text());usable={};splits={s:[] for s in original};rows={k:[] for k in old};total=0;mapping=[]
    task=old['episodes'][ids[0]]['tasks'][0]
    for ep,previous in enumerate(ids):
        split=next(s for s,v in original.items() if previous in v)
        target_split='train' if a.split=='all100' else split;splits[target_split].append(ep)
        name=f'data/chunk-000/episode_{ep:06d}.parquet';table=pq.read_table(source/f'data/chunk-{previous//1000:03d}/episode_{previous:06d}.parquet').to_pylist()
        for row in table:row.update(episode_index=ep,index=total+row['frame_index'],task_index=0)
        write_parquet(table,data/name);total+=len(table)
        for key in old:
            row=copy.deepcopy(old[key][previous]);row['episode_index']=ep
            if key=='sources':row['original_split']=split;row['split']=target_split
            rows[key].append(row)
        usable[str(ep)]=quality['usable_frames'][str(previous)]
        q=data/f'quality/episode_{ep:06d}.jsonl';q.parent.mkdir(exist_ok=True);q.symlink_to((source/f'quality/episode_{previous:06d}.jsonl').resolve())
        for camera in ['opst_cam','wrist_cam','side_cam']:
            dest=data/f'videos/chunk-000/observation.images.{camera}/episode_{ep:06d}.mp4';dest.parent.mkdir(parents=True,exist_ok=True)
            dest.symlink_to((source/f'videos/chunk-{previous//1000:03d}/observation.images.{camera}/episode_{previous:06d}.mp4').resolve())
        mapping.append(dict(episode=ep,source_episode=previous,original_split=split,training_split=target_split))
    for key,values in rows.items():jsonl_write(data/f'meta/{key}.jsonl',values)
    jsonl_write(data/'meta/tasks.jsonl',[dict(task_index=0,task=task)])
    info=json.loads((source/'meta/info.json').read_text());info.update(total_episodes=100,total_frames=total,total_tasks=1,total_chunks=1,total_videos=300,splits={k:f'{v[0]}:{v[-1]+1}' for k,v in splits.items() if v})
    json_write(data/'meta/info.json',info);json_write(data/'meta/splits.json',splits)
    prep=json.loads((source/'meta/image_preprocessing.json').read_text());prep['source_sessions']=sorted({r['raw_session'] for r in rows['sources']});json_write(data/'meta/image_preprocessing.json',prep)
    quality['usable_frames']=usable;quality['base_calibration_acceptance']=dataset_acceptance(data,rows['camera']);json_write(data/'meta/quality_policy.json',quality)
    from lerobot.datasets.compute_stats import aggregate_stats
    stats=aggregate_stats([{k:{n:np.asarray(v) for n,v in values.items()} for k,values in rows['episodes_stats'][ep]['stats'].items()} for ep in splits['train']])
    json_write(data/'meta/train_stats.json',{k:{n:v.tolist() for n,v in values.items()} for k,values in stats.items()})
    c=json.loads((PROJECT/'scratch/real_robot/finetuning-runs/static-continuation-constant-20261112/dom-contract.json').read_text())
    bundle=PROJECT/'scratch/real_robot/finetuning-checkpoints';index=json.loads((bundle/'model-index.json').read_text());identity=index['aliases']['krv100-full-best'];entry=next(x for x in index['checkpoints'] if x['id']==identity);checkpoint=(bundle/entry['path']).resolve()
    c.update(instruction_mode='recorded',recipe_version='static-orange-overfit-v1',local_verified_dataset=str(data),dataset_quality_policy_sha256=sha(data/'meta/quality_policy.json'),
        dataset_source='Isolated static-orange-only derivative',families=['krv100'],checkpoint={'krv100':identity},
        checkpoint_preferences=[dict(checkpoint_id=identity,path=str(checkpoint),source='released_pretrained_krv',sha256={n:sha(checkpoint/n) for n in ['config.json','model.safetensors']})],
        microbatch=16,gradient_accumulation=1,num_workers=4,prefetch_factor=2,persistent_workers=True,batched_decode=True,
        decoded_video_cache=str(output/'decoded_rgb'),decoder_cache_bytes=64*1024**2,fused_adamw=True,
        max_epochs=100,early_stopping_patience=20,learning_rate=1e-5,weight_decay=0.,augmentation={},
        scheduler=dict(type='warmup_plateau',warmup_steps=100,patience=4,factor=.5,min_lr=1e-6),
        evaluation_split='train' if a.split=='all100' else 'validation',validation_flow_max_batches=4,
        checkpoint_every_steps=2000,first_recovery_checkpoint_step=500,checkpoint_keep_last=2,
        action_validation_indices=None,training_started=False,
        overfit_diagnostic=dict(total_episodes=100,training_episodes=len(splits['train']),split=a.split,
            independent_holdout=a.split!='all100',augmentation=False,weight_decay=0.,
            purpose='Memorize static orange with native pretrained KRV; not a generalization benchmark'))
    for key in ['sampling_weights','sampling_weights_sha256','sampling_audit','checkpoint_smoke_reports']:c.pop(key,None)
    c['action_objective'].update(validation_anchors=64,validation_noise_seeds=[12345])
    c['wandb']['run_names']={'krv100':output.name+'-krv100'}
    json_write(output/'launch_contract.json',c);json_write(output/'selection.json',dict(source=str(source),episodes=mapping,original_split_counts={k:sum(x['original_split']==k for x in mapping) for k in original},new_split_counts={k:len(v) for k,v in splits.items()},pretrained_checkpoint=str(checkpoint)))
    print(json.dumps(dict(output=str(output),split_counts={k:len(v) for k,v in splits.items()},frames=total,pretrained=str(checkpoint)),indent=2))


if __name__=='__main__':main()
