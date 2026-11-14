"""Select moving + static orange episodes without starting or changing training."""
import argparse,hashlib,json
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=PROJECT/'scratch/real_robot/finetuning-runs/static-continuation-constant-20261112/dataset')
    p.add_argument('--output',type=Path,default=PROJECT/'scratch/real_robot/training_plans/smolvla-kr-full-orange-20261113/selection.json')
    a=p.parse_args();source=a.source.resolve();metadata={r['episode_index']:r for r in map(json.loads,(source/'meta/episodes.jsonl').read_text().splitlines())}
    sources={r['episode_index']:r for r in map(json.loads,(source/'meta/sources.jsonl').read_text().splitlines())};splits=json.loads((source/'meta/splits.json').read_text());quality=json.loads((source/'meta/quality_policy.json').read_text());selection={};summary={};seen=set()
    for split,episodes in splits.items():
        selection[split]=[];counts=dict(moving_collection=0,static_collection=0,eligible_anchors=0)
        for ep in episodes:
            task=json.loads(metadata[ep]['tasks'][0])
            if task.get('objects')!='rolling orange':continue
            row=sources[ep];identity=row['source_episode']
            if identity in seen:raise ValueError('Duplicate orange source episode')
            seen.add(identity);static=bool(row.get('static_object_collection'));kind='static_collection' if static else 'moving_collection';counts[kind]+=1
            usable=set(quality['usable_frames'][str(ep)]);anchors=sum(max(0,i-2) in usable for i in usable);counts['eligible_anchors']+=anchors
            selection[split].append(dict(source_episode_index=ep,source_episode=identity,static_collection=static,eligible_anchors=anchors))
        counts['episodes']=len(selection[split]);summary[split]=counts
    report=dict(source=str(source),quality_policy_sha256=hashlib.sha256((source/'meta/quality_policy.json').read_bytes()).hexdigest(),
        family='smolvla_kr',prompt='Pick up the rolling orange and place it in the box.',summary=summary,episodes=selection,
        preserve_original_splits=True,training_started=False,initialization='awaiting clarification: random entire model vs original backbone without prior task fine-tuning',
        note='Moving/static are collection categories, not measured speed labels for every frame. No weights are loaded and no dataset is modified.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
