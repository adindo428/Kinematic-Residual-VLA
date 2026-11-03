#!/usr/bin/env python3
"""Apply the accepted 25% rule to saved comparisons without changing originals."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from tracking_benchmark import MODELS, SCORING_VERSION, SUCCESS_DEFINITION, TIME_DEFINITION, tracking_score
from tracking_online_common import write_json


def rescore(previous,output):
    previous=Path(previous);output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.load(open(previous/'manifest.json'))
    records=[];groups={}
    for model in MODELS:
        paths=sorted((previous/model).glob('*/episode_*/metrics.json'))
        assert len(paths)==30
        for path in paths:
            original=json.load(open(path))
            assert original['failure_type'] in (None,'policy_termination')
            data=[json.loads(line) for line in (path.parent/'steps.jsonl').open()]
            completed=bool(original['metrics']['completed'])
            if completed:
                assert original['failure_type'] is None and not original['error']
                assert original['recorded_seconds']==original['requested_seconds']
            score=tracking_score(data,completed,original['requested_seconds'],200,require_terminal_state=False)
            if completed:assert score['telemetry_complete']
            row={'model':model,'episode':original['episode'],'speed_multiplier':original['speed_multiplier'],
                 'completed':score['completed'],'success':score['success'],
                 'whole_episode_within_2cm_percent':100*score['whole_episode_within_2cm_fraction'],
                 'failure_capped_time_s':score['failure_capped_time_s'],
                 'failure_type':original['failure_type'],'source':str(path)}
            records.append(row)
        for speed in (1.,1.5,2.):
            rows=[r for r in records if r['model']==model and r['speed_multiplier']==speed]
            assert len(rows)==10
            groups[f'{model}_{speed:g}x']={'model_label':manifest['model_labels'][model],
                'episodes':10,'completed':sum(r['completed'] for r in rows),
                'successes':sum(r['success'] for r in rows),
                'success_percent':10*sum(r['success'] for r in rows),
                'mean_whole_episode_within_2cm_percent':float(np.mean([r['whole_episode_within_2cm_percent'] for r in rows])),
                'mean_failure_capped_time_s':float(np.mean([r['failure_capped_time_s'] for r in rows]))}
    write_json(output/'summary.json',{'scoring_version':SCORING_VERSION,'success_definition':SUCCESS_DEFINITION,
        'time_definition':TIME_DEFINITION,'source':str(previous),'original_results_unchanged':True,
        'recorded_rollouts':120,'scope':'Rescoring the earlier 10 held-out tracking trajectories at 1x, 1.5x and 2x; not the new nine-dimension suite.',
        'groups':groups})
    with (output/'per_episode.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print(json.dumps(groups,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rescore(a.previous,a.output)
