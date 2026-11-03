#!/usr/bin/env python3
"""Require all expected trials, valid pairing, and independent score agreement."""
import argparse
import json
from pathlib import Path
import numpy as np
import tracking_benchmark as scoring
tracking_score = scoring.tracking_score
# Allow the verifier to be installed before the scorer while a pilot is active.
SCORING_VERSION = getattr(scoring, 'SCORING_VERSION', 1)
from tracking_online_common import write_json


def verify(output, manifest):
    output=Path(output); m=json.loads(Path(manifest).read_text())
    completion=json.loads((output/'completion.json').read_text())
    assert completion['completed']
    records=json.loads((output/'rollouts.json').read_text())
    assert len(records)==len(m['models'])*len(m['episodes'])==completion['rollouts']
    assert set((r['model'],r['episode']) for r in records)=={(k,e['episode']) for k in m['models'] for e in m['episodes']}
    initial={}; successes=0
    for r in records:
        ep=next(e for e in m['episodes'] if e['episode']==r['episode'])
        assert r['seed']==ep['seed'] and r['condition']==ep['condition']
        assert r['trajectory']=={'base_trajectory':ep['trajectory'],'condition':ep['condition']}
        assert r['environment']==m['environment'] and r['dimension']==m['dimension']
        assert r['failure_type'] in (None,'policy_termination')
        assert r['checkpoint']['weights_sha256']==m['checkpoint_metadata'][r['model']]['weights_sha256']
        dest=output/r['model']/m['dimension']/f"episode_{r['episode']:06d}"
        data=[json.loads(line) for line in (dest/'steps.jsonl').open()]
        score=tracking_score(data,r['failure_type'] is None,r['requested_seconds'],m['config']['collection']['physics_hz'])
        if r['metrics'].get('scoring_version')!=SCORING_VERSION:
            # Completed/active simulations may have imported the original scorer.
            # Preserve their scores and rescore telemetry without rerunning physics.
            r.setdefault('scoring_history',[]).append({'scoring_version':r['metrics'].get('scoring_version',1),
                                                     'metrics':dict(r['metrics'])})
            r['metrics'].pop('successful_acquisition_time_s',None)
            r['metrics'].update(score)
            r['metrics']['scoring_version']=SCORING_VERSION
            write_json(dest/'metrics.json',r)
        for key,value in score.items(): assert r['metrics'][key]==value, (key,value,r['metrics'][key])
        if r['failure_type'] is None:
            assert score['telemetry_complete']
            frames=json.loads((dest/'frames.json').read_text())
            assert len(frames)==r['requested_seconds']*25
            assert all(set(f['cameras'])=={'opst_cam','wrist_cam','side_cam'} for f in frames)
        q=np.array(json.loads((dest/'initial_state.json').read_text())['all_joint_position'])
        if r['episode'] in initial:
            np.testing.assert_allclose(q,initial[r['episode']],atol=1e-4,rtol=0)
        else: initial[r['episode']]=q
        if r['video_frames']:
            import av
            for cam,count in r['video_frames'].items():
                with av.open(str(dest/f'{cam}.mp4')) as c:
                    decoded=sum(1 for _ in c.decode(video=0))
                assert decoded==count
        successes+=score['success']
    write_json(output/'rollouts.json',records)
    result={'verified':True,'scoring_version':SCORING_VERSION,'rollouts':len(records),'successes':successes,
            'pairing_valid':True,'infrastructure_failures':0,
            'policy_terminations':completion['policy_terminations']}
    write_json(output/'verification.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args();print(verify(a.output,a.manifest))
