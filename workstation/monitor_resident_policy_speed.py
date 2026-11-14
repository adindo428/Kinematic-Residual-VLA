#!/usr/bin/env python3
"""Passively preserve policy-call latency and simulator-adjusted chunk cadence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import capture_table1_runtime as capture

ROOT=capture.REPO/'scratch/dom45/evaluations/krv-full-original-dom-20261106/resident_schedule'
OUT=ROOT/'policy_speed'


def summarize_log(path,window=100,chunk_size=20):
    samples=[];previous=None;post=True;episode=0
    with Path(path).open(errors='replace') as stream:
        for line in stream:
            if not line.endswith('\n'):continue
            if capture.TASK.search(line):episode+=1;post=False;previous=None
            if capture.DONE.search(line):post=True;previous=None
            match=capture.NATIVE.search(line);clock=capture.CLOCK.search(line)
            if post or not match or not clock:continue
            at=datetime.strptime(clock[0],'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=capture.LOCAL).timestamp()
            index=int(match[1]);interval=None
            if previous is not None and index>previous[1] and at>previous[0]:interval=at-previous[0]
            samples.append({'at':at,'index':index,'episode':episode,'ms':float(match[2])*1000,
                            'intentional_sleep_ms':float(match[3])*1000,'log_interval_s':interval})
            previous=(at,index)
    recent=samples[-window:];latency=capture.stats([r['ms'] for r in recent])
    intervals=[r['log_interval_s'] for r in recent if r['log_interval_s'] is not None]
    median_interval=statistics.median(intervals) if intervals else None
    mean_ms=latency['mean_ms']
    return {'total_scored_chunks':len(samples),'window_chunks':window,'latency':latency,
            'nominal_chunks_per_s_from_mean_latency':1000/mean_ms if mean_ms else None,
            'action_equivalents_per_s_from_mean_latency':chunk_size*1000/mean_ms if mean_ms else None,
            'median_intentional_sleep_ms':statistics.median(r['intentional_sleep_ms'] for r in recent) if recent else None,
            'within_episode_log_intervals':len(intervals),'median_within_episode_log_interval_s':median_interval,
            'wall_log_cadence_hz':1/median_interval if median_interval else None,
            'chunk_size':chunk_size,'last_sample_utc':datetime.fromtimestamp(recent[-1]['at'],timezone.utc).isoformat() if recent else None,
            'measurement_boundary':'Native host policy-call timer; excludes intentional dt_scale sleep, final CPU action transfer and explicit CUDA synchronization.',
            'rate_interpretation':'Nominal and action-equivalent Hz are reciprocals of logged mean latency, not achieved control frequency. Wall Hz uses adjacent scored log timestamps within episodes and includes simulator adjustment; it is not an action-publication timestamp.',
            'source_log':str(path)}


def refresh():
    state=capture.read(ROOT/'status.json')
    if state.get('stage') in ('paused','complete') and (ROOT/'additional_models.json').exists():
        for model in capture.read(ROOT/'additional_models.json'):
            path=Path(model['control_root'])/'status.json'
            if not path.exists():continue
            candidate=capture.read(path)
            if candidate.get('stage') in ('loading_resident_policies','evaluating','block_complete','starting'):
                state=candidate;break
    job=state.get('job')
    if not job:return state.get('stage')
    run=Path(state['run']);plan=capture.read(run/'plan.json')
    result={'updated_utc':datetime.now(timezone.utc).isoformat(),'evaluation_stage':state['stage'],
            'active_model':job['model'],'dimension':job['dimension'],'scene':job['scene'],
            'comparison':'Same scene, observational timing; checkpoint trial progress and sample windows can differ. Not controlled KR overhead.',
            'policies':{}}
    for peer in plan['jobs']:
        if peer['scene']!=job['scene']:continue
        log=Path(peer['folder'])/'client.log'
        if not log.exists():continue
        metrics=summarize_log(log)
        if not metrics['total_scored_chunks']:continue
        model=next(m for m in plan['models'] if m['key']==peer['model'])
        record={'updated_utc':result['updated_utc'],'model':peer['model'],'label':model['label'],
                'job':peer,'checkpoint_sha256':model['weight_sha256'],**metrics}
        capture.write(OUT/'blocks'/(peer['id']+'.json'),record)
        capture.write(OUT/(peer['model']+'.json'),record)
        result['policies'][peer['model']]=record
    capture.write(OUT/'latest.json',result)
    return state['stage']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    while True:
        stage=refresh()
        if args.once or stage in ('first_pass_queue_complete','complete','paused','failed'):return
        time.sleep(15)


if __name__=='__main__':main()
