"""Outcome-supported grasp annotation from saved wrist images and measured motion.

Automatic concurrence is explicitly distinguished from human-reviewed holding.
No robot access or Hub writes. The source dataset is never modified.
"""
import argparse
from bisect import bisect_right
from collections import Counter
import copy
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pyarrow.parquet as pq
from .dom_dataset import DATA_PATH,json_write,jsonl_read,jsonl_write,numeric_stats,sha256,write_parquet
from .grasp_phase import MAX_HOLDING_GRIPPER_NORMALIZED,MINIMUM_HOLDING_LIFT_M

METHOD='success_wrist_color_measured_close_lift_v2'


def lift_intervals(rows, measured):
    """Candidates require measured closure and sustained >=20mm measured lift.

    A grasped object prevents the gripper reaching its empty fully-closed value;
    allow normalized aperture <=0.35, but require an earlier open state. Cut at
    the first reopening trend, not at the end of the episode. Retries stay separate.
    """
    commands=np.array([r['action'][-1] for r in rows]);z=np.array([r['observation.state'][2] for r in rows])
    measured=np.asarray(measured,float);closed=(commands<=MAX_HOLDING_GRIPPER_NORMALIZED)&np.isfinite(measured)&(measured<=MAX_HOLDING_GRIPPER_NORMALIZED)
    candidates=[];seen_open=False;i=0
    while i<len(rows):
        if commands[i]>.5 and np.isfinite(measured[i]) and measured[i]>.5:seen_open=True
        if not closed[i]:i+=1;continue
        begin=i
        while i<len(rows) and closed[i]:i+=1
        end=i
        if not seen_open or end-begin<8:continue
        seen_open=False
        baseline=z[begin];minimum=baseline;confirm=None
        for j in range(begin,end-2):
            minimum=min(minimum,z[j])
            if j-begin>=5 and np.all(z[j:j+3]-minimum>=MINIMUM_HOLDING_LIFT_M):confirm=j;baseline=minimum;break
        if confirm is None:continue
        command_min=commands[begin:confirm+1].min();measured_min=measured[begin:confirm+1].min()
        stop=end
        for j in range(confirm+1,end):
            command_min=min(command_min,commands[j]);measured_min=min(measured_min,measured[j])
            if commands[j]>command_min+.15 or measured[j]>measured_min+.15:stop=j;break
        if stop-confirm>=4:
            candidates.append(dict(close_frame=int(begin),start_frame=int(confirm),end_frame=int(stop),
                lift_m=float(z[confirm]-baseline),baseline_z_m=float(baseline),
                measured_gripper_at_lift=float(measured[confirm])))
    return candidates


def colored_object(frame, task):
    """Restricted fingertip ROI; returns visual evidence, never a label alone."""
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);height,width=frame.shape[:2]
    low,high=([35,85,45],[90,255,255]) if task=='marker' else ([0,130,65],[24,255,255])
    mask=cv2.inRange(hsv,np.array(low,np.uint8),np.array(high,np.uint8))
    mask[:int(height*.55)]=0;mask[:,:int(width*.22)]=0;mask[:,int(width*.78):]=0
    if task=='marker':
        # White lettering fragments the green barrel. Join those small gaps and
        # follow the barrel, rather than switching among tiny colored letters.
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
    options=[];minimum=16 if task=='marker' else 55
    for k in range(1,n):
        x,y,w,h,area=stats[k];cx,cy=centers[k]
        if area<minimum or area>height*width*.16:continue
        score=-float(area) if task=='marker' else np.linalg.norm(np.array([cx/width-.5,cy/height-.84]))
        options.append((score,
            dict(area_px=int(area),centroid_px=[float(cx),float(cy)],bbox=[int(x),int(y),int(w),int(h)])))
    return min(options,key=lambda x:x[0])[1] if options else None


def visual_concurrence(frames, task):
    evidence=[colored_object(f,task) for f in frames]
    if any(e is None for e in evidence):return False,evidence,'object color not visible in every sampled wrist frame'
    centers=np.array([e['centroid_px'] for e in evidence]);areas=np.array([e['area_px'] for e in evidence])
    span=float(np.max(np.linalg.norm(centers-np.median(centers,axis=0),axis=1)))
    if span>frames[0].shape[1]*.075:return False,evidence,'object moves relative to wrist/fingertips'
    if areas.max()/areas.min()>3.5:return False,evidence,'object appearance/visibility changes too much'
    return True,evidence,'success + measured closure/lift + stable wrist object appearance'


def measured_session(session):
    manifest_path=session/'manifest.json';robot=session/'robot.jsonl'
    if not manifest_path.exists() or not robot.exists():return np.array([]),np.array([])
    manifest=json.loads(manifest_path.read_text());closed=manifest['gripper_closed_percent'];opened=manifest['gripper_open_percent']
    times=[];grips=[]
    for line in robot.open():
        r=json.loads(line)
        if r.get('follower_calibrated') is None:continue
        times.append(r['host_read_end_monotonic_s']);grips.append(2*(r['follower_calibrated'][-1]-closed)/(opened-closed)-1)
    return np.asarray(times),np.asarray(grips)


def prepare(dataset,output):
    dataset,output=Path(dataset).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False);(output/'sheets').mkdir();cv2.setNumThreads(1)
    sources=jsonl_read(dataset/'meta/sources.jsonl');episodes=jsonl_read(dataset/'meta/episodes.jsonl')
    report=[];annotations={};cache={};page=[];page_records=[];page_number=0
    def flush():
        nonlocal page,page_records,page_number
        if not page:return
        name=f'sheets/page-{page_number:03d}.jpg';cv2.imwrite(str(output/name),np.concatenate(page,axis=0))
        for record in page_records:record['contact_sheet']=name
        page=[];page_records=[];page_number+=1
    for source,episode in zip(sources,episodes):
        ep=episode['episode_index'];session=Path(source['raw_session'])
        raw_episode=Path(source['source_episode'])
        saved=json.loads(raw_episode.read_text()) if raw_episode.exists() else {}
        task=json.loads(episode['tasks'][0])['objects'];task='marker' if 'marker' in task else 'striped_ball' if 'striped' in task else 'orange'
        rows=pq.read_table(dataset/DATA_PATH.format(episode_chunk=ep//1000,episode_index=ep)).to_pylist()
        quality=jsonl_read(dataset/f'quality/episode_{ep:06d}.jsonl')
        if session not in cache:cache[session]=measured_session(session)
        times,grips=cache[session];measured=[]
        for q in quality:
            t=q['anchor_monotonic_s'];j=bisect_right(times,t)-1
            measured.append(grips[j] if j>=0 and 0<=t-times[j]<=.12 else np.nan)
        candidates=lift_intervals(rows,measured)
        outcome=saved.get('outcome');record=dict(episode_index=ep,task=task,source_episode=source['source_episode'],
            outcome='success',outcome_source='saved_success' if outcome=='success' else 'operator_declared_all_saved_episodes_successful',
            raw_measured_gripper_available=bool(len(times)),expected_table_contact=task=='marker',
            table_contact_source='operator task-level statement; no per-frame force/contact measurement',candidates=[])
        # A contradictory saved outcome is never silently overwritten.
        if outcome not in (None,'success'):raise ValueError(f'Episode {ep} contradicts the success declaration: {outcome}')
        video=dataset/f'videos/chunk-{ep//1000:03d}/observation.images.wrist_cam/episode_{ep:06d}.mp4'
        cap=cv2.VideoCapture(str(video));accepted=[]
        for candidate in candidates:
            start,end=candidate['start_frame'],candidate['end_frame']
            indices=np.unique(np.linspace(start,end-1,max(3,int(np.ceil((end-start)/8))+1)).astype(int)).tolist()
            frames=[]
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_MSEC,rows[i]['timestamp']*1000);ok,image=cap.read()
                if not ok:raise ValueError(f'Cannot decode wrist ep{ep} frame{i}')
                frames.append(image)
            good,evidence,note=visual_concurrence(frames,task)
            candidate.update(accepted_automatically=bool(good),visual_sample_frames=indices,visual_evidence=evidence,reason=note)
            if good:accepted.append([start,end])
            record['candidates'].append(candidate)
            # Review sheets show pre-closure, lift, carry, last-held and post-interval.
            review_frames=[max(0,candidate['close_frame']-4),start,(start+end-1)//2,end-1,min(len(rows)-1,end+5)]
            tiles=[]
            for i in review_frames:
                cap.set(cv2.CAP_PROP_POS_MSEC,rows[i]['timestamp']*1000);ok,image=cap.read()
                if not ok:raise ValueError(f'Cannot decode review ep{ep} frame{i}')
                image=cv2.resize(image,(180,180));tile=np.zeros((216,180,3),np.uint8);tile[36:]=image
                cv2.putText(tile,f'ep{ep} f{i} {task}',(3,13),0,.34,(255,255,255),1)
                cv2.putText(tile,f'z{rows[i]["observation.state"][2]:.3f} g{measured[i]:.2f} '+('AUTO' if good else 'REVIEW'),(3,28),0,.31,(0,255,100) if good else (0,200,255),1)
                tiles.append(tile)
            page.append(np.concatenate(tiles,axis=1));page_records.append(candidate)
            if len(page)==6:flush()
        cap.release()
        annotations[str(ep)]=dict(held_intervals=accepted,annotation_kind='automatic_success_visual_telemetry',
            method=METHOD,manual_visual_review=False,review_note='Automatic sampled wrist-color concurrence; not a human-reviewed interval.',
            outcome=record['outcome'],outcome_source=record['outcome_source'],expected_table_contact=record['expected_table_contact'])
        report.append(record)
        if (ep+1)%50==0:print(f'Prepared {ep+1}/{len(episodes)} episodes',flush=True)
    flush()
    json_write(output/'annotations.json',annotations)
    summary=dict(schema='krvla_grasp_annotation_review_v1',dataset=str(dataset),method=METHOD,
        source_quality_policy_sha256=sha256(dataset/'meta/quality_policy.json'),episodes=len(report),
        automatic_held_episodes=sum(bool(a['held_intervals']) for a in annotations.values()),
        automatic_held_frames=sum(b-a for item in annotations.values() for a,b in item['held_intervals']),
        unresolved_episodes=[r['episode_index'] for r in report if not annotations[str(r['episode_index'])]['held_intervals']],
        note='Success is episode ground truth, not pickup timing. Automated intervals require wrist appearance AND measured closure/lift; manual confirmations remain separate.',
        operator_declaration='All episodes I saved are successful',
        episode_reports=report)
    json_write(output/'review.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('episode_reports','unresolved_episodes')},indent=2))
    return summary


def refine_marker_visuals(dataset,review_directory,output):
    """Reuse telemetry candidates; refine marker color-fragment association only."""
    dataset,review_directory,output=map(lambda p:Path(p).resolve(),(dataset,review_directory,output))
    output.mkdir(parents=True,exist_ok=False);(output/'sheets').symlink_to(review_directory/'sheets',target_is_directory=True)
    report=json.loads((review_directory/'review.json').read_text());annotations=json.loads((review_directory/'annotations.json').read_text())
    cv2.setNumThreads(1)
    for record in report['episode_reports']:
        ep=record['episode_index'];annotation=annotations[str(ep)];annotation['method']=METHOD
        if record['task']!='marker':continue
        video=dataset/f'videos/chunk-{ep//1000:03d}/observation.images.wrist_cam/episode_{ep:06d}.mp4'
        cap=cv2.VideoCapture(str(video));accepted=[]
        for candidate in record['candidates']:
            frames=[]
            for i in candidate['visual_sample_frames']:
                cap.set(cv2.CAP_PROP_POS_MSEC,i*40);ok,image=cap.read()
                if not ok:raise ValueError(f'Cannot decode marker ep{ep}:{i}')
                frames.append(image)
            good,evidence,note=visual_concurrence(frames,'marker')
            candidate['previous_visual_association']=dict(accepted=candidate['accepted_automatically'],reason=candidate['reason'])
            candidate.update(accepted_automatically=bool(good),visual_evidence=evidence,reason=note)
            if good:accepted.append([candidate['start_frame'],candidate['end_frame']])
        cap.release();annotation['held_intervals']=accepted
    report.update(method=METHOD,review_parent=str(review_directory),
        automatic_held_episodes=sum(bool(a['held_intervals']) for a in annotations.values()),
        automatic_held_frames=sum(b-a for item in annotations.values() for a,b in item['held_intervals']),
        unresolved_episodes=[int(ep) for ep,a in annotations.items() if not a['held_intervals']],
        contact_sheet_note='Sheets reused from the initial pass; their AUTO/REVIEW text is historical. JSON decisions use refined barrel-component association.')
    json_write(output/'annotations.json',annotations);json_write(output/'review.json',report)
    print(json.dumps({k:report[k] for k in ['method','automatic_held_episodes','automatic_held_frames']},indent=2))
    return report


def apply(dataset,annotations_path,output):
    """New derivative; preserve existing repairs, splits, actions and manual labels."""
    dataset,annotations_path,output=map(lambda p:Path(p).resolve(),(dataset,annotations_path,output))
    if output.exists() or output==dataset or dataset in output.parents:raise ValueError('Choose a new output outside the source dataset')
    annotations=json.loads(annotations_path.read_text());staging=output.with_name(output.name+'.incomplete');staging.mkdir(parents=True)
    shutil.copytree(dataset/'meta',staging/'meta');(staging/'videos').symlink_to(dataset/'videos',target_is_directory=True)
    sources=jsonl_read(dataset/'meta/sources.jsonl');stats=jsonl_read(dataset/'meta/episodes_stats.jsonl')
    policy=json.loads((dataset/'meta/quality_policy.json').read_text());counts=Counter();usable={}
    for source in sources:
        ep=source['episode_index'];name=DATA_PATH.format(episode_chunk=ep//1000,episode_index=ep);qname=f'quality/episode_{ep:06d}.jsonl'
        original=pq.read_table(dataset/name).to_pylist();rows=copy.deepcopy(original);quality=jsonl_read(dataset/qname)
        annotation=annotations[str(ep)];held=np.zeros(len(rows),bool);reviewed_held=np.zeros(len(rows),bool)
        if annotation.get('outcome')!='success':raise ValueError('Held annotations require a successful episode')
        for begin,end in annotation['held_intervals']:
            if not isinstance(begin,int) or not isinstance(end,int) or not 0<=begin<end<=len(rows):raise ValueError('Invalid held interval')
            held[begin:end]=True
        for begin,end in annotation.get('manual_reviewed_intervals',[]):
            if not isinstance(begin,int) or not isinstance(end,int) or not 0<=begin<end<=len(rows):raise ValueError('Invalid reviewed interval')
            reviewed_held[begin:end]=True
        if np.any(reviewed_held & ~held):raise ValueError('Reviewed holding must be included in the accepted intervals')
        for i,q in enumerate(quality):
            q['episode_success']=True;q['episode_success_source']=annotation.get('outcome_source','operator_declared_all_saved_episodes_successful')
            q['task_table_contact_expected']=annotation['expected_table_contact']
            if held[i] and not q.get('phase_confirmed',False):
                reviewed=bool(reviewed_held[i] or annotation.get('manual_visual_review',False))
                rows[i]['observation.environment_state'][6:9]=[0.,0.,0.]
                q.update(manipulation_phase='held_confirmed' if reviewed else 'held_success_inferred',phase_confirmed=reviewed,
                    phase_annotation_source='sampled_wrist_measured_close_lift_review_v1' if reviewed else annotation['method'],phase_annotation_manual_review=reviewed,
                    phase_annotation_kind='sampled_wrist_measured_close_lift_review' if reviewed else annotation['annotation_kind'],velocity_required=False,
                    velocity_input_source='neutral_after_success_supported_grasp',tracking_usable=True)
                counts['new_manual_confirmed_frames' if reviewed else 'automatic_held_frames']+=1
        # Count against the source, keeping all original repair provenance.
        previous=jsonl_read(dataset/qname)
        counts['newly_usable_frames']+=sum(q['tracking_usable'] and not p['tracking_usable'] for q,p in zip(quality,previous))
        counts['manual_confirmed_frames']+=sum(q.get('phase_confirmed',False) for q in quality)
        counts['episodes']+=1;counts['frames']+=len(rows)
        for before,after in zip(original,rows):
            if any(before[k]!=after[k] for k in before if k!='observation.environment_state'):raise ValueError('Annotation modified robot/actions/timing')
            if before['observation.environment_state'][:6]!=after['observation.environment_state'][:6]:raise ValueError('Annotation modified object pose')
        write_parquet(rows,staging/name);jsonl_write(staging/qname,quality)
        usable[str(ep)]=[i for i,q in enumerate(quality) if q['tracking_usable']]
        geometry=[r['observation.environment_state'] for r,q in zip(rows,quality) if q['tracking_usable']]
        if geometry:stats[ep]['stats']['observation.environment_state']=numeric_stats(geometry)
        source['grasp_annotation']=dict(method=annotation['method'],accepted_held_intervals=annotation['held_intervals'],
            manually_reviewed_intervals=annotation.get('manual_reviewed_intervals',[]),
            annotation_sha256=sha256(annotations_path),manual_confirmations_preserved=True)
    jsonl_write(staging/'meta/sources.jsonl',sources);jsonl_write(staging/'meta/episodes_stats.jsonl',stats)
    policy.update(usable_frames=usable,automatic_held_policy=METHOD,
        held_velocity='Neutral model input for manually confirmed or success+wrist+measured-closure/lift-supported holding; not physical zero velocity')
    json_write(staging/'meta/quality_policy.json',policy)
    splits=json.loads((staging/'meta/splits.json').read_text())
    from lerobot.common.datasets.compute_stats import aggregate_stats
    aggregate=aggregate_stats([{k:{n:np.asarray(v) for n,v in values.items()} for k,values in stats[ep]['stats'].items()} for ep in splits['train']])
    json_write(staging/'meta/train_stats.json',{k:{n:v.tolist() for n,v in values.items()} for k,values in aggregate.items()})
    shutil.copy2(annotations_path,staging/'meta/grasp_annotations.json')
    report=dict(schema='krvla_grasp_annotation_derivative_v1',complete=True,source_dataset=str(dataset),
        source_quality_policy_sha256=sha256(dataset/'meta/quality_policy.json'),annotation_sha256=sha256(annotations_path),
        method=METHOD,counts=dict(counts),splits_unchanged=True,robot_actions_timing_unchanged=True,
        object_positions_unchanged=True,depth_correction_reapplied=False,
        automatic_is_not_manual_confirmation=True,held_velocity_is_neutral_input=True)
    json_write(staging/'grasp_annotation_report.json',report)
    json_write(staging/'export_report.json',dict(complete=True,episodes=len(sources),frames=counts['frames'],source_dataset=str(dataset),grasp_annotation_report='grasp_annotation_report.json'))
    (staging/'README.md').write_text('# Grasp-annotated local training derivative\n\nSee grasp_annotation_report.json and meta/grasp_annotations.json. Automatic success-supported intervals are distinguished from manual confirmations. Videos are reused; raw data and splits are unchanged.\n')
    staging.rename(output);print(json.dumps(report,indent=2));return report


def review_decisions(directory,decisions_path):
    """Overlay explicit visual decisions; preserve the original automated output."""
    directory=Path(directory);annotations=json.loads((directory/'annotations.json').read_text())
    report=json.loads((directory/'review.json').read_text());episodes={r['episode_index']:r for r in report['episode_reports']}
    decisions=json.loads(Path(decisions_path).read_text())
    for decision in decisions:
        ep=decision['episode_index'];annotation=annotations[str(ep)]
        matches=[c for c in episodes[ep]['candidates'] if c['start_frame']==decision['start_frame']]
        if len(matches)!=1 or matches[0]['contact_sheet']!=decision['contact_sheet']:raise ValueError('Review decision does not identify a unique candidate/sheet')
        candidate=matches[0];interval=[candidate['start_frame'],candidate['end_frame']]
        candidate['manual_review']=decision
        if decision['decision']=='accept':
            if interval not in annotation['held_intervals']:annotation['held_intervals'].append(interval)
            annotation.setdefault('manual_reviewed_intervals',[]).append(interval)
        elif decision['decision']=='reject':
            annotation['held_intervals']=[v for v in annotation['held_intervals'] if v!=interval]
        else:raise ValueError('Unknown review decision')
    for ep,annotation in annotations.items():
        annotation['outcome_source']=episodes[int(ep)]['outcome_source']
        annotation['held_intervals'].sort()
    report['manual_review_decisions']=decisions
    report['accepted_held_episodes']=sum(bool(a['held_intervals']) for a in annotations.values())
    report['unresolved_episodes']=[int(ep) for ep,a in annotations.items() if not a['held_intervals']]
    report['operator_declaration']='All episodes I saved are successful'
    report['visual_criteria']=dict(marker_hsv=[[35,85,45],[90,255,255]],orange_hsv=[[0,130,65],[24,255,255]],
        fingertip_roi_fraction=[.22,.55,.78,1.],max_centroid_radius_fraction=.075,max_color_area_ratio=3.5,
        measured_gripper_max_normalized=.35,minimum_sustained_lift_m=.020,
        marker_component_selection='largest component after 5x5 color-mask closing; white lettering does not become separate target objects',
        annotation_code_sha256=sha256(Path(__file__)))
    json_write(directory/'annotations_reviewed.json',annotations)
    json_write(directory/'review_with_decisions.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','refine','apply'])
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--annotations',type=Path);args=parser.parse_args()
    if args.mode=='prepare':prepare(args.dataset,args.output)
    elif args.mode=='refine':
        if args.annotations is None:parser.error('refine requires --annotations pointing to the original review directory')
        refine_marker_visuals(args.dataset,args.annotations,args.output)
    else:
        if args.annotations is None:parser.error('apply requires --annotations')
        apply(args.dataset,args.annotations,args.output)


if __name__=='__main__':main()
