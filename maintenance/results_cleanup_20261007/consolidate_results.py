"""Consolidate retained evaluation files before removing obsolete result folders."""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path('/home/aparame/Research/Dynamic_VLA/RESULTS/tracking_comparison_four_75epoch')
OLD = ROOT/'Coordinate_NOT_Conditioned_Velocity'
RUN = ROOT/'Coordinate_Conditioned_Velocity/16787490'
CURRENT = RUN/'comparison_with_existing'
RAW = RUN/'live_tracking'
MODELS = ('baseline','velocity','residual','coordinate','coordinate_velocity')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,data):
    temporary=path.with_name(path.name+'.cleanup_tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def plan():
    assert OLD.is_dir() and not OLD.is_symlink() and RAW.is_dir() and not RAW.is_symlink()
    assert CURRENT.is_dir() and CURRENT.resolve().is_relative_to(ROOT.resolve())
    summary=json.loads((CURRENT/'summary.json').read_text())
    assert summary['complete'] and summary['pairing_valid'] and summary['recorded_rollouts']==150
    assert json.loads((RAW/'completion.json').read_text())['rollouts']==30
    assert len(summary['comparison_videos'])==30
    rows=list(CURRENT.glob('*/*/episode_*/metrics.json'))
    assert len(rows)==150
    links=sorted(p for p in CURRENT.rglob('*') if p.is_symlink())
    sources=[]
    for p in links:
        target=p.resolve(strict=True)
        assert target.is_file(),f'Unexpected linked directory: {p}'
        assert target.is_relative_to(OLD/'live_tracking') or target.is_relative_to(RAW/'coordinate_velocity')
        assert target.stat().st_dev==p.parent.stat().st_dev
        sources.append(target)
    assert len(sources)==len(set(sources)) and len(links)==2846
    expected_files=set()
    for model in MODELS:
        source=RAW if model=='coordinate_velocity' else OLD/'live_tracking'
        records=list(source.glob(f'{model}/*/episode_*/metrics.json'))
        assert len(records)==30
        for p in records:
            expected_files.update(q.resolve() for q in p.parent.iterdir() if q.is_file() and q.name!='metrics.json')
    assert set(sources)==expected_files,'Some source rollout data is not retained in the latest comparison'
    for p in ROOT.iterdir():
        assert p.name in ('Coordinate_Conditioned_Velocity','Coordinate_NOT_Conditioned_Velocity',
                          'latest_with_velocity','tracking_comparison_with_velocity.png','comparison_videos_with_velocity','README.md')
    assert not (RUN/'offline_imitation').exists()
    return {'retained_rollouts':150,'retained_comparison_videos':30,'materialized_files':len(links),
            'removed_folders':[str(OLD),str(RAW)],'retained_folder':str(CURRENT),
            'unique_offline_results_retained_at':str(RUN/'offline_imitation'),
            'removed_shortcuts':['latest_with_velocity','tracking_comparison_with_velocity.png','comparison_videos_with_velocity']},links


def apply():
    report,links=plan()
    provenance=RUN/'provenance'
    history=provenance/'source_evaluations'
    history.mkdir(exist_ok=False)
    report['started_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(provenance/'cleanup_plan.json',report)
    # Record every linked file's content before moving it, including every camera/telemetry file.
    hashes={str(p.relative_to(CURRENT)):digest(p.resolve()) for p in links}
    summary_before=(CURRENT/'summary.json').read_bytes()
    plot_before=digest(CURRENT/'tracking_comparison.png')
    comparisons_before={str(p.relative_to(CURRENT)):digest(p) for p in (CURRENT/'comparison_videos').glob('*.mp4')}
    rows_before={str(p.relative_to(CURRENT)):json.loads(p.read_text()) for p in CURRENT.glob('*/*/episode_*/metrics.json')}
    print('Verified inventory. Moving 2,846 linked files into the retained comparison.',flush=True)
    for i,p in enumerate(links,1):
        os.replace(p.resolve(strict=True),p)
        if i%500==0:print(f'Consolidated {i}/{len(links)} files',flush=True)
    assert not any(p.is_symlink() for p in CURRENT.rglob('*'))
    for relative,expected in hashes.items():assert digest(CURRENT/relative)==expected,relative
    # Fix public video paths; preserve all numerical/experimental fields exactly.
    rows=[]
    for relative,before in sorted(rows_before.items()):
        path=CURRENT/relative;after=copy.deepcopy(before)
        after['videos']={camera:str(path.parent/Path(video).name) for camera,video in before['videos'].items()}
        for video in after['videos'].values():assert Path(video).is_file()
        assert {k:v for k,v in before.items() if k!='videos'}=={k:v for k,v in after.items() if k!='videos'}
        write_json(path,after);rows.append(after)
    write_json(CURRENT/'rollouts.json',rows)
    # Retain unique offline outputs and the small original records of both source evaluations.
    offline_hashes={str(p.relative_to(OLD/'offline_imitation')):digest(p) for p in (OLD/'offline_imitation').rglob('*') if p.is_file()}
    os.rename(OLD/'offline_imitation',RUN/'offline_imitation')
    for relative,expected in offline_hashes.items():assert digest(RUN/'offline_imitation'/relative)==expected
    for source,job in ((OLD/'live_tracking','16741254'),(RAW,'16787490')):
        dest=history/job;dest.mkdir()
        for name in ('manifest.json','summary.json','completion.json','per_episode.csv','runtime.json',
                     'policy.log','simulator.log','dependencies.log','summary.log'):
            path=source/name
            if path.is_file() and not path.is_symlink():os.rename(path,dest/name)
    for path in (OLD/'provenance').glob('*.json'):
        dest=provenance/path.name
        assert not dest.exists();os.rename(path,dest)
    revision=provenance/'plot_revision';revision.mkdir()
    for old_name,new_name in (('tracking_comparison_five_methods_completion.png','previous_five_method_completion.png'),
                              ('recreate_plot_no_field.py','recreate_plot.py'),
                              ('tracking_comparison_no_field_metrics.json','metrics.json')):
        path=CURRENT/old_name
        if path.exists():os.rename(path,revision/new_name)
    duplicate_plot=CURRENT/'tracking_comparison_no_field.png'
    if duplicate_plot.exists():
        assert digest(duplicate_plot)==plot_before;duplicate_plot.unlink()
    manifest=json.loads((CURRENT/'manifest.json').read_text())
    manifest['historical_rollout_sources']=manifest['rollout_sources']
    manifest['rollout_sources']={model:str(CURRENT) for model in MODELS}
    reused=manifest['reused_evaluation']
    reused['original_path']=reused.pop('path')
    reused['retained_rollouts_path']=str(CURRENT)
    reused['original_manifest_path']=str(history/'16741254/manifest.json')
    reused['original_runtime_path']=str(history/'16741254/runtime.json')
    write_json(CURRENT/'manifest.json',manifest)
    lineage=json.loads((CURRENT/'provenance.json').read_text())
    lineage['historical_source_metrics_sha256']=lineage.pop('source_metrics_sha256')
    lineage['historical_old_results']=lineage.pop('old_results')
    lineage['historical_new_results']=lineage.pop('new_results')
    lineage['retained_results']=str(CURRENT)
    lineage['source_metrics_sha256']={str(CURRENT/relative):digest(CURRENT/relative) for relative in rows_before}
    lineage['cleanup_record']=str(provenance/'cleanup_report.json')
    write_json(CURRENT/'provenance.json',lineage)
    assert (CURRENT/'summary.json').read_bytes()==summary_before
    assert digest(CURRENT/'tracking_comparison.png')==plot_before
    for relative,expected in comparisons_before.items():assert digest(CURRENT/relative)==expected
    # Validate video headers and paired initial states before deleting any obsolete directory.
    import av,numpy as np
    states={}
    for row in rows:
        directory=Path(row['videos']['opst_cam']).parent
        state=json.loads((directory/'initial_state.json').read_text())
        key=(row['episode'],row['speed_multiplier'])
        if key in states:assert np.allclose(state['all_joint_position'],states[key],rtol=0,atol=1e-4)
        else:states[key]=state['all_joint_position']
        for camera,path in row['videos'].items():
            with av.open(path) as container:
                stream=container.streams.video[0]
                assert stream.frames==row['video_frames'][camera]
                assert (stream.width,stream.height)==(480,360)
    for video in json.loads(summary_before)['comparison_videos']:
        with av.open(video['path']) as container:
            stream=container.streams.video[0]
            assert stream.frames==250 and (stream.width,stream.height)==(2400,512)
    report.update({'verified_camera_videos':450,'verified_comparison_videos':30,
                   'all_retained_file_hashes_match':True,'plot_unchanged':True,'summary_unchanged':True,
                   'paired_initial_states_match':True,'offline_file_hashes':offline_hashes,
                   'retained_payload_sha256':hashes})
    write_json(provenance/'cleanup_report.json',report)
    print('Validated 150 trials, 450 camera videos, 30 comparison videos, unchanged metrics and plot. Removing obsolete copies/caches.',flush=True)
    # Both directories are explicit user-authorized obsolete output locations.
    for directory in (OLD,RAW):
        assert not directory.is_symlink() and directory.is_dir() and directory.resolve().is_relative_to(ROOT)
        shutil.rmtree(directory)
    for name in report['removed_shortcuts']:
        path=ROOT/name
        assert path.is_symlink();path.unlink()
    assert not OLD.exists() and not RAW.exists()
    assert not any(p.is_symlink() for p in CURRENT.rglob('*'))
    assert len(list(CURRENT.glob('*/*/episode_*/metrics.json')))==150
    (ROOT/'README.md').write_text('# Latest 75-epoch tracking comparison\n\n'
        'All retained results are under Coordinate_Conditioned_Velocity/16787490/.\n\n'
        '- comparison_with_existing/tracking_comparison.png: latest four-method plot, without Field or sample-count labels; includes mean object speeds.\n'
        '- comparison_with_existing/comparison_videos/: 30 five-panel closed-loop videos.\n'
        '- comparison_with_existing/: all 150 trials, all camera videos, telemetry, summary and per-episode CSV.\n'
        '- offline_imitation/: unique offline results from the original four checkpoints; no Coordinate+Velocity offline run.\n'
        '- provenance/: training records, original source summaries/logs, plot revision and consolidation checks.\n\n'
        'The obsolete Coordinate_NOT_Conditioned_Velocity directory and repeated standalone evaluation outputs were removed after the retained data were consolidated and verified.\n')
    (RUN/'README.md').write_text('# Consolidated Coordinate+Velocity comparison\n\n'
        'comparison_with_existing/ is the self-contained retained evaluation: 150 trials across five methods, including all raw camera videos and telemetry. '
        'The public plot shows the requested four methods; the existing comparison videos still show all five.\n\n'
        'offline_imitation/ retains unique original four-checkpoint offline results. provenance/ retains training records, original manifests/summaries/logs and cleanup verification.\n'
        'The former standalone live_tracking/ and old Coordinate_NOT_Conditioned_Velocity/ directories are no longer required.\n')
    (CURRENT/'README.md').write_text('# Retained closed-loop tracking comparison\n\n'
        'This directory contains 150 recorded trials for DOM-VLA, DOM-VLA+Velocity, KR-VLA-Field, KR-VLA-Coordinate and KR-VLA-Coordinate+Velocity. '
        'All camera videos, action predictions, initial states and telemetry are regular files here; no old result folder is needed.\n\n'
        'tracking_comparison.png is the revised four-method plot: Field is omitted, sample-count labels are removed, and the right panel shows time within 2 cm of the hover target. '
        'The mean object speeds are 0.091, 0.137 and 0.183 m/s at 1x, 1.5x and 2x.\n\n'
        'comparison_videos/ retains the 30 original five-panel videos. summary.json and per_episode.csv retain every trial and method, including the original Coordinate early termination at 2x. '
        'Only Coordinate+Velocity was newly evaluated; the other 120 trials were reused.\n\n'
        'Metrics use completed rollouts. Simulation pauses for inference. Velocity is the current simulator signal in robot-base coordinates. '
        'Training records and original source summaries are in ../provenance/.\n')
    report['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(provenance/'cleanup_report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('retained_payload_sha256','offline_file_hashes')},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true')
    if p.parse_args().apply:apply()
    else:print(json.dumps(plan()[0],indent=2))
