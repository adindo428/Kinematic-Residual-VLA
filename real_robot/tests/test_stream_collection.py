"""Background collection publishing with fake Hub calls and real snapshots."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from real_robot.stream_upload import complete_rows,snapshot,save,run as run_worker,process_job
from real_robot.collect_static_dataset import blocks,collector_command,run_plan,merge_split_blocks,reconcile_segment
from real_robot.dom_dataset import export_sessions,jsonl_read,json_write
from real_robot.dataset_quality import process
from real_robot.tests.test_dom_collection import make_session


@pytest.fixture
def stats_namespace(monkeypatch):
    try:import lerobot.common.datasets.compute_stats
    except ModuleNotFoundError:
        import lerobot.datasets.compute_stats as stats
        monkeypatch.setitem(sys.modules,'lerobot.common.datasets.compute_stats',stats)


def test_live_snapshot_ignores_partial_log_line_and_never_includes_open_attempt(tmp_path,stats_namespace):
    recording=tmp_path/'recording';manifest,episode=make_session(recording)
    with (recording/'robot.jsonl').open('a') as stream:stream.write('{"partial":')
    assert len(complete_rows(recording/'robot.jsonl'))==76
    frozen=snapshot(recording,[episode],tmp_path/'snapshot')
    assert len(list((frozen/'episodes').glob('*.json')))==1
    before=(frozen/'robot.jsonl').read_bytes()
    with (recording/'robot.jsonl').open('a') as stream:stream.write(' "later"}\n')
    assert (frozen/'robot.jsonl').read_bytes()==before
    # Export canonical source IDs so an eventual full export cannot duplicate it.
    export=tmp_path/'dom';report=export_sessions([frozen],export)
    assert report['episodes']==1
    source=jsonl_read(export/'meta/sources.jsonl')[0]
    assert source['raw_session']==str(recording.resolve())
    assert source['source_episode']==str(recording.resolve()/'episodes/episode_000000.json')


@pytest.mark.parametrize('split',['validation','test'])
def test_heldout_only_batch_is_processed_without_training_statistics(tmp_path,stats_namespace,split):
    recording=tmp_path/'recording';make_session(recording,split=split)
    raw=tmp_path/'dom';export_sessions([recording],raw)
    output=tmp_path/'processed';process(raw,output,repair=True)
    assert json.loads((output/'meta/splits.json').read_text())[split]==[0]
    assert not (output/'meta/train_stats.json').exists()


def fake_episode(session,i):
    row=dict(episode_id=i,outcome='success',start_monotonic_s=i*2.,end_monotonic_s=i*2.+1.)
    save(session/'recording/episodes'/f'episode_{i:06d}.json',row)


def test_failed_upload_retries_original_batch_before_new_episodes_and_does_not_repeat_success(tmp_path):
    session=tmp_path/'session';root=session/'stream-upload'
    save(root/'config.json',dict(batch_episodes=2,repo='owner/dataset',public=False))
    fake_episode(session,0)
    def failed(root,job,config,**kwargs):
        job['export']='/saved/encoded/batch';save(root/'jobs'/job['id']/'job.json',job)
        raise ConnectionError('offline')
    with patch('real_robot.stream_upload.process_job',side_effect=failed):
        assert run_worker(session,once=True)==1
    fake_episode(session,1);fake_episode(session,2)
    calls=[]
    def uploaded(root,job,config,**kwargs):
        calls.append([e['episode_id'] for e in job['episodes']])
        if calls==[[0]]:assert job['export']=='/saved/encoded/batch'
        job['stage']='uploaded';save(root/'jobs'/job['id']/'job.json',job)
        return dict(commit='abc',dataset_path='/saved',url='https://huggingface.co/datasets/owner/dataset')
    with patch('real_robot.stream_upload.process_job',side_effect=uploaded):
        assert run_worker(session,once=True)==0
        assert calls==[[0],[1,2]]
        assert run_worker(session,once=True)==0
        assert calls==[[0],[1,2]]
    state=json.loads((root/'status.json').read_text())
    assert state['uploaded_episode_ids']==[0,1,2]


def test_default_plan_uses_separate_sessions_for_80_10_10():
    plan=blocks()
    assert [(b['split'],b['episodes']) for b in plan]==[
        ('train',80),('validation',10),('test',10)]
    assert [b['episodes'] for b in blocks(size=20)]==[20,20,20,20,10,10]
    config=dict(task='marker',viewer='rerun',image_preprocessing='/fixed.json',repo='owner/data',public=False,upload_batch_episodes=5)
    command=collector_command(config,plan[-1],10,Path('/new/test-session'))
    assert command[command.index('--split')+1]=='test'
    assert '--static-object' in command and '--stream-upload' in command


def test_bad_alignment_does_not_block_upload_of_valid_episodes(tmp_path,stats_namespace):
    session=tmp_path/'session';recording=session/'recording';recording.parent.mkdir()
    _,episode=make_session(recording)
    bad=dict(episode,episode_id=1,start_monotonic_s=200.,end_monotonic_s=201.)
    json_write(recording/'episodes/episode_000001.json',bad)
    root=session/'stream-upload';root.mkdir()
    job=dict(id='batch-000000-000001',episodes=[episode,bad],stage='pending')
    with patch('real_robot.hub_upload.upload_dataset',return_value={'commit':'abc'}):
        process_job(root,job,dict(repo='owner/data',public=False),finished=True)
    assert job['exported_episode_ids']==[0]
    assert set(job['quarantined_episode_ids'])=={'1'}
    assert job['quarantined_episode_ids']['1']['reason']
    assert json.loads((recording/'episodes/episode_000001.json').read_text())['outcome']=='success'


def test_all_rejected_batch_is_recorded_without_upload_or_false_success_count(tmp_path,stats_namespace):
    session=tmp_path/'session';recording=session/'recording';recording.parent.mkdir()
    _,episode=make_session(recording)
    bad=dict(episode,episode_id=1,start_monotonic_s=200.,end_monotonic_s=201.)
    root=session/'stream-upload';root.mkdir()
    job=dict(id='batch-000001',episodes=[bad],stage='pending')
    with patch('real_robot.hub_upload.upload_dataset') as upload:
        assert process_job(root,job,dict(repo='owner/data',public=False),finished=True) is None
        upload.assert_not_called()
    assert job['stage']=='quarantined' and job['exported_episode_ids']==[]
    assert set(job['quarantined_episode_ids'])=={'1'}


def test_reconcile_counts_only_uploaded_episodes_and_preserves_rejections(tmp_path):
    session=tmp_path/'session'
    fake_episode(session,0);fake_episode(session,1)
    save(session/'recording/manifest.json',{})
    save(session/'recording/summary.json',dict(error=None,cleanup_errors=[]))
    save(session/'stream-upload/status.json',dict(state='completed',uploaded_episode_ids=[0],
         quarantined_episode_ids={'1':{'reason':'camera frame stale'}}))
    segment=dict(session=str(session))
    assert reconcile_segment(segment)==1
    assert segment['saved_successes']==2 and segment['episodes']==1
    assert '1' in segment['quarantined_episode_ids']


def test_block_migration_keeps_recorded_sessions_and_splits():
    state=dict(config={'block_size':20},blocks=blocks(size=20))
    state['blocks'][0]['segments']=[dict(session='/first',episodes=20,state='completed')]
    state['blocks'][1]['segments']=[dict(session='/second',state='pending')]
    merge_split_blocks(state,100)
    assert [b['episodes'] for b in state['blocks']]==[80,10,10]
    assert [s['session'] for s in state['blocks'][0]['segments']]==['/first','/second']
    assert state['plan_history'][0]['blocks'][1]['segments'][0]['session']=='/second'


def test_encoded_batch_is_reused_after_upload_failure_and_canonical_sources_deduplicate(tmp_path,stats_namespace):
    session=tmp_path/'session';recording=session/'recording';recording.parent.mkdir()
    _,episode=make_session(recording)
    root=session/'stream-upload';root.mkdir()
    job=dict(id='batch-000000',episodes=[episode],stage='pending')
    config=dict(repo='owner/dataset',public=False)
    with patch('real_robot.hub_upload.upload_dataset',side_effect=ConnectionError('offline')):
        with pytest.raises(ConnectionError):process_job(root,job,config,finished=True)
    encoded=Path(job['export'])
    assert (encoded/'meta/info.json').exists()
    with (patch('real_robot.dom_dataset.export_sessions',side_effect=AssertionError('Must not re-encode')),
          patch('real_robot.hub_upload.upload_dataset',return_value={'commit':'abc'}) as upload):
        assert process_job(root,job,config,finished=True)=={'commit':'abc'}
        assert upload.call_args.kwargs['append'] is True
    # A later ordinary full export has the same canonical episode identity.
    full=tmp_path/'full';export_sessions([recording],full)
    from real_robot.merge_datasets import merge_datasets
    combined=tmp_path/'combined'
    report=merge_datasets([encoded,full],combined,append=True,link_immutable=True)
    assert report['episodes']==1
    original=encoded/'data/chunk-000/episode_000000.parquet'
    assert original.stat().st_ino==(combined/'data/chunk-000/episode_000000.parquet').stat().st_ino


def test_plan_pauses_on_early_q_and_resumes_remainder_without_recollecting(tmp_path):
    state=dict(config=dict(task='orange',viewer='rerun',image_preprocessing='/fixed.json',
        repo='owner/data',public=False,upload_batch_episodes=5),blocks=[dict(split='validation',episodes=10,segments=[])])
    seen=[]
    def collect(command,**kwargs):
        seen.append(command)
        return SimpleNamespace(returncode=0)
    def reconcile(segment,run):
        count=4 if not seen[:-1] else 6
        segment.update(state='completed',episodes=count)
        return count
    with patch('real_robot.collect_static_dataset.reconcile_segment',side_effect=reconcile):
        assert run_plan(tmp_path,state,collect)==0
        assert state['state']=='paused'
        assert run_plan(tmp_path,state,collect)==0
    assert state['state']=='completed'
    assert [c[c.index('--episodes')+1] for c in seen]==['10','6']
    assert seen[0][seen[0].index('--output')+1]!=seen[1][seen[1].index('--output')+1]
