import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workstation'))
from profile_family_performance import select_configuration
import run_benchmarked_family_campaign as controller


def entry(name):return dict(family=name,sha256=name+'-weights')


def report(item):
    return dict(family=item['family'],checkpoint_sha256=item['sha256'],state='completed',passed=True,effective_batch=16,
        architecture_changes=False,architecture_signature='fixture',weights_saved=False,
        selected=dict(flow_microbatch=4,auxiliary_microbatch=2,loader_workers=2,prefetch_factor=2,decoder_cache_episodes=8,vectorized_photometric=True))


def test_select_fastest_configuration_with_real_memory_headroom():
    rows=[dict(passed=True,mean_update_s=.3,peak_reserved_bytes=80),dict(passed=True,mean_update_s=.1,peak_reserved_bytes=110),dict(passed=True,mean_update_s=.2,peak_reserved_bytes=90)]
    assert select_configuration(rows,100) is rows[2]
    with pytest.raises(ValueError):select_configuration(rows,50)


def test_campaign_cannot_commit_settings_until_every_checkpoint_has_a_matching_benchmark():
    a,b=entry('smolvla_kr'),entry('vlash_baseline');campaign={'entries':[a,b]};reports={a['family']:report(a)}
    with pytest.raises(ValueError):controller.commit_profiles(campaign,reports)
    assert 'family_performance' not in campaign
    reports[b['family']]=report(b);new=controller.commit_profiles(campaign,reports)
    assert new['benchmark_before_training'] and len(new['family_performance'])==2
    reports[a['family']]['selected']['flow_microbatch']=16
    assert new['family_performance'][a['family']]['flow_microbatch']==4
    wrong=report(a);wrong['checkpoint_sha256']='different';assert not controller.benchmark_matches(wrong,a)


@pytest.mark.parametrize('fail_second',[False,True])
def test_controller_saves_all_benchmarks_before_any_new_training(tmp_path,monkeypatch,fail_second):
    items=[entry('smolvla_kr'),entry('vlash_baseline')];dependency=tmp_path/'previous';dependency.mkdir()
    (dependency/'final-evaluation-status.json').write_text(json.dumps({'state':'completed'}))
    campaign=tmp_path/'campaign.json';campaign.write_text(json.dumps(dict(entries=items,source_hashes={},current_queue_dependency=str(dependency))))
    (tmp_path/'downloads.json').write_text(json.dumps({'models':{e['family']:{'state':'verified'} for e in items}}))
    monkeypatch.setattr(sys,'argv',['controller','--campaign',str(campaign)])
    monkeypatch.setattr(controller.subprocess,'check_output',lambda *a,**k:'24000\n')
    calls=[]
    def run(command,**kwargs):
        calls.append(command);family=command[command.index('--family')+1];out=Path(command[command.index('--output')+1])
        item=next(e for e in items if e['family']==family)
        if any(x.endswith('profile_family_performance.py') for x in command):
            value=report(item)
            if fail_second and family==items[1]['family']:
                value.update(passed=False,state='failed');out.write_text(json.dumps(value));return SimpleNamespace(returncode=1)
            out.write_text(json.dumps(value))
        else:
            assert all((tmp_path/'benchmarks'/f'{e["family"]}.json').exists() for e in items)
            if command[command.index('--mode')+1]=='train':
                out.mkdir();(out/'status.json').write_text(json.dumps({'state':'completed'}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(controller.subprocess,'run',run);controller.main()
    value=json.loads((tmp_path/'queue.json').read_text())
    if fail_second:
        assert value['state']=='benchmark_failed'
        assert all(any(x.endswith('profile_family_performance.py') for x in command) for command in calls)
    else:
        assert value['state']=='completed'
        assert (tmp_path/'benchmark-summary.json').exists()
        assert all(any(x.endswith('profile_family_performance.py') for x in command) for command in calls[:2])
