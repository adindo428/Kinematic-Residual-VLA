import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import local
import pytest


def args(tmp_path):
    return argparse.Namespace(stage=tmp_path/'stage with spaces',root=tmp_path/'results',assets=tmp_path/'assets',
        isaaclab=tmp_path/'Isaac Lab/isaaclab.sh',policy_python=tmp_path/'policy env/bin/python',
        port=26000,dry_run=True,resume_verified=False)


def test_portable_launch_preserves_streaming_and_separate_environments(tmp_path):
    a=args(tmp_path)
    server,client=local.commands(a,tmp_path/'list.txt',tmp_path/'run',20)
    assert server[0]==str(a.isaaclab) and client[0]==str(a.policy_python)
    assert '-s' in client and '-d' in client and '--headless' in server and '--enable_cameras' in server
    assert server[server.index('--num_envs')+1]=='1'
    assert client[client.index('-p')+1]==str(a.stage/'checkpoint')
    assert '--img_port' in client and '--act_port' in server
    assert not any('/scratch/aparame' in item for item in server+client)


def test_dry_run_needs_no_gpu_and_creates_no_run_directory(tmp_path,capsys):
    a=args(tmp_path);a.root.mkdir()
    (a.root/'protocol.json').write_text(json.dumps({'checkpoint':str(a.stage/'checkpoint'),'dimension_order':['CR']}))
    local.run_one(a,'pilot')
    assert not (a.root/'pilot').exists()
    assert json.loads(capsys.readouterr().out)['mode']=='pilot'


def test_existing_pilot_cannot_be_overwritten(tmp_path):
    a=args(tmp_path);a.dry_run=False;a.root.mkdir();(a.root/'pilot').mkdir()
    sentinel=a.root/'pilot/sentinel';sentinel.write_text('previous evidence')
    (a.root/'protocol.json').write_text(json.dumps({'checkpoint':str(a.stage/'checkpoint'),'dimension_order':['CR']}))
    with pytest.raises(FileExistsError):local.run_one(a,'pilot')
    assert sentinel.read_text()=='previous evidence'


def test_separate_gpu_selection_preserves_ssd_cache_paths(tmp_path,monkeypatch):
    a=args(tmp_path);a.sim_gpu=0;a.policy_gpu=1
    monkeypatch.setenv('HF_HOME',str(tmp_path/'ssd-cache'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2')
    original,server,client=local.process_environments(a,tmp_path/'run')
    assert server['CUDA_VISIBLE_DEVICES']=='0'
    assert client['CUDA_VISIBLE_DEVICES']=='1'
    assert original['CUDA_VISIBLE_DEVICES']=='2'
    assert server['HF_HOME']==client['HF_HOME']==str(tmp_path/'ssd-cache')
    assert server['DOM_OFFICIAL_SOURCE']==client['DOM_OFFICIAL_SOURCE']==str(a.stage/'source')


def test_portable_kit_settings_reach_simulator_only(tmp_path):
    a=args(tmp_path);a.kit_args='--portable --portable-root=/ssd/kit'
    server,client=local.commands(a,tmp_path/'list.txt',tmp_path/'run',20)
    assert server[server.index('--kit_args')+1]==a.kit_args
    assert '--kit_args' not in client
