import hashlib,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workstation'))
from smolvla_orange_eval import build_command


def setup(tmp_path):
    campaign=tmp_path/'campaign.json';campaign.write_text(json.dumps(dict(entries=[dict(family='smolvla_baseline',velocity_conditioning=None)],
        training_recipe=dict(instruction_mode='recorded',prompt='Pick up the rolling orange and place it in the box.'))))
    run=tmp_path/'runs/smolvla_baseline';run.mkdir(parents=True)
    (run/'status.json').write_text(json.dumps(dict(state='completed')))
    for name,epoch,kind in [('best.pt',14,'ema'),('latest.pt',30,'raw')]:
        (run/name).write_bytes(name.encode())
        pointer=dict(path=str(run/name),epoch=epoch,weights=kind,completed_epoch=True)
        (run/('best_checkpoint.json' if name=='best.pt' else 'latest_checkpoint.json')).write_text(json.dumps(pointer))
    return campaign,run


def test_best_and_last_select_their_own_completed_weights(tmp_path):
    campaign,run=setup(tmp_path)
    for kind,epoch,weights in [('best',14,'ema'),('last',30,'raw')]:
        command,record=build_command(campaign,kind)
        assert record['epoch']==epoch and record['weights']==weights
        assert command[command.index('--checkpoint')+1]==kind
        assert command[command.index('--velocity-innovation')+1]=='off'
        assert command[command.index('--rtc')+1]=='guided'
        assert '--execute' not in command and '--record-inputs' in command
        assert command[command.index('--expected-checkpoint-sha256')+1]==hashlib.sha256(Path(record['source']).read_bytes()).hexdigest()
    command,_=build_command(campaign,'last',True,3,tmp_path/'start.json')
    assert '--execute' in command and command[command.index('--max-trials')+1]=='3'
    assert command[command.index('--start-pose')+1]==str(tmp_path/'start.json')
    moving,_=build_command(campaign,'best',object_motion='moving')
    assert moving[moving.index('--object-motion')+1]=='moving'
    off,_=build_command(campaign,'best',rtc='off')
    assert off[off.index('--rtc')+1]=='off'
    with pytest.raises(ValueError,match='RTC'):build_command(campaign,'best',rtc_execution_horizon=0)


def test_training_and_partial_recovery_are_not_launchable(tmp_path):
    campaign,run=setup(tmp_path);(run/'status.json').write_text(json.dumps(dict(state='training')))
    with pytest.raises(ValueError,match='completed training'):build_command(campaign,'best')
    (run/'status.json').write_text(json.dumps(dict(state='completed')))
    p=run/'latest_checkpoint.json';pointer=json.loads(p.read_text());pointer['completed_epoch']=False;p.write_text(json.dumps(pointer))
    with pytest.raises(ValueError,match='Partial recovery'):build_command(campaign,'last')


def test_user_stopped_run_requires_verified_frozen_checkpoints(tmp_path):
    campaign,run=setup(tmp_path)
    (run/'status.json').write_text(json.dumps(dict(state='stopped')))
    with pytest.raises(ValueError,match='verified evaluation release'):build_command(campaign,'best')
    records={}
    for selection,pointer_file in [('best','best_checkpoint.json'),('last','latest_checkpoint.json')]:
        pointer=json.loads((run/pointer_file).read_text());path=Path(pointer['path'])
        records[selection]=dict(path=str(path),epoch=pointer['epoch'],sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (run/'evaluation_release.json').write_text(json.dumps(dict(user_requested=True,training_process_exited=True,snapshots_verified=True,checkpoints=records)))
    for selection in ['best','last']:assert build_command(campaign,selection)[1]['epoch']==records[selection]['epoch']
    (run/'best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):build_command(campaign,'best')


def test_mask_modes_trial_limit_and_timeout_are_forwarded(tmp_path):
    campaign,_=setup(tmp_path)
    command,_=build_command(campaign,'best',max_trials=3,trial_seconds=30)
    assert '--auto-mask' not in command and '--new-mask' not in command
    assert command[command.index('--max-trials')+1]=='3'
    assert float(command[command.index('--trial-seconds')+1])==30.
    manual,_=build_command(campaign,'last',max_trials=0,trial_seconds=45,mask_mode='manual')
    assert '--new-mask' not in manual and '--auto-mask' not in manual
    assert manual[manual.index('--max-trials')+1]=='0'
    reuse,_=build_command(campaign,'last',mask_mode='reuse')
    assert '--new-mask' not in reuse and '--auto-mask' not in reuse
    with pytest.raises(ValueError,match='trial-seconds'):build_command(campaign,'best',trial_seconds=float('nan'))
    settings=json.loads(campaign.read_text());settings['entries'][0]['velocity_conditioning']='current_xyz_v7';campaign.write_text(json.dumps(settings))
    tracked,_=build_command(campaign,'best',mask_mode='auto')
    assert '--auto-mask' in tracked
    tracked_manual,_=build_command(campaign,'best',mask_mode='manual')
    assert '--new-mask' in tracked_manual
