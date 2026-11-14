import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workstation'))
from prepare_closed_loop_pair import pair_status
from real_robot.evaluate import read_run_contract,prepare_checkpoint


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data))


def setup_pair(root):
    write(root/'dom-contract.json',{'observation_history':[-2,0]})
    for family in ['domv','krv100']:
        run=root/'dom-runs'/family;checkpoint=run/'epoch-003'
        write(run/'status.json',{'state':'completed','epoch':5})
        write(run/'best_checkpoint.json',{'path':str(checkpoint),'epoch':3})
        write(checkpoint/'provenance.json',{'completed_epoch':True,'epoch':3,'validation_loss':.1})
        write(checkpoint/'config.json',{'residual_phase_gate_mode':'constant',
            'condition_on_object_velocity':True,'enable_residual_flow':family=='krv100','chunk_size':20})
        (checkpoint/'model.safetensors').touch()


def test_pair_requires_both_training_runs_complete(tmp_path):
    setup_pair(tmp_path)
    write(tmp_path/'dom-runs/domv/status.json',{'state':'epoch_complete','epoch':4})
    result=pair_status(tmp_path)
    assert not result['ready'] and 'commands' not in result
    write(tmp_path/'dom-runs/domv/status.json',{'state':'completed','epoch':5})
    result=pair_status(tmp_path,'marker')
    assert result['ready']
    assert all('--velocity-innovation on' in cmd and '--task marker' in cmd for cmd in result['commands'].values())
    assert result['models']['krv100']['selected_epoch']==3


def test_continuation_contract_and_selected_native_gating_preserved(tmp_path):
    setup_pair(tmp_path)
    assert read_run_contract(tmp_path/'dom-runs')['observation_history']==[-2,0]
    for family in ['domv','krv100']:
        checkpoint,_=prepare_checkpoint(tmp_path/'dom-runs',family,'best')
        cfg=json.loads((checkpoint/'config.json').read_text())
        assert cfg['residual_phase_gate_mode']=='constant'
        assert cfg['enable_residual_flow']==(family=='krv100')
        assert cfg['condition_on_object_velocity']
        assert cfg['delta_timestamps']['observation']==[-2,0]
    bad=tmp_path/'dom-runs/krv100/epoch-003/config.json'
    cfg=json.loads(bad.read_text());cfg['residual_phase_gate_mode']='causal_rules';write(bad,cfg)
    with pytest.raises(ValueError,match='constant gate'):pair_status(tmp_path)


def test_explicit_epoch_does_not_repoint_best_or_last(tmp_path):
    setup_pair(tmp_path)
    checkpoint,record=prepare_checkpoint(tmp_path/'dom-runs','krv100','epoch-003')
    assert record['epoch']==3 and checkpoint.name=='krv100-epoch-003'
    assert (checkpoint/'model.safetensors').resolve()==tmp_path/'dom-runs/krv100/epoch-003/model.safetensors'
    from real_robot.evaluate import checkpoint_selection
    import argparse
    assert checkpoint_selection('epoch-010')=='epoch-010'
    with pytest.raises(argparse.ArgumentTypeError):checkpoint_selection('../epoch-003')
