"""Exercise current-velocity conditioning with Coordinate residualization enabled."""
import ast,json,os,sys
from pathlib import Path
from types import SimpleNamespace
import pytest,torch,yaml
from test_tracking_velocity_conditioning import ns,policy

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from tracking_variants import validate_variant

def test_coordinate_neural_velocity_and_analytic_fields_share_current_signal():
    p=policy();p.config.enable_residual_flow=True;p.config.residual_flow_formulation='coordinate'
    env=torch.zeros(2,2,9);env[:,0,6:9]=-999;env[:,-1,6:9]=torch.tensor([3.,6.,11.])
    batch={'observation.state':torch.ones(2,2,6),'observation.environment_state':env}
    batch.update(ns['extract_residual_batch_fields'](p,batch))
    state=ns['prepare_state'](p,batch)
    torch.testing.assert_close(state[:,6:9],torch.ones(2,3))
    torch.testing.assert_close(batch[p.config.residual_velocity_key][:,-1],env[:,-1,6:9])
    projection=torch.nn.Linear(32,12);projection(state).square().sum().backward()
    assert projection.weight.grad[:,6:9].abs().sum()>0

def test_new_coordinate_guard_accepts_both_branches_and_rejects_old_when_required():
    c=SimpleNamespace(enable_residual_flow=True,residual_flow_formulation='coordinate',condition_on_object_velocity=True)
    validate_variant(c,'coordinate',require_neural_velocity=True)
    c.condition_on_object_velocity=False
    validate_variant(c,'coordinate')
    with pytest.raises(ValueError):validate_variant(c,'coordinate',require_neural_velocity=True)

def test_coordinate_uses_baselines_velocity_normalization_and_recipe():
    c=yaml.safe_load((ROOT/'configs/tracking_dom75_full20_coordinate_velocity.yaml').read_text())
    v=yaml.safe_load((ROOT/'configs/dynamicvla_dom_tracking_velocity75.yaml').read_text())
    assert c['TRAIN']==v['TRAIN'] and c['DATASET']==v['DATASET']
    assert c['POLICY']['CONDITION_ON_OBJECT_VELOCITY'] and c['POLICY']['ENABLE_RESIDUAL_FLOW']
    assert c['POLICY']['RESIDUAL_FLOW_FORMULATION']=='coordinate'
    for k in ('VELOCITY_INPUT_MEAN','VELOCITY_INPUT_STD'):assert c['POLICY'][k]==v['POLICY'][k]

def test_validation_selection_record_matches_saved_weights(tmp_path):
    tree=ast.parse((ROOT/'core/train.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_save_best_eval_metadata')
    namespace={'os':os,'json':json}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'train.py','exec'),namespace)
    namespace['_save_best_eval_metadata'](str(tmp_path),'model.best_validation_action_mse.safetensors','validation','action_mse',.2,17)
    d=json.loads((tmp_path/'best_eval.json').read_text())
    assert d=={'weights':'model.best_validation_action_mse.safetensors','split':'validation','metric':'action_mse','value':.2,'epoch':17}
    assert not (tmp_path/'best_eval.json.tmp').exists()
