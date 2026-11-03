"""Guard against rerunning baselines or accepting a mismatched new Coordinate checkpoint."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tracking_variants import checkpoints_from_args, validate_variant
from summarize_tracking_online_three import main


def test_single_coordinate_velocity_checkpoint_routes_only_new_model():
    assert checkpoints_from_args(SimpleNamespace(coordinate_velocity='new'))=={'coordinate_velocity':'new'}
    with pytest.raises(ValueError):checkpoints_from_args(SimpleNamespace())
    c=SimpleNamespace(enable_residual_flow=True,residual_flow_formulation='coordinate',condition_on_object_velocity=True)
    validate_variant(c,'coordinate_velocity')
    c.condition_on_object_velocity=False
    with pytest.raises(ValueError):validate_variant(c,'coordinate_velocity')
    validate_variant(c,'coordinate')
    c.condition_on_object_velocity=True;c.residual_flow_formulation='field'
    with pytest.raises(ValueError):validate_variant(c,'coordinate_velocity')


def test_five_method_report_preserves_failure_and_checks_all_pairs(tmp_path):
    models=['baseline','velocity','residual','coordinate','coordinate_velocity']
    (tmp_path/'manifest.json').write_text(json.dumps({'models':dict.fromkeys(models,'checkpoint'),'episodes':[{'episode':8}],'speed_multipliers':[1,1.5,2]}))
    for speed in (1,1.5,2):
        for model in models:
            dest=tmp_path/model/f'{speed}x'/'episode_000008';dest.mkdir(parents=True)
            (dest/'initial_state.json').write_text(json.dumps({'all_joint_position':[0]*9}))
            complete=not (model=='coordinate' and speed==2)
            row={'model':model,'episode':8,'scenario':f'{speed}x','speed_multiplier':speed,'trajectory':{'center':[0,0,0]},
                 'metrics':{'completed':complete,'success':False,'hover_rmse_m':.04,'hover_p95_m':.07,'orientation_rmse_deg':20,'position_within_2cm_fraction':.1,'fraction_above_train_max_speed':.5,'mean_inference_seconds':.25},
                 'failure_type':None if complete else 'policy_termination','error':None,'requested_seconds':10,'recorded_seconds':10 if complete else 3,
                 'videos':{'opst_cam':str(dest/'opst_cam.mp4')}}
            (dest/'metrics.json').write_text(json.dumps(row))
    main(SimpleNamespace(output=tmp_path,no_videos=True))
    report=json.loads((tmp_path/'summary.json').read_text())
    assert report['complete'] and report['pairing_valid'] and report['recorded_rollouts']==15
    assert report['policy_terminations']==1 and len(report['pairwise'])==30
    assert report['groups']['coordinate_2x']['completed']==0
    assert report['groups']['coordinate_velocity_2x']['completed']==1
