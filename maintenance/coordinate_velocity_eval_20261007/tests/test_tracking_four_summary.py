"""A four-method report must retain failed trials and compare all six pairs."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_tracking_online_three import main
from tracking_variants import VARIANT_LABELS
from tracking_visuals import PanelRenderer

def test_four_model_report_and_failure_counts(tmp_path):
    models=['baseline','velocity','residual','coordinate']
    (tmp_path/'manifest.json').write_text(json.dumps({'models':dict.fromkeys(models,'checkpoint'),'episodes':[{'episode':8}],'speed_multipliers':[1,1.5,2]}))
    for speed in (1,1.5,2):
      for model in models:
        dest=tmp_path/model/f'{speed}x'/'episode_000008';dest.mkdir(parents=True)
        (dest/'initial_state.json').write_text(json.dumps({'all_joint_position':[0]*9}))
        complete=not (model=='velocity' and speed==2)
        row={'model':model,'episode':8,'scenario':f'{speed}x','speed_multiplier':speed,'trajectory':{'center':[0,0,0]},
          'metrics':{'completed':complete,'success':False,'hover_rmse_m':.04,'hover_p95_m':.07,'orientation_rmse_deg':20,'position_within_2cm_fraction':.1,'fraction_above_train_max_speed':.5,'mean_inference_seconds':.25},
          'failure_type':None if complete else 'policy_termination','error':None,'requested_seconds':10,'recorded_seconds':10 if complete else 3,
          'videos':{'opst_cam':str(dest/'opst_cam.mp4')}}
        (dest/'metrics.json').write_text(json.dumps(row))
        assert PanelRenderer(row).render(None,100).shape==(512,480,3)
    main(SimpleNamespace(output=tmp_path,no_videos=True))
    report=json.loads((tmp_path/'summary.json').read_text())
    assert report['complete'] and report['pairing_valid'] and report['recorded_rollouts']==12
    assert report['policy_terminations']==1 and len(report['pairwise'])==18
    assert report['groups']['velocity_2x']['completed']==0
    assert report['groups']['velocity_2x']['completed_mean_hover_rmse_m'] is None
    assert (tmp_path/'tracking_comparison.png').is_file()
