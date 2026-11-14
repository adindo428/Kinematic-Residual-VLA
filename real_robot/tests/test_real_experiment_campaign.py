import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workstation'))
from real_robot.family_live import current_family_inputs,select_family_checkpoint,family_batch
from real_robot.evaluate import worker_command
from real_experiment_campaign import schedule


def payload():
    return dict(front=np.zeros((2,480,640,3),dtype=np.uint8),wrist=np.zeros((2,480,640,3),dtype=np.uint8),
        state=np.arange(12,dtype=np.float32).reshape(2,6),environment=np.arange(18,dtype=np.float32).reshape(2,9),
        required=np.array([True,True]),valid=np.array([True,True]))


def test_native_family_consumes_current_velocity_and_state_not_old_observation_or_future():
    data=payload();data['future_velocity']=np.full((20,3),999.)
    result=current_family_inputs(data)
    np.testing.assert_array_equal(result['state'],data['state'][-1]);np.testing.assert_array_equal(result['velocity'],data['environment'][-1,6:9])
    data['required'][-1]=False;data['valid'][-1]=False
    assert not current_family_inputs(data)['velocity'].any()
    data['required'][-1]=True
    with pytest.raises(ValueError,match='invalid'):current_family_inputs(data)


def test_native_live_batch_matches_training_state_shape_rgb_layout_and_neutral_placeholder_targets():
    data=payload();data['front'][-1,:,:,0]=255
    spec=dict(schema='krvla_image_preprocessing_v1',source_size_wh=[640,480],target_size_wh=[384,384],padding='left_top',pad_value=0,
              front_crop_xywh=[0,0,640,480])
    batch=family_batch(data,{'instruction':'pick the ball'},spec,device='cpu')
    assert batch['observation.state'].shape==(1,6) and batch['action'].shape==(1,20,7)
    np.testing.assert_array_equal(batch['reference_state'][0],data['state'][-1])
    assert batch['observation.images.opst_cam'].shape==(1,3,384,384)
    assert batch['observation.images.opst_cam'][0,0].sum()==0 and batch['observation.images.opst_cam'][0,2].sum()>0
    assert batch['offset'].item()==0 and not batch['action_is_pad'].any()


def test_incomplete_training_and_partial_recovery_never_enter_real_trials(tmp_path):
    campaign=tmp_path/'campaign.json';campaign.write_text(json.dumps({'entries':[{'family':'smolvla_kr'}]}))
    run=tmp_path/'runs/smolvla_kr';run.mkdir(parents=True)
    (run/'status.json').write_text(json.dumps({'state':'training'}))
    with pytest.raises(ValueError,match='completed'):select_family_checkpoint(campaign,'smolvla_kr')
    (run/'status.json').write_text(json.dumps({'state':'completed'}))
    checkpoint=run/'recovery.pt';checkpoint.write_bytes(b'fixture')
    (run/'latest_checkpoint.json').write_text(json.dumps({'path':str(checkpoint),'completed_epoch':False,'epoch':1}))
    with pytest.raises(ValueError,match='Partial'):select_family_checkpoint(campaign,'smolvla_kr','last')
    checkpoint=run/'best.pt';checkpoint.write_bytes(b'fixture')
    (run/'best_checkpoint.json').write_text(json.dumps({'path':str(checkpoint),'epoch':1,'weights':'ema'}))
    selected,record,_=select_family_checkpoint(campaign,'smolvla_kr')
    assert selected==checkpoint and record['weights']=='ema' and record['neural_observation_offsets']==[0]


def test_family_worker_isolated_from_dom_runtime_and_receives_frozen_hash(tmp_path):
    smol=worker_command('smolvla_kr',tmp_path/'best.pt',tmp_path/'campaign.json',tmp_path,tmp_path/'spec.json','sha')
    dom=worker_command('domv',tmp_path/'epoch-015',tmp_path/'campaign.json',tmp_path,tmp_path/'spec.json','sha')
    assert smol[0].endswith('family-python.sh') and '--expected-sha256' in smol and 'sha' in smol
    assert dom[0].endswith('finetune-python.sh') and any(x.endswith('real_policy_worker.py') for x in dom)


def test_plan_covers_every_model_and_keeps_pilots_out_of_scored_repeats():
    models=['dom','domv','krv100','smolvla_baseline','smolvla_kr'];tasks=['striped_ball','marker','orange'];conditions=['iid_slow','iid_medium','speed_ood']
    rows=schedule(models,tasks,conditions)
    assert rows==schedule(models,tasks,conditions)
    assert sum(x['trials'] for x in rows if x['phase']=='pilot')==45
    assert sum(x['trials'] for x in rows if x['phase']=='scored')==900
    for task in tasks:
        for condition in conditions:
            block=[x for x in rows if x['task']==task and x['condition']==condition and x['phase']=='scored']
            assert {x['model'] for x in block}==set(models)


def test_rollout_summary_separates_pilots_preview_and_success_time(tmp_path):
    from types import SimpleNamespace
    from real_experiment_campaign import summarize
    requests=tmp_path/'session-requests';requests.mkdir()
    for name,phase,execute,outcomes in [('pilot','pilot',True,[('success','operator labelled success')]),
        ('scored','scored',True,[('success','operator labelled success'),('failure','timeout'),('failure','operator takeover/quit')]),
        ('preview','scored',False,[('success','operator labelled success')])]:
        output=tmp_path/name/'controller';output.mkdir(parents=True)
        (requests/(name+'.json')).write_text(json.dumps(dict(phase=phase,execute=execute,model='dom',task='marker',condition='iid_slow',output=str(output.parent))))
        for i,(outcome,reason) in enumerate(outcomes):
            (output/f'trial-{i:03d}.json').write_text(json.dumps(dict(outcome=outcome,reason=reason,start_monotonic_s=10,end_monotonic_s=15,holding_confirmed=outcome=='success')))
    summarize(SimpleNamespace(root=tmp_path));groups=json.loads((tmp_path/'rollout-summary.json').read_text())['groups']
    assert len(groups)==2
    scored=next(x for x in groups if x['phase']=='scored')
    assert scored['trials']==3 and scored['successes']==1 and scored['mean_successful_completion_s']==5
    assert scored['timeouts']==1 and scored['takeovers']==1 and scored['holding_confirmed']==1


def test_all_trained_objects_comes_from_eligible_training_anchors_not_task_catalog_or_validation(tmp_path):
    from real_experiment_campaign import trained_task_specifications
    dataset=tmp_path/'dataset';(dataset/'meta').mkdir(parents=True)
    (dataset/'meta/splits.json').write_text(json.dumps({'train':[0,1,2],'validation':[3]}))
    (dataset/'meta/quality_policy.json').write_text(json.dumps({'usable_frames':{'0':[0,1],'1':[0],'2':[],'3':[0]}}))
    rows=[dict(episode_index=i,tasks=[json.dumps({'instruction':instruction,'task':'place'})])
          for i,instruction in enumerate(['pick orange','pick marker','pick bottle','pick ball'])]
    (dataset/'meta/episodes.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    campaign=tmp_path/'campaign.json';campaign.write_text(json.dumps({'training_recipe':{'local_verified_dataset':str(dataset)}}))
    collection=tmp_path/'collection.json';collection.write_text(json.dumps({'tasks':[dict(id=name,instruction='pick '+name)
        for name in ['orange','marker','bottle','ball','tape']]}))
    actual=trained_task_specifications(campaign,collection)
    assert set(actual)=={'orange','marker'}
    assert actual['orange']['training_episodes']==1 and actual['orange']['training_anchors']==2


def test_single_static_pilot_retains_execution_guards_and_valid_camera_duration(tmp_path,monkeypatch):
    import real_experiment_campaign as campaign
    from types import SimpleNamespace
    checkpoint=tmp_path/'best.pt';checkpoint.write_bytes(b'weights')
    spec=tmp_path/'image.json';spec.write_text('{}')
    plan=dict(state='ready_for_supervised_pilot',tasks=['marker'],conditions=[{'id':'static'}],
        selected={'smolvla_kr':dict(checkpoint=str(checkpoint),checkpoint_sha256=campaign.sha(checkpoint),
            image_spec=str(spec),image_spec_sha256=campaign.sha(spec),config_sha256=None)},
        settings=dict(motion_profile='hardware',joint_governor='on',ik_method='bounded_dls',geometry_profile='grasp',
            pose_smoothing_ms=80,trial_seconds=30,start_position_tolerance_mm=10,start_joint_tolerance_deg=2,
            object_prediction_max_gap_s=2.,object_prediction_max_displacement_m=.05,object_occlusion_model='contact_stop_assumption'))
    (tmp_path/'plan.json').write_text(json.dumps(plan));calls=[]
    monkeypatch.setattr(campaign.subprocess,'run',lambda cmd,**kwargs:calls.append(cmd) or SimpleNamespace(returncode=0))
    campaign.launch(SimpleNamespace(root=tmp_path,model='smolvla_kr',task='marker',phase='pilot',condition='static',max_trials=1,execute=True,new_mask=True))
    cmd=calls[0]
    assert cmd[cmd.index('--max-trials')+1]=='1' and cmd[cmd.index('--trial-seconds')+1]=='30'
    assert float(cmd[cmd.index('--seconds')+1])+120<=1800
    assert cmd[cmd.index('--joint-governor')+1]=='on' and '--record-inputs' in cmd and '--execute' in cmd
    request=json.loads(next((tmp_path/'session-requests').glob('*.json')).read_text())
    assert request['condition']=='static' and request['planned_trials']==1
    assert request['tracking_prediction']['model']=='contact_stop_assumption'
    assert cmd[cmd.index('--object-prediction-max-gap-s')+1]=='2.0'
    assert cmd[cmd.index('--object-occlusion-model')+1]=='contact_stop_assumption'


def test_rollout_results_keep_occlusion_recipes_separate(tmp_path):
    from types import SimpleNamespace
    from real_experiment_campaign import summarize
    requests=tmp_path/'session-requests';requests.mkdir()
    for name,prediction in [('old',dict(model='legacy_coasting_only',max_gap_s=.08)),
                             ('new',dict(model='constant_velocity',max_gap_s=.5,max_displacement_m=.05,acceleration_std_mps2=.1))]:
        controller=tmp_path/name/'controller';controller.mkdir(parents=True)
        (requests/(name+'.json')).write_text(json.dumps(dict(phase='pilot',execute=True,model='smolvla_kr',task='marker',condition='static',
            tracking_prediction=prediction,output=str(controller.parent))))
        (controller/'trial-001.json').write_text(json.dumps(dict(outcome='failure',reason='timeout',start_monotonic_s=10,end_monotonic_s=40)))
    summarize(SimpleNamespace(root=tmp_path));groups=json.loads((tmp_path/'rollout-summary.json').read_text())['groups']
    assert len(groups)==2 and all(x['trials']==1 for x in groups)
    assert {x['tracking_prediction']['max_gap_s'] for x in groups}=={.08,.5}


def test_plain_smolvla_ignores_absent_or_invalid_object_fields_but_kr_requires_them():
    from real_robot.family_live import family_object_inputs_required
    from real_robot.evaluate import policy_object_inputs_required
    assert not family_object_inputs_required('smolvla_baseline')
    assert family_object_inputs_required('smolvla_kr')
    assert family_object_inputs_required('smolvla_baseline','current_xyz_v7')
    assert not policy_object_inputs_required('smolvla_baseline',Path('unused.pt'))
    assert policy_object_inputs_required('smolvla_kr',Path('unused.pt'))
    data=payload();data.pop('environment');data.pop('required');data.pop('valid')
    spec=dict(schema='krvla_image_preprocessing_v1',source_size_wh=[640,480],target_size_wh=[384,384],padding='left_top',pad_value=0,front_crop_xywh=[0,0,640,480])
    batch=family_batch(data,{'instruction':'original prompt'},spec,device='cpu',object_inputs_required=False)
    assert 'velocity' not in batch and 'velocity_required' not in batch
    np.testing.assert_array_equal(batch['reference_state'][0],data['state'][-1])
    data.update(environment=np.full((2,9),np.nan),required=np.ones(2,bool),valid=np.zeros(2,bool))
    same=family_batch(data,{'instruction':'original prompt'},spec,device='cpu',object_inputs_required=False)
    for key in batch:
        if hasattr(batch[key],'numpy'):np.testing.assert_array_equal(batch[key],same[key])
    with pytest.raises(ValueError):family_batch(data,{'instruction':'original prompt'},spec,device='cpu',object_inputs_required=True)
