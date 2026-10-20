import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from scipy.spatial.transform import Rotation
from tracking_online_common import PosePlan, pose_command, summarize_rollout


def test_pose_plan_preserves_label_horizon_and_linear_velocity():
    chunk=np.zeros((50,7));chunk[:,0]=.2+np.arange(50)*.04
    plan=PosePlan(3.,np.zeros(6),chunk)
    xyz,_,v,_=plan.sample(3.1)
    np.testing.assert_allclose(xyz,[.1,0,0]);np.testing.assert_allclose(v,[1,0,0])
    np.testing.assert_allclose(plan.sample(3.2)[0],chunk[0,:3])
    np.testing.assert_allclose(plan.sample(3.24)[0],chunk[1,:3])


def test_pose_plan_uses_short_rotation_across_euler_wrap():
    anchor=np.zeros(6);anchor[5]=np.deg2rad(359)
    chunk=np.zeros((50,7));chunk[:,5]=np.deg2rad(1)
    _,r,_,w=PosePlan(0,anchor,chunk).sample(.1)
    assert np.linalg.norm(r.as_rotvec())<1e-8
    np.testing.assert_allclose(w,[0,0,np.deg2rad(10)],atol=1e-10)


def test_plan_holds_after_last_label():
    chunk=np.ones((50,7))
    xyz,_,v,w=PosePlan(0,np.zeros(6),chunk).sample(100)
    np.testing.assert_allclose(xyz,[1,1,1]);np.testing.assert_allclose(v,0);np.testing.assert_allclose(w,0)


def test_low_level_servo_handles_tool_offset_and_limits():
    cfg=dict(kp=4,kd=.5,orientation_kp=8,max_linear_speed=.3,max_angular_speed=1.57,
             action_offset=[0,0,.107],observation_offset=[0,0,.1034],damping=.03,joint_margin=.02)
    state=dict(ee_position=np.zeros(3),ee_quaternion=np.array([1.,0,0,0]),
               ee_velocity=np.zeros(3),joint_position=np.zeros(7))
    limits=np.tile([-2,2],(7,1));jac=np.eye(6,7)
    command=pose_command(state,(np.array([10.,0,0]),Rotation.identity(),np.zeros(3),np.zeros(3)),jac,limits,np.ones(7),cfg,.005)
    assert command['saturated']
    assert np.max(np.abs(command['joint_velocity_target']))<=1
    assert np.max(np.abs(command['joint_position_target']))<=.005


def test_failed_episode_cannot_count_as_success():
    quality=dict(transition_seconds=.75,rmse=.02,max_error=.05,orientation_degrees=5,contact_force=.5)
    row=dict(hover_error_m=.001,orientation_error_deg=1,settled=True,object_speed_mps=.2,
             contact_force_n=0,joint_limit_violation=False,workspace_violation=False,saturated=False)
    result=summarize_rollout([row],quality,.15,False)
    assert not result['success'] and not result['completed']
    assert result['fraction_above_train_max_speed']==1
    assert summarize_rollout([row],quality,.15,True)['success']


def test_time_scaled_object_motion_preserves_position_and_scales_derivatives():
    collector=Path(__file__).resolve().parents[2]/'Data_Collection_Dynamic_Objects'
    sys.path.insert(0,str(collector))
    from dom_tracking.trajectory import Trajectory
    base=Trajectory(np.array([.45,0,.16]),np.array([.04,.1,.04]),np.array([1.,.8,1.2]),np.array([.2,.4,.1]))
    fast=Trajectory(base.center,base.amplitude,base.omega*2,base.phase)
    p,v,a=base.sample(.6);fp,fv,fa=fast.sample(.3)
    np.testing.assert_allclose(fp,p);np.testing.assert_allclose(fv,2*v);np.testing.assert_allclose(fa,4*a)


def test_summary_counts_missing_rollouts_without_inventing_results(tmp_path):
    import json
    from types import SimpleNamespace
    from summarize_tracking_online import main
    manifest={'episodes':[{'episode':8}],'speed_multipliers':[1,1.5,2]}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    main(SimpleNamespace(output=tmp_path,no_videos=True))
    result=json.loads((tmp_path/'summary.json').read_text())
    assert result['expected_rollouts']==6 and result['recorded_rollouts']==0
    assert not result['complete'] and result['paired']=={}
    assert result['groups']['baseline_1x']['mean_hover_rmse_m'] is None


def test_joint_guard_reports_policy_termination():
    from tracking_online_common import TrackingTermination
    import pytest
    cfg=dict(kp=4,kd=.5,orientation_kp=8,max_linear_speed=.3,max_angular_speed=1.57,
             action_offset=[0,0,.107],observation_offset=[0,0,.1034],damping=.03,joint_margin=.02)
    state=dict(ee_position=np.zeros(3),ee_quaternion=np.array([1.,0,0,0]),
               ee_velocity=np.zeros(3),joint_position=np.full(7,3.))
    with pytest.raises(TrackingTermination):
        pose_command(state,(np.zeros(3),Rotation.identity(),np.zeros(3),np.zeros(3)),
                     np.eye(6,7),np.tile([-2,2],(7,1)),np.ones(7),cfg,.005)


def test_three_model_report_keeps_failure_counts_and_excludes_truncated_rmse(tmp_path):
    import json
    from types import SimpleNamespace
    from summarize_tracking_online_three import main
    models={'baseline':'a','residual':'b','coordinate':'c'}
    (tmp_path/'manifest.json').write_text(json.dumps({'models':models,'episodes':[{'episode':8}],'speed_multipliers':[1.]}))
    for i,model in enumerate(models):
        dest=tmp_path/model/'iid'/'episode_000008';dest.mkdir(parents=True)
        (dest/'initial_state.json').write_text(json.dumps({'all_joint_position':[0]*9}))
        row={'model':model,'scenario':'iid','speed_multiplier':1.,'episode':8,'trajectory':{'same':True},
             'failure_type':'policy_termination' if i==0 else None,'error':'joint limit' if i==0 else None,
             'requested_seconds':10,'recorded_seconds':1 if i==0 else 10,
             'videos':{'opst_cam':str(dest/'opst_cam.mp4')},
             'metrics':{'completed':i>0,'success':i>0,'hover_rmse_m':.001 if i==0 else .04/i}}
        (dest/'metrics.json').write_text(json.dumps(row))
    main(SimpleNamespace(output=tmp_path,no_videos=True))
    s=json.loads((tmp_path/'summary.json').read_text())
    assert s['complete'] and not s['all_rollouts_completed'] and s['policy_terminations']==1
    assert s['groups']['baseline_1x']['completed_mean_hover_rmse_m'] is None
    assert s['groups']['baseline_1x']['success_rate_over_expected']==0
    assert s['pairwise']['coordinate_minus_baseline_1x']['completed_matched_pairs']==0
    assert s['pairwise']['coordinate_minus_residual_1x']['right_minus_left_hover_rmse_m']==-.02


def test_comparison_video_marks_terminated_panel(tmp_path):
    import av
    from summarize_tracking_online_three import comparison_video
    models=['baseline','residual','coordinate'];records={}
    for model,n in zip(models,[1,3,3]):
        path=tmp_path/f'{model}.mp4'
        with av.open(str(path),'w') as out:
            stream=out.add_stream('libx264',rate=25);stream.width=480;stream.height=360;stream.pix_fmt='yuv420p'
            for _ in range(n):
                frame=av.VideoFrame.from_ndarray(np.full((360,480,3),180,dtype=np.uint8),format='rgb24')
                for packet in stream.encode(frame):out.mux(packet)
            for packet in stream.encode():out.mux(packet)
        records[model]={'videos':{'opst_cam':str(path)},'video_frames':{'opst_cam':n},'metrics':{'completed':n==3}}
    destination=tmp_path/'comparison.mp4'
    assert comparison_video(records,models,destination,25,3)==3
    with av.open(str(destination)) as container:
        frames=[f.to_ndarray(format='rgb24') for f in container.decode(video=0)]
    from tracking_visuals import HEIGHT, HEADER
    assert len(frames)==3 and frames[-1].shape==(HEIGHT,1440,3)
    assert frames[-1][HEADER+5,0].max()<40  # terminated baseline panel
    assert frames[-1][HEADER+5,500].min()>150  # continuing residual panel
