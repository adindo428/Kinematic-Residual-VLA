import json
from types import SimpleNamespace
import numpy as np
import pytest
from real_robot.replay_actions import ReplayPipeline,error_metrics,transform,replay_frame
from real_robot.dom_dataset import pose6,sha256
from real_robot.vx300s_kinematics import forward

Q=np.array([0.,-.2,.4,0.,1.,0.]);TCP=[.16,0,0]


def test_replay_clock_keeps_every_frame_under_submillisecond_jitter():
    period=.04
    elapsed=np.arange(169)*period+np.r_[0,np.tile([-.0003,.0004],84)]
    assert [replay_frame(t,period)for t in elapsed]==list(range(169))
    # A genuine missed control cycle remains a skipped source frame.
    assert replay_frame(5*period+.001,period)==5
    for t,p in [(-.001,period),(0,0),(float('nan'),period)]:
        with pytest.raises(ValueError):replay_frame(t,p)


def test_absolute_rotation_comparison_handles_euler_wrap():
    a=np.array([[.3,0,.1,.1,.2,2*np.pi-.01]])
    b=a.copy();b[:,5]-=2*np.pi
    assert error_metrics(a,b)['orientation_p95_deg']<1e-10
    assert error_metrics(a,b)['position_rmse_mm']==0
    with pytest.raises(ValueError):transform([0,0,0,0,0,float('nan')])


def test_direct_and_full_pipeline_use_raw_absolute_pose_without_a_vla():
    q=Q.copy();p=ReplayPipeline(q,q,[180]*6,TCP)
    target=np.r_[pose6(forward(Q+.02,TCP)),.5]
    for _ in range(160):
        command,stages=p.propose(target,q,.04);p.accept(command);q=command.copy()
    np.testing.assert_allclose(forward(q,TCP),transform(target[:6]),atol=2e-5)
    assert stages['ik']['tcp_speed_ceiling_mps'] is None
    assert stages['ik']['angular_speed_ceiling_deg_s'] is None
    np.testing.assert_allclose(stages['target_pose6'][:3],target[:3])


def test_five_episode_device_loop_matches_starts_records_feedback_and_returns_home(monkeypatch,tmp_path):
    from real_robot import replay_control as m
    batch=tmp_path/'batch';batch.mkdir();items=[]
    for ep in range(5):
        n=20;qs=np.array([Q+np.sin(i/(n-1)*np.pi)*.01 for i in range(n)])
        poses=np.array([pose6(forward(q,TCP))for q in qs]);actions=np.column_stack([poses,np.full(n,2*(2800-2475)/(3116-2475)-1)])
        f=batch/f'episode_{ep:06d}.npz';np.savez(f,timestamp=np.arange(n)*.04,actions=actions,expert_measured_pose6=poses,
            expert_measured_joints=qs,expert_command_joints=qs,expert_measured_gripper_raw=np.full(n,2800),expert_command_gripper_raw=np.full(n,2800))
        items.append(dict(episode=ep,file=f.name,archive_sha256=sha256(f),tcp_offset_m=TCP))
    (batch/'replay_contract.json').write_text(json.dumps(dict(fps=25,episodes=items,repo='fixture',revision='fixture')))
    clock=[1.];actual=Q.copy();goal=Q.copy();grip=[2800];writes=[]
    class Arm:
        def check_health(self):pass
        def telemetry(self):
            actual[:]+=.65*(goal-actual)
            return actual.copy(),np.zeros(9),{9:grip[0]}
        def goals(self):return goal.copy()
        def joint_velocity_limits_deg_s(self):return [180]*6
        def read_register(self,motor,address,width):return grip[0] if address==116 else 100
        def goal(self,value):goal[:]=value;writes.append(value.copy())
        def close(self):pass
        packet=SimpleNamespace(write4ByteTxRx=lambda *a:(0,0));dxl=SimpleNamespace(COMM_SUCCESS=0);port=None
    output=tmp_path/'run'
    def key(delay):
        stage=json.loads((output/'status.json').read_text())['stage']
        return ord(' ') if stage in ['waiting_start','ready','between'] else 255
    first_window=[True]
    def show(*args):
        if first_window[0]:clock[0]+=.2;first_window[0]=False
    monkeypatch.setattr(m,'TapeArm',Arm);monkeypatch.setattr(m,'GeometryGuard',lambda *a:SimpleNamespace(check=lambda *a:None))
    monkeypatch.setattr(m,'time',SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda dt:clock.__setitem__(0,clock[0]+dt)))
    monkeypatch.setattr(m.cv2,'imshow',show);monkeypatch.setattr(m.cv2,'waitKey',key);monkeypatch.setattr(m.cv2,'destroyAllWindows',lambda:None)
    monkeypatch.setattr(m,'render_plots',lambda *a:[])
    m.run(batch,output,execute=True)
    summary=json.loads((output/'summary.json').read_text())
    assert summary['all_five_completed'] and summary['returned_home'] and summary['no_vla']
    for ep in range(5):
        rows=[json.loads(x)for x in(output/f'episode_{ep:06d}.jsonl').open()]
        assert len(rows)>10 and all('actual_pose6'in r and 'ik_pose6'in r and 'command_pose6'in r for r in rows)
    assert len(writes)>50
    # Resume a paused batch after its first successful episode; do not replay it.
    import shutil
    interrupted=tmp_path/'interrupted';interrupted.mkdir()
    shutil.copy2(output/'manifest.json',interrupted/'manifest.json')
    partial=dict(summary,episodes=summary['episodes'][:1],all_five_completed=False,error='interrupted while paused')
    (interrupted/'summary.json').write_text(json.dumps(partial))
    for suffix in ['.json','.jsonl']:shutil.copy2(output/('episode_000000'+suffix),interrupted/('episode_000000'+suffix))
    output=tmp_path/'resumed'
    m.run(batch,output,execute=True,resume_from=interrupted)
    resumed=json.loads((output/'summary.json').read_text())
    assert resumed['all_five_completed'] and resumed['returned_home']
    assert (output/'episode_000000.jsonl').read_bytes()==(interrupted/'episode_000000.jsonl').read_bytes()
    output=tmp_path/'single-governed'
    m.run(batch,output,execute=True,max_episodes=1,return_home=False,acceleration_deg_s2=600.,jerk_deg_s3=6000.)
    single=json.loads((output/'summary.json').read_text())
    assert single['requested_episodes']==[0] and single['all_requested_completed']
    assert single['stage']=='completed_hold' and not single['returned_home'] and not single['all_five_completed']
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['joint_acceleration_deg_s2']==600 and manifest['joint_jerk_deg_s3']==6000
    assert not (output/'episode_000001.jsonl').exists()
    with pytest.raises(ValueError,match='Acceleration'):
        m.run(batch,tmp_path/'invalid',execute=True,acceleration_deg_s2=float('nan'))
