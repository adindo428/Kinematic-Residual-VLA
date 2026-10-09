import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
from tracking_benchmark import BenchmarkPath, ObservationCondition, tracking_score, mean_episode_success_percent


def rows(count=1800,seconds=10):
    return [{'time':i/200,'hover_error_m':.03 if i<count else .1} for i in range(seconds*200+1)]


def test_inclusive_3cm_boundary_and_orientation_irrelevant():
    data=rows()
    for r in data:r['orientation_error_deg']=180
    s=tracking_score(data,True,10,200)
    assert s['success_fraction']==.9 and s['success_percent']==90
    assert s['whole_episode_within_2cm_fraction']==0
    data[400]['hover_error_m']=.030001
    assert tracking_score(data,True,10,200)['success_fraction']==1799/2000


@pytest.mark.parametrize('count',[0,1,100,499,500,501,2000])
def test_continuous_percentage_has_no_binary_gate(count):
    s=tracking_score(rows(count),True,10,200)
    assert s['success']==count/2000
    assert s['success_percent']==100*count/2000


def test_initial_second_counts_terminal_sample_has_zero_duration():
    data=rows(200);data[-1]['hover_error_m']=0
    assert tracking_score(data,True,10,200)['success_fraction']==.1


def test_nonconsecutive_intervals_count():
    data=rows(0)
    for i,r in enumerate(data):r['hover_error_m']=.01 if i%4==0 else .1
    assert tracking_score(data,True,10,200)['success_fraction']==.25


def test_equal_episode_weights_not_pooled_durations():
    a=tracking_score(rows(2000,10),True,10,200)
    b=tracking_score(rows(0,30),True,30,200)
    assert mean_episode_success_percent([a,b])==50


def test_truncated_episode_retains_full_horizon_denominator():
    data=rows(2000)[:1600]
    s=tracking_score(data,False,10,200)
    assert not s['completed'] and not s['telemetry_complete']
    assert s['success_fraction']==.8 and s['observed_within_3cm_fraction']==1
    assert tracking_score([],False,10,200)['success_fraction']==0


def test_explicit_legacy_grid_without_terminal_state():
    with pytest.raises(ValueError):tracking_score(rows()[:-1],True,10,200)
    s=tracking_score(rows()[:-1],True,10,200,require_terminal_state=False)
    assert s['success_fraction']==.9 and s['completed']


@pytest.mark.parametrize('corruption',['duplicate','nan','negative','missing'])
def test_bad_telemetry_rejected(corruption):
    data=rows()
    if corruption=='duplicate':data[400]['time']=data[399]['time']
    if corruption=='nan':data[400]['hover_error_m']=float('nan')
    if corruption=='negative':data[400]['hover_error_m']=-1
    if corruption=='missing':data.pop(400)
    with pytest.raises(ValueError):tracking_score(data,True,10,200)


def test_acquisition_time_separate_from_coverage():
    data=rows(0)
    data[200]['hover_error_m']=.03
    s=tracking_score(data,True,10,200)
    assert s['time_to_first_within_3cm_s']==1 and s['success_fraction']==1/2000
    assert tracking_score(rows(0),True,10,200)['acquisition_time_capped_s']==10


@pytest.mark.parametrize('dimension', ['DA', 'DR', 'MG', 'CR'])
def test_current_velocity_agrees_with_actual_target_motion(dimension):
    trajectory=dict(center=[.45,0,.16], amplitude=[.04,.07,.03], omega=[.8,1.2,1.], phase=[.2,.1,.3])
    p=BenchmarkPath(trajectory, {'dimension': dimension, 'speed_multiplier':1.8, 'push_direction':[0,1,0]})
    for t in [0.,1.4,2.1,3.,5.1,6.]:
        h=1e-6
        pos, vel, acc=p.sample(t)
        np.testing.assert_allclose((p.sample(t+h)[0]-p.sample(t-h)[0])/(2*h),vel,atol=1e-8)
        np.testing.assert_allclose((p.sample(t+h)[1]-p.sample(t-h)[1])/(2*h),acc,atol=1e-8)


def test_rgb_freeze_is_camera_only_and_recovers():
    o=ObservationCondition({'dimension':'MP'})
    a={'opst_cam':np.zeros((2,2,3),dtype='uint8')}
    b={'opst_cam':np.ones((2,2,3),dtype='uint8')}
    o.apply(a,1.96)
    assert not o.apply(b,2.08)['opst_cam'].any()
    assert o.apply(b,2.24)['opst_cam'].all()
