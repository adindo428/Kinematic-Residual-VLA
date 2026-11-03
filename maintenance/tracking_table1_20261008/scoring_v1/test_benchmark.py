import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
from tracking_benchmark import BenchmarkPath, ObservationCondition, tracking_score


def rows():
    return [{'time': i/200, 'hover_error_m': .02 if i>=200 else .1} for i in range(2001)]


def test_exact_user_success_rule_inclusive_boundary_and_orientation_irrelevant():
    data = rows()
    for r in data: r['orientation_error_deg'] = 180
    s = tracking_score(data, True, 10, 200)
    assert s['success'] and s['successful_acquisition_time_s']==1.


@pytest.mark.parametrize('index', [200, 201, 1999, 2000])
def test_one_bad_sample_or_bad_final_state_fails(index):
    data = rows(); data[index]['hover_error_m'] = .020001
    s = tracking_score(data, True, 10, 200)
    assert not s['success'] and s['failure_capped_time_s']==10


def test_termination_missing_and_nan_cannot_pass():
    assert not tracking_score(rows(), False, 10, 200)['success']
    assert not tracking_score(rows()[:-1], True, 10, 200)['success']
    data=rows(); data[400]['time']=data[399]['time']
    assert not tracking_score(data, True, 10, 200)['success']
    data=rows(); data[400]['hover_error_m']=float('nan')
    assert not tracking_score(data, True, 10, 200)['success']


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
