import numpy as np
import pytest
from real_robot.velocity_innovation import VelocityInnovation
from real_robot.policy_stream import ActionStream


def action(gripper=1.):
    return np.array([.2,.1,.3,.4,.5,.6,gripper])


def test_constant_velocity_is_zero_despite_latency_and_skipped_prefix():
    correction=VelocityInnovation();stream=ActionStream('trial')
    for t in np.arange(0,.321,.04):correction.observe(float(t),[.1,-.2,.3])
    stream.merge('trial',0,np.tile(action(),(20,1)),8)
    selected=stream.pop(8)
    assert stream.popped_source==0
    np.testing.assert_allclose(correction.apply(selected,0.,.33),action(),atol=1e-15)


def test_acceleration_uses_trapezoid_and_original_anchor():
    correction=VelocityInnovation()
    for t in np.arange(0,.201,.04):correction.observe(float(t),[.1*t,0,0])
    result=correction.apply(action(),0.,.2)
    assert result[0]-action()[0]==pytest.approx(.5*.1*.2**2)
    np.testing.assert_array_equal(result[3:],action()[3:])
    # A newer chunk gets its own source; prefix displacement is not added twice.
    result=correction.apply(action(),.12,.2)
    assert result[0]-action()[0]==pytest.approx(.5*.1*.08**2)


def test_euclidean_caps_and_episode_latched_close_fade():
    correction=VelocityInnovation()
    for t in np.arange(0,1.001,.1):correction.observe(float(t),[10*t,10*t,0])
    for i in range(15):
        previous=correction.offset.copy();correction.apply(action(),0.,1.)
        assert np.linalg.norm(correction.offset-previous)<=.005+1e-12
        assert np.linalg.norm(correction.offset)<=.05+1e-12
    assert np.linalg.norm(correction.offset)==pytest.approx(.05)
    correction.apply(action(0),0.,1.)
    assert np.linalg.norm(correction.offset)==pytest.approx(.045)
    for i in range(10):correction.apply(action(),0.,1.)
    np.testing.assert_allclose(correction.offset,0,atol=1e-15)
    assert correction.closed
    correction.reset();assert not correction.closed


def test_duplicate_camera_record_does_not_integrate_again():
    correction=VelocityInnovation()
    correction.observe(0,[0,0,0]);correction.observe(.1,[.1,0,0])
    correction.observe(.1,[999,999,999])
    assert len(correction.samples)==2
    assert correction.apply(action(),0,.1)[0]-action()[0]==pytest.approx(.005)


def test_gap_or_prediction_cannot_be_integrated_as_measured_motion():
    correction=VelocityInnovation()
    correction.observe(0,[0,0,0]);correction.observe(.1,[.1,0,0])
    correction.interrupt();correction.observe(.15,[.5,0,0])
    np.testing.assert_allclose(correction.apply(action(),0,.15),action())
    correction.observe(.5,[1,0,0])
    assert len(correction.samples)==1
    np.testing.assert_allclose(correction.apply(action(),.15,.5),action())


def test_last_measured_velocity_only_extends_to_current_control_time():
    correction=VelocityInnovation()
    correction.observe(0,[.1,0,0]);correction.observe(.1,[.12,0,0])
    assert correction.apply(action(),0,.12)[0]-action()[0]==pytest.approx(.0014)
    with pytest.raises(ValueError):correction.observe(.2,[np.nan,0,0])
