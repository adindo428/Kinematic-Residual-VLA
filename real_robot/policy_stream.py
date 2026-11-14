"""Device-free 25 Hz LAAS indexing and live tracking checks."""
import numpy as np


def action_clock_index(elapsed,period=.04):
    """Nearest nominal action time; tolerate timer jitter around sample centres.

    Late cycles still discard elapsed actions; never send catch-up bursts.
    Observation indexing, chunk expiry and command deadlines must share this
    clock so a timestamp boundary cannot acquire conflicting interpretations.
    """
    if not np.isfinite([elapsed,period]).all() or elapsed<0 or period<=0:
        raise ValueError('Expected nonnegative elapsed time and positive period')
    return int(np.floor(elapsed/period+.5))


def action_deadline_elapsed(index,elapsed,period=.04):
    """A sampled target expires at the next nominal action time.

    Nearest-sample selection is centred on a control tick; crossing its
    half-period rounding boundary during computation does not expire a target.
    The next full tick remains a hard deadline, so late work is never sent.
    """
    if not isinstance(index,(int,np.integer)) or index<0:
        raise ValueError('Expected a nonnegative action index')
    if not np.isfinite([elapsed,period]).all() or elapsed<0 or period<=0:
        raise ValueError('Expected nonnegative elapsed time and positive period')
    return elapsed>=(index+1)*period


class ActionStream:
    def __init__(self, token, max_source_age_steps=None, *, handoff_steps=0):
        # Experimental overlap handoff. Disabled in live evaluation until its
        # latency and tracking have been assessed; not an acceleration bound.
        if not isinstance(handoff_steps,int) or isinstance(handoff_steps,bool) or handoff_steps<0:
            raise ValueError('handoff_steps must be a nonnegative integer')
        self.handoff_steps=handoff_steps
        self.token=token; self.pending={}; self.sources={}; self.last_source=-1; self.last_consumed=-1
        self.max_source_age_steps=max_source_age_steps
        self.popped_source=None

    def merge(self, token, source_index, actions, current_index):
        values=np.asarray(actions,float)
        if token!=self.token or source_index<=self.last_source:
            return 0
        if values.ndim!=2 or values.shape[1]!=7 or not np.isfinite(values).all():
            raise ValueError('Expected finite Nx7 absolute actions')
        self.last_source=source_index
        self.pending={i:a for i,a in self.pending.items() if i>=current_index and i>self.last_consumed}
        self.sources={i:s for i,s in self.sources.items() if i in self.pending}
        for offset,action in enumerate(values):
            index=source_index+offset
            if index>=current_index and index>self.last_consumed:
                previous=self.pending.get(index)
                if previous is not None and self.handoff_steps and index-current_index<self.handoff_steps:
                    # Blend time-aligned future actions. Never stretch a chunk
                    # or replay its expired prefix. SO(3) interpolation avoids
                    # taking a long Euler-angle path at +/-pi boundaries.
                    from scipy.spatial.transform import Rotation
                    u=(index-current_index)/self.handoff_steps
                    weight=u*u*u*(10+u*(-15+6*u))
                    blended=previous+weight*(action-previous)
                    old_rotation=Rotation.from_euler('xyz',previous[3:6])
                    relative=old_rotation.inv()*Rotation.from_euler('xyz',action[3:6])
                    blended[3:6]=(old_rotation*Rotation.from_rotvec(weight*relative.as_rotvec())).as_euler('xyz')
                    self.pending[index]=blended
                else:self.pending[index]=action.copy()
                self.sources[index]=source_index
        return min(len(values),max(0,current_index-source_index))

    def rtc_prefix(self,index,horizon=20):
        """Snapshot only the contiguous, time-aligned pending physical plan."""
        rows=[]
        for step in range(index,index+horizon):
            if step<=self.last_consumed or step not in self.pending:break
            if self.max_source_age_steps is not None and step-self.sources[step]>self.max_source_age_steps:break
            rows.append(self.pending[step].copy())
        return np.asarray(rows,dtype=np.float32).reshape(-1,7)

    def pop(self, index):
        self.pending={i:a for i,a in self.pending.items() if i>=index}
        self.sources={i:s for i,s in self.sources.items() if i in self.pending}
        if index<=self.last_consumed:return None
        result=self.pending.pop(index,None)
        source=self.sources.pop(index,None)
        self.last_consumed=index
        self.popped_source=source
        if (source is not None and self.max_source_age_steps is not None
                and index-source>self.max_source_age_steps):return None
        return result


def motion_profile(name):
    profiles={
        'hardware':dict(tcp_m_s=None,joints_deg_s=None,angular_deg_s=None,joint_accel_deg_s2=120.,joint_jerk_deg_s3=600.),
        'smooth':dict(tcp_m_s=.15,joints_deg_s=60.,angular_deg_s=45.,joint_accel_deg_s2=120.,joint_jerk_deg_s3=600.),
        'auto':dict(tcp_m_s=None,joints_deg_s=None,angular_deg_s=90.),
        'pilot':dict(tcp_m_s=.08,joints_deg_s=12.,angular_deg_s=20.),
        'standard':dict(tcp_m_s=.15,joints_deg_s=25.,angular_deg_s=45.),
    }
    return profiles[name].copy()



def live_camera(record,now,rig_sha,max_age=.2):
    """Image freshness is independent of object detection/velocity availability."""
    stamp=(record.get('timing') or {}).get('oldest_host_arrival_monotonic_s')
    if (record.get('schema')!='front_side_observation_v1' or not record.get('running')
            or record.get('rig_sha256')!=rig_sha or record.get('stale',True)
            or stamp is None or not np.isfinite(stamp) or not 0<=now-stamp<=max_age):
        raise ValueError('Front camera stale or rig mismatch')
    return stamp


def live_object(record, now, rig_sha, max_age=.2):
    if (record.get('schema')!='front_side_observation_v1' or not record.get('running')
            or record.get('rig_sha256')!=rig_sha or record.get('output_frame')!='right_base_estimate'
            or record.get('stale',True)):
        raise ValueError('Tracking unavailable or camera/base calibration mismatch')
    stamp=(record.get('timing') or {}).get('oldest_host_arrival_monotonic_s')
    if stamp is None or not np.isfinite(stamp) or not 0<=now-stamp<=max_age:
        raise ValueError('Fixed-camera observation stale')
    track=record.get('track') or {}; velocity=record.get('velocity_report') or {}
    if (not track.get('accepted') or not track.get('ready') or track.get('frame')!='right_base_estimate'
            or velocity.get('status')!='estimated'):
        raise ValueError('Fresh object position and measured velocity required')
    # A stationary estimate supported by fresh accepted positions is valid.
    # Missing/zero-filled velocities still fail the report-status check above.
    measurement_age=track.get('measurement_age_s',0.)
    if not np.isfinite(measurement_age) or not 0<=measurement_age<=max_age:
        raise ValueError('Object measurement stale')
    p=np.asarray(track.get('position_m'),float); v=np.asarray(velocity.get('velocity_mps'),float)
    if p.shape!=(3,) or v.shape!=(3,) or not np.isfinite(np.r_[p,v]).all():
        raise ValueError('Invalid tracked geometry')
    return np.r_[p,np.zeros(3),v]


def gripper_raw(value, closed_raw, open_raw):
    # Flow matching has an unconstrained output head. Saturate its continuous
    # gripper prediction before mapping to the physically calibrated endpoints.
    if not np.isfinite(value):
        raise ValueError('Predicted gripper must be finite')
    return int(round(closed_raw+(float(np.clip(value,-1,1))+1)/2*(open_raw-closed_raw)))


class LiveObjectEstimator:
    """Bounded occlusion prediction from a trusted measured anchor.

    Camera frames must stay fresh even when their object track disappears. A
    prediction never replaces the measured anchor or renews its expiry time.
    Covariance follows the configured motion hypothesis and process noise;
    this quantifies that model's assumptions, not measured occlusion accuracy.
    """
    def __init__(self,max_gap_s=.5,max_displacement_m=.05,acceleration_std_mps2=.1,motion_model='constant_velocity'):
        values=[max_gap_s,max_displacement_m,acceleration_std_mps2]
        if (not np.isfinite(values).all() or not 0<=max_gap_s<=5.
                or not 0<max_displacement_m<=.1 or acceleration_std_mps2<0):
            raise ValueError('Gap must be 0..5 s, displacement 0..0.1 m, and acceleration noise nonnegative')
        if motion_model not in ('constant_velocity','contact_stop_assumption'):raise ValueError('Unknown occlusion motion model')
        self.motion_model=motion_model;self.stop_tau=.15;self.stop_position_diffusion=.005;self.stop_velocity_std=.003
        self.max_gap=float(max_gap_s);self.max_displacement=float(max_displacement_m)
        self.acceleration_std=float(acceleration_std_mps2);self.reset()

    @property
    def settings(self):
        return dict(model=self.motion_model,max_gap_s=self.max_gap,max_displacement_m=self.max_displacement,
            contact_stop_is_assumption=self.motion_model=='contact_stop_assumption',stop_time_constant_s=self.stop_tau,
            stopped_position_diffusion_m_sqrt_s=self.stop_position_diffusion,stopped_velocity_std_mps=self.stop_velocity_std,
            acceleration_std_mps2=self.acceleration_std,max_position_std_m=.025,max_velocity_std_mps=.15,
            prediction_time='current_front_frame_host_stamp',prediction_renews_anchor=False)

    @staticmethod
    def covariance(value):
        try:cov=np.asarray(value,float)
        except (ValueError,TypeError):return None
        if (cov.shape!=(6,6) or not np.isfinite(cov).all()
                or not np.allclose(cov,cov.T,atol=1e-10) or np.linalg.eigvalsh(cov).min() < -1e-10):return None
        return cov.copy()

    def reset(self):
        self.previous=None;self.stamp=None;self.rig=None;self.anchor_covariance=None;self.diagnostics={}

    def read(self,record,now,rig_sha):
        self.diagnostics={'source':'unavailable'}
        stamp=(record.get('timing') or {}).get('oldest_host_arrival_monotonic_s')
        camera_valid=(record.get('rig_sha256')==rig_sha and record.get('running')
            and record.get('schema')=='front_side_observation_v1' and record.get('output_frame')=='right_base_estimate'
            and not record.get('stale',True) and stamp is not None and np.isfinite(stamp)
            and np.isfinite(now) and 0<=now-stamp<=.2)
        if not camera_valid:
            self.reset();self.diagnostics={'source':'unavailable','reason':'camera_stale_or_calibration_mismatch'}
            raise ValueError('Front camera stale or rig mismatch; prediction anchor cleared')
        if self.stamp is not None and (self.rig!=rig_sha or stamp<self.stamp):
            self.reset();self.diagnostics={'source':'unavailable','reason':'clock_or_calibration_changed'}
            raise ValueError('Observation clock or calibration changed; prediction anchor cleared')
        try:
            environment=live_object(record,now,rig_sha)
        except ValueError as original:
            track=record.get('track')
            # A new measured-but-unreliable track (warming/uncertain/too-fast)
            # must not be disguised as an occlusion. The old filter may expire
            # and publish no track while the front RGB-D stream remains fresh.
            missing=(track is None or (not track.get('accepted') and track.get('frame')=='right_base_estimate'
                                       and track.get('status') in ('coasting','lost')))
            if not missing or self.previous is None or self.anchor_covariance is None or self.max_gap==0:raise original
            gap=float(stamp-self.stamp);anchor_age=float(now-self.stamp)
            self.diagnostics=dict(source='unavailable',reason='prediction_limit',horizon_s=gap,anchor_age_s=anchor_age)
            if not 0<=gap<=anchor_age<=self.max_gap:raise original
            if self.motion_model=='contact_stop_assumption':
                effective_gap=-self.stop_tau*np.expm1(-gap/self.stop_tau)
                effective_age=-self.stop_tau*np.expm1(-anchor_age/self.stop_tau)
                velocity_decay=float(np.exp(-gap/self.stop_tau));uncertainty_decay=float(np.exp(-anchor_age/self.stop_tau))
            else:
                effective_gap=gap;effective_age=anchor_age;velocity_decay=uncertainty_decay=1.
            displacement=self.previous[6:]*effective_gap
            if np.linalg.norm(self.previous[6:]*effective_age)>self.max_displacement:raise original
            # Bound uncertainty through the control time (more conservative
            # than the slightly older image time used by the policy input).
            transition=np.eye(6);transition[:3,3:]=effective_age*np.eye(3);transition[3:,3:]=uncertainty_decay*np.eye(3)
            acceleration=np.vstack([.5*effective_age**2*np.eye(3),effective_age*uncertainty_decay*np.eye(3)])
            cov=transition@self.anchor_covariance@transition.T+self.acceleration_std**2*(acceleration@acceleration.T)
            if self.motion_model=='contact_stop_assumption':
                cov[:3,:3]+=np.eye(3)*self.stop_position_diffusion**2*anchor_age
                cov[3:,3:]+=np.eye(3)*self.stop_velocity_std**2*(1-uncertainty_decay**2)
            pstd=float(np.sqrt(max(0.,np.linalg.eigvalsh(cov[:3,:3]).max())))
            vstd=float(np.sqrt(max(0.,np.linalg.eigvalsh(cov[3:,3:]).max())))
            if pstd>.025 or vstd>.15:raise original
            environment=self.previous.copy();environment[:3]+=displacement;environment[6:]*=velocity_decay
            self.diagnostics=dict(source=self.motion_model+'_occlusion_prediction',predicted=True,contact_stop_is_assumption=self.motion_model=='contact_stop_assumption',horizon_s=gap,anchor_age_s=anchor_age,
                source_host_stamp_s=self.stamp,displacement_m=float(np.linalg.norm(displacement)),
                position_std_m=pstd,velocity_std_mps=vstd,assumed_velocity_mps=self.previous[6:].tolist(),
                model_acceleration_std_mps2=self.acceleration_std,predicted_velocity_mps=environment[6:].tolist())
            return environment
        # Do not let a repeated camera record renew the measurement anchor.
        if self.stamp is None or stamp>self.stamp:
            self.previous=environment.copy();self.stamp=float(stamp);self.rig=rig_sha
            self.anchor_covariance=self.covariance((record.get('track') or {}).get('covariance'))
            if self.anchor_covariance is not None and (np.linalg.eigvalsh(self.anchor_covariance[:3,:3]).max()>.025**2
                    or np.linalg.eigvalsh(self.anchor_covariance[3:,3:]).max()>.15**2):self.anchor_covariance=None
        self.diagnostics=dict(source='accepted_measurement',predicted=False,horizon_s=0.,source_host_stamp_s=self.stamp)
        return environment
