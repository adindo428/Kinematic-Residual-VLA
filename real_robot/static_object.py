"""Stationary approach declarations and offline telemetry phase labels; no I/O."""
import numpy as np

SCHEMA = 'krvla_static_object_v1'
AUTOMATIC_PHASE_METHOD = 'static_success_measured_close_lift_open_v1'


def lock_position(observation, now, max_age_s=.3, max_displacement_m=.02):
    fusion = observation.get('fusion') or {}
    position = np.asarray(fusion.get('position_output_m'), dtype=float)
    stamp = (observation.get('timing') or {}).get('oldest_host_arrival_monotonic_s',
              observation.get('host_published_monotonic_s'))
    if (observation.get('stale', True) or fusion.get('stale', True)
            or position.shape != (3,) or not np.isfinite(position).all()
            or stamp is None or not np.isfinite(stamp) or not 0 <= now-stamp <= max_age_s):
        raise ValueError('Static start needs a fresh object position; place it in view and retry SPACE')
    return dict(schema=SCHEMA, position_m=position.tolist(), measurement_monotonic_s=float(stamp),
                locked_monotonic_s=float(now), max_displacement_m=max_displacement_m,
                position_quality=((observation.get('views') or {}).get('front') or {}).get('quality') or {},
                declaration='operator_confirms_stationary_until_contact_or_lift',
                invalidated_monotonic_s=None)


def validate_lock(context, begin):
    position = np.asarray(context.get('position_m'), dtype=float)
    times = [context.get('measurement_monotonic_s'), context.get('locked_monotonic_s')]
    if (context.get('schema') != SCHEMA or position.shape != (3,) or not np.isfinite(position).all()
            or any(t is None or not np.isfinite(t) for t in times)
            or not 0 <= begin-times[0] <= .3 or abs(begin-times[1]) > 1e-6
            or context.get('declaration') != 'operator_confirms_stationary_until_contact_or_lift'
            or context.get('max_displacement_m') != .02):
        raise ValueError('Invalid per-episode static position lock')
    invalidated = context.get('invalidated_monotonic_s')
    if invalidated is not None and (not np.isfinite(invalidated) or invalidated < begin):
        raise ValueError('Invalid static assumption invalidation time')
    return position


def observe_motion(context, observation, now):
    """Invalidate once on a fresh measured displacement; never re-anchor silently."""
    if context.get('invalidated_monotonic_s') is not None:
        return False
    try:
        current = lock_position(observation, now)
    except ValueError:
        return False  # Occlusion is allowed; it is not proof of stationarity.
    distance = float(np.linalg.norm(np.asarray(current['position_m'])-context['position_m']))
    if distance > context['max_displacement_m']:
        context.update(invalidated_monotonic_s=float(now), invalidation_reason='measured_displacement',
                       measured_displacement_m=distance)
        return True
    return False


def assumption_active(context, events, now):
    ends = [context.get('invalidated_monotonic_s'), events.get('contact'),
            events.get('operator_visible_held'), events.get('release')]
    return all(t is None or now < t for t in ends)


def quality_valid(quality):
    return (quality.get('static_collection_schema') == SCHEMA
            and quality.get('static_assumption_valid') is True
            and quality.get('velocity_input_source') == 'operator_static_assumption')


def annotate_automatic_phases(rows, quality, outcome):
    """Backdate sustained closed-gripper lifts to onset in successful demos.

    This uses future samples offline and the saved episode success declaration.
    It is not independent visual evidence or a live grasp-success detector.
    """
    if outcome!='success':return []
    from .grasp_annotations import lift_intervals
    measured=np.asarray([q.get('measured_gripper_normalized',np.nan) for q in quality],float)
    candidates=lift_intervals(rows,measured)
    z=np.asarray([row['observation.state'][2] for row in rows],float)
    intervals=[]
    for candidate in candidates:
        close,confirmed,end=candidate['close_frame'],candidate['start_frame'],candidate['end_frame']
        bottom=close+int(np.argmin(z[close:confirmed+1]))
        onset=confirmed
        for i in range(bottom+1,confirmed+1):
            if z[i]-z[bottom]>=.003 and np.all(z[i:min(i+3,confirmed+1)]-z[bottom]>=.002):
                onset=i;break
        intervals.append(dict(start_frame=onset,end_frame=end,confirmed_frame=confirmed,
                              minimum_z_frame=bottom,close_frame=close,
                              method=AUTOMATIC_PHASE_METHOD,uses_future=True,manual_visual_review=False))
    # Once any grip opens after a supported lift, the original table position
    # must never become valid again (even if the jaws close for another reason).
    if intervals:
        first=intervals[0]['start_frame']
        for q in quality[first:]:
            q.update(static_assumption_valid=False,velocity_input_source='unavailable_after_static_assumption')
    for interval in intervals:
        for q in quality[interval['start_frame']:interval['end_frame']]:
            q.update(static_automatic_held=True,static_automatic_method=AUTOMATIC_PHASE_METHOD,
                     episode_success=True,episode_success_source='operator_saved_success',
                     phase_annotation_manual_review=False,phase_annotation_source=AUTOMATIC_PHASE_METHOD,
                     phase_annotation_kind='automatic_success_telemetry',phase_annotation_uses_future=True,
                     static_automatic_interval=interval)
    return intervals
