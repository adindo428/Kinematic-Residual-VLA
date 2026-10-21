"""Pure state machine for one stationary blue-tape pickup and overhead release."""
import cv2
import numpy as np

from .geometry import validate_transform
from .vx300s_kinematics import forward, inverse, LOWER, UPPER
from .wrist_alignment import corrections, line_error


def rx(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def inside_polygon(point, polygon, margin=0.):
    return cv2.pointPolygonTest(np.asarray(polygon, np.float32), tuple(map(float, point)), True) >= margin


def clearance_center(polygon):
    """Largest inscribed circle of a convex interior footprint.

    Averaging hull vertices biases the release toward densely sampled edges.
    Maximize the actual distance to all edges instead.
    """
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    polygon = np.asarray(polygon, float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or not np.isfinite(polygon).all():
        raise ValueError('Invalid box interior polygon')
    if len(polygon) < 3 or np.linalg.matrix_rank(polygon-polygon.mean(axis=0)) < 2:
        raise ValueError('Box interior has no usable area')
    equations = ConvexHull(polygon).equations
    result = linprog([0., 0., -1.],
        A_ub=np.column_stack([equations[:, :2], np.linalg.norm(equations[:, :2], axis=1)]),
        b_ub=-equations[:, 2], bounds=[(None, None), (None, None), (0., None)], method='highs')
    if not result.success:
        raise ValueError('Cannot find a release point inside the box')
    return result.x[:2], float(result.x[2])


def pose_error(actual, desired):
    actual, desired = np.asarray(actual), np.asarray(desired)
    translation = np.linalg.norm(actual[:3, 3]-desired[:3, 3])
    rotation = np.linalg.norm(cv2.Rodrigues(desired[:3, :3].T@actual[:3, :3])[0])
    return float(translation), float(rotation)


class PoseServo:
    """Preserve holding compensation while rate-limiting six motor goals."""
    def __init__(self, actual, retained):
        self.nominal = np.array(actual, float)
        self.goal = np.array(retained, float)
        self.compensation = self.goal-self.nominal
        if np.max(np.abs(self.compensation)) > np.deg2rad(3):
            raise ValueError('Existing holding compensation exceeds 3 degrees')

    def step(self, target, actual, dt):
        target, actual = np.asarray(target), np.asarray(actual)
        if not 0 < dt <= .15 or not np.isfinite([*target, *actual, dt]).all():
            raise ValueError('Invalid pick servo feedback/timing')
        if np.max(np.abs(self.goal-actual)) > np.deg2rad(6):
            raise RuntimeError('Pick following error exceeds 6 degrees')
        rate = np.deg2rad(3)
        difference = target-self.nominal
        fraction = min(1., rate*dt/max(np.max(np.abs(difference)), 1e-9))
        candidate = self.nominal+fraction*difference
        old_tcp = forward(self.nominal, [.16, 0, 0])[:3, 3]
        new_tcp = forward(candidate, [.16, 0, 0])[:3, 3]
        distance = np.linalg.norm(new_tcp-old_tcp)
        if distance > .020*dt:
            candidate = self.nominal+(candidate-self.nominal)*(.020*dt/distance)
        self.nominal = candidate
        self.compensation = np.clip(self.compensation+.4*(self.nominal-actual)*dt,
                                    -np.deg2rad(3), np.deg2rad(3))
        requested = self.nominal+self.compensation
        self.goal += np.clip(requested-self.goal, -rate*dt, rate*dt)
        if np.any(self.goal < LOWER) or np.any(self.goal > UPPER):
            raise RuntimeError('Pick goal outside joint limits')
        return self.goal.copy()


class PickGeometry:
    """Robot/table/box checks with a reduced *finger-only* contact margin.

    The gripper slide model spans both endpoints of the taught opening range.
    It is a conservative envelope, not a finger linkage calibration.
    """
    def __init__(self, path, initial, box):
        from .aloha_tape_hover import GeometryGuard
        self.guard = GeometryGuard(path, initial)
        self.box = box

    def check(self, q, contact=False, carrying=False, over_box=False):
        if np.any(np.asarray(q) < LOWER) or np.any(np.asarray(q) > UPPER):
            raise RuntimeError('Joint limit guard')
        g = self.guard
        for aperture in (.021, .057):
            g._pose(q)
            g.data.qpos[6:] = [aperture, -aperture]
            g.mj.mj_forward(g.model, g.data)
            for c in g.data.contact:
                if c.dist >= -.001:
                    continue
                pair = tuple(sorted((c.geom1, c.geom2)))
                if pair not in g.baseline or c.dist < g.baseline[pair]-.002:
                    raise RuntimeError('New/worsening modeled robot contact')
            for gid in range(g.model.ngeom):
                name = g.mj.mj_id2name(g.model, g.mj.mjtObj.mjOBJ_GEOM, gid) or ''
                if name == 'table' or '/1_base' in name or '/2_shoulder' in name:
                    continue
                rotation = g.data.geom_xmat[gid].reshape(3,3)
                if g.model.geom_type[gid] == g.mj.mjtGeom.mjGEOM_MESH:
                    mid = g.model.geom_dataid[gid]
                    start, n = g.model.mesh_vertadr[mid], g.model.mesh_vertnum[mid]
                    vertices = g.model.mesh_vert[start:start+n]@rotation.T+g.data.geom_xpos[gid]
                    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
                elif g.model.geom_type[gid] == g.mj.mjtGeom.mjGEOM_BOX:
                    extent = np.abs(rotation)@g.model.geom_size[gid]
                    lo, hi = g.data.geom_xpos[gid]-extent, g.data.geom_xpos[gid]+extent
                    signs = np.array([[a,b,c] for a in (-1,1) for b in (-1,1) for c in (-1,1)])
                    vertices = (signs*g.model.geom_size[gid])@rotation.T+g.data.geom_xpos[gid]
                else:
                    continue
                finger = 'gripper_finger' in name
                minimum = .002 if contact and finger else .020
                if lo[2] < minimum:
                    raise RuntimeError(f'Table clearance: {name} at {lo[2]:.3f}m, minimum {minimum:.3f}m')
                polygon = np.asarray(self.box['interior_polygon_xy_m'])
                # Solid outer footprint envelope (includes uncertainty beyond the ROI).
                boxlo, boxhi = polygon.min(axis=0)-.025, polygon.max(axis=0)+.025
                below = vertices[vertices[:,2] < self.box['rim_z_base_m']+.010]
                overlaps = len(below) and np.all(below[:,:2].max(axis=0)>=boxlo) and np.all(below[:,:2].min(axis=0)<=boxhi)
                if overlaps:
                    rectangle = np.array([[boxlo[0],boxlo[1]],[boxhi[0],boxlo[1]],boxhi,[boxlo[0],boxhi[1]]],np.float32)
                    hull = cv2.convexHull(below[:,:2].astype(np.float32)).reshape(-1,2)
                    if len(hull)>=3:
                        overlaps = cv2.intersectConvexConvex(hull,rectangle)[0]>1e-8
                    else:
                        overlaps = any(inside_polygon(v,rectangle) for v in hull)
                    if overlaps:
                        raise RuntimeError(f'Robot geometry enters box/rim envelope: {name}')
        tcp = forward(q, [.16,0,0])[:3,3]
        if carrying and not contact and tcp[2]-.045 < self.box['rim_z_base_m']+.020:
            raise RuntimeError('Held-tape envelope below transport clearance')
        if over_box and not inside_polygon(tcp[:2], self.box['interior_polygon_xy_m'], .035):
            raise RuntimeError('Release point lacks 35mm clearance inside box interior')

    def path(self, initial, target, **kwargs):
        n = max(2, int(np.ceil(np.max(np.abs(np.asarray(target)-initial))/np.deg2rad(.5)))+1)
        for u in np.linspace(0,1,n):
            self.check((1-u)*np.asarray(initial)+u*np.asarray(target), **kwargs)


class StationaryPick:
    """One attempt. Decisions depend on measured pose, mask and gripper feedback."""
    def __init__(self, template, box, object_point, actual, now):
        self.template, self.box = template, box
        self.stage = 'OPEN'
        self.started = self.stage_started = now
        self.stable_since = None
        self.last_wrist_stamp = None
        self.roll_gain = None
        self.probe = None
        self.alignment_origin = None
        self.closed_feature = None
        self.contact_raw = None
        self.mask_stable_frames = 0
        self.target_pose = forward(actual, template['tcp_offset_m'])
        self.target_q = None
        self.shift = np.asarray(object_point)-template['object_reference_m']
        if np.linalg.norm(self.shift[:2]) > .15 or abs(self.shift[2]) > .025:
            raise ValueError('Tape outside the first-pilot reference neighborhood (150mm XY / 25mm Z)')
        self.poses = {k: np.array(v['T_base_tcp'],float) for k,v in template['stages'].items()}
        for k in ('pregrasp','before_close','lifted'):
            self.poses[k][:3,3] += self.shift
        release = self.poses['release']
        release[:2,3] = box['release_xy_m']
        release[2,3] = max(release[2,3], box['rim_z_base_m']+.065)
        self.poses['above_box'][:2,3] = box['release_xy_m']
        self.poses['above_box'][2,3] = max(release[2,3]+.060, .23)
        self.open_raw = template['gripper']['open_raw']
        self.closed_raw = template['gripper']['empty_closed_raw']
        self.gripper_target = self.open_raw
        self.done = False

    def advance(self, name, now, pose=None):
        self.stage = name
        self.stage_started = now
        self.stable_since = None
        self.mask_stable_frames = 0
        self.target_q = None
        if pose is not None:
            self.target_pose = np.array(pose,float)

    def flags(self):
        return {'contact': self.stage in ('DESCEND','CLOSE','LIFT'),
                'carrying': self.stage in ('TRANSPORT','ABOVE_BOX','RELEASE'),
                'over_box': self.stage in ('RELEASE','VERIFY_RELEASE','RETREAT')}

    def settle(self, condition, now, seconds=.4):
        if not condition:
            self.stable_since = None
            return False
        if self.stable_since is None:
            self.stable_since = now
        return now-self.stable_since >= seconds

    def update(self, now, actual, raw_gripper, gripper_current, wrist, stationary=True, progress=True):
        if now-self.started > 360 or now-self.stage_started > 100:
            raise RuntimeError('Pick stage/attempt timeout')
        if not stationary and self.stage in ('OPEN','PREGRASP','PROBE','ALIGN','DESCEND'):
            raise RuntimeError('Tape moved before closure; stationary pilot aborted')
        pose = forward(actual, self.template['tcp_offset_m'])
        feature = (wrist or {}).get('feature')
        stamp = ((wrist or {}).get('timing') or {}).get('sensor_timestamp_s')
        fresh = stamp is not None and stamp != self.last_wrist_stamp
        if fresh:
            self.last_wrist_stamp = stamp
        distance, angle = pose_error(pose,self.target_pose)
        arrived = distance < .003 and angle < np.deg2rad(2)
        if self.target_q is not None:
            arrived = np.max(np.abs(np.asarray(actual)-self.target_q)) < np.deg2rad(.6)
        stage = self.stage
        if stage in ('TRANSPORT','ABOVE_BOX'):
            if (feature is None or feature.get('depth_m') is None or
                    abs(feature['depth_m']-self.closed_feature['depth_m'])>.04 or
                    abs(gripper_current)<.12 or raw_gripper<=self.closed_raw+45):
                raise RuntimeError('Held tape lost during transport')
        if not progress:
            if stage=='ALIGN' and (feature is None or not feature.get('axis_reliable') or feature.get('depth_m') is None):
                raise RuntimeError('Wrist tape axis/depth lost while planning alignment')
            if stage=='DESCEND' and feature is None:
                raise RuntimeError('Tape lost while planning descent')
            return
        if stage == 'OPEN':
            if self.settle(raw_gripper >= self.open_raw-12 and abs(gripper_current)<.12, now):
                self.advance('PREGRASP',now,self.poses['pregrasp'])
        elif stage == 'PREGRASP':
            if self.settle(arrived and feature is not None and feature.get('axis_reliable'),now):
                self.alignment_origin = (pose.copy(), float(actual[5]))
                self.probe = {'axis': feature['axis_rad'], 'q8':float(actual[5]), 'q':np.asarray(actual).copy()}
                self.advance('PROBE',now,pose)
                self.target_q = np.asarray(actual).copy()
                self.target_q[5] += np.deg2rad(3)
        elif stage == 'PROBE':
            if self.settle(arrived and feature is not None and feature.get('axis_reliable'),now,.7):
                dq = actual[5]-self.probe['q8']
                da = line_error(feature['axis_rad'],self.probe['axis'])
                gain = da/dq if abs(dq) > np.deg2rad(1.5) else 0.
                if not -1.8 <= gain <= -.35:
                    raise RuntimeError(f'Wrist roll/image response not verified: gain {gain:.3f}')
                self.roll_gain = gain
                self.advance('ALIGN',now,pose)
        elif stage == 'ALIGN':
            if feature is None or not feature.get('axis_reliable') or feature.get('depth_m') is None:
                raise RuntimeError('Wrist tape axis/depth lost during alignment')
            reference = self.template['wrist_reference']
            pixel_error = np.linalg.norm(np.asarray(feature['center_px'])-reference['center_px'])
            axis_error = line_error(reference['axis_rad'],feature['axis_rad'])
            if self.settle(pixel_error < 7 and abs(axis_error)<np.deg2rad(3) and arrived,now,.7):
                delta = pose[:3,3]-self.poses['pregrasp'][:3,3]
                roll = actual[5]-self.alignment_origin[1]
                grasp = self.poses['before_close'].copy()
                grasp[:3,3] += delta
                grasp[:3,:3] = grasp[:3,:3]@rx(roll)
                self.poses['before_close'] = grasp
                self.advance('DESCEND',now,grasp)
            elif fresh and arrived:
                camera = forward(actual)@np.asarray(self.template['T_gripper_wrist_camera'])
                delta, _ = corrections(feature,reference,wrist['timing']['intrinsics'],camera)
                desired = pose.copy()
                desired[:3,3] += delta
                desired[:3,:3] = pose[:3,:3]@rx(float(np.clip(.35*axis_error/self.roll_gain,-np.deg2rad(1),np.deg2rad(1))))
                if np.linalg.norm(desired[:2,3]-self.alignment_origin[0][:2,3]) > .035:
                    raise RuntimeError('Wrist centering exceeded 35mm correction budget')
                if abs(actual[5]-self.alignment_origin[1]) > np.deg2rad(65):
                    raise RuntimeError('Wrist alignment exceeded 65 degree correction budget')
                self.target_pose = desired
        elif stage == 'DESCEND':
            if feature is None:
                raise RuntimeError('Tape lost during descent')
            if self.settle(arrived,now,.5):
                self.advance('CLOSE',now,self.target_pose)
        elif stage == 'CLOSE':
            self.gripper_target = max(self.closed_raw, raw_gripper-32)
            contact = abs(gripper_current) >= .18 and self.closed_raw+45 < raw_gripper < self.open_raw-45
            if self.settle(contact and feature is not None and feature.get('depth_m') is not None,now,.4):
                self.contact_raw = raw_gripper
                self.closed_feature = dict(feature)
                self.gripper_target = max(self.closed_raw,raw_gripper-32)
                lift = self.target_pose.copy()
                lift[2,3] = max(lift[2,3]+.14,self.box['rim_z_base_m']+.095)
                self.advance('LIFT',now,lift)
            elif now-self.stage_started > 8:
                raise RuntimeError('No supported gripper contact; pickup failed')
        elif stage == 'LIFT':
            if self.settle(arrived,now):
                self.advance('VERIFY_LIFT',now,self.target_pose)
        elif stage == 'VERIFY_LIFT':
            held = feature is not None and feature.get('depth_m') is not None
            if held:
                held = (abs(feature['depth_m']-self.closed_feature['depth_m']) < .035
                    and np.linalg.norm(np.asarray(feature['center_px'])-self.closed_feature['center_px']) < 65
                    and raw_gripper > self.closed_raw+45 and abs(gripper_current)>.12)
            if fresh:
                self.mask_stable_frames = self.mask_stable_frames+1 if held else 0
            if self.mask_stable_frames >= 5 and now-self.stage_started>.4:
                high = self.target_pose.copy()
                high[2,3] = max(.25,self.poses['above_box'][2,3])
                self.advance('TRANSPORT',now,high)
            elif now-self.stage_started > 5:
                raise RuntimeError('Lift verification failed: tape did not remain with tool')
        elif stage == 'TRANSPORT':
            if self.settle(arrived,now):
                self.advance('ABOVE_BOX',now,self.poses['above_box'])
        elif stage == 'ABOVE_BOX':
            if self.settle(arrived,now):
                self.advance('RELEASE',now,self.poses['release'])
        elif stage == 'RELEASE':
            if arrived:
                self.gripper_target = self.open_raw
            if self.settle(arrived and raw_gripper>=self.open_raw-12 and abs(gripper_current)<.12,now,.5):
                self.advance('VERIFY_RELEASE',now,self.target_pose)
        elif stage == 'VERIFY_RELEASE':
            placed = False
            if feature is not None and feature.get('depth_m') is not None:
                intr = wrist['timing']['intrinsics']; z = feature['depth_m'];u,v = feature['center_px']
                # Optical distortion is small near the working image center;
                # placement is corroborated conservatively, never inferred from a timer.
                camera_point = np.array([(u-intr['ppx'])*z/intr['fx'],(v-intr['ppy'])*z/intr['fy'],z,1.])
                point = forward(actual)@np.asarray(self.template['T_gripper_wrist_camera'])@camera_point
                placed = (inside_polygon(point[:2],self.box['interior_polygon_xy_m'])
                    and -.015 <= point[2] <= self.box['rim_z_base_m']+.04
                    and z > self.closed_feature['depth_m']+.025)
            if fresh:
                self.mask_stable_frames = self.mask_stable_frames+1 if placed else 0
            if self.mask_stable_frames>=5 and now-self.stage_started>.5:
                retreat = self.target_pose.copy();retreat[2,3] += .06
                self.advance('RETREAT',now,retreat)
            elif now-self.stage_started>10:
                raise RuntimeError('Tape released; placement could not be visually verified')
        elif stage == 'RETREAT':
            if self.settle(arrived,now):
                self.advance('COMPLETE',now,self.target_pose)
                self.done = True
        return self.stage
