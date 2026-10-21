"""Pick/place geometry and pose helpers shared by the active demonstrator."""
import cv2
import numpy as np
from .vx300s_kinematics import forward, LOWER, UPPER


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


class PickGeometry:
    """Robot/table/box checks with a reduced *finger-only* contact margin.

    The gripper slide model spans both endpoints of the taught opening range.
    It is a conservative envelope, not a finger linkage calibration.
    """
    def __init__(self, path, initial, box):
        from .collision import GeometryGuard
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
