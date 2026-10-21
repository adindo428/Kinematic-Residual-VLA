"""Shared robot self/table geometry checks using the calibrated mesh model."""
import numpy as np


class GeometryGuard:
    """Check the same mesh model as the offline path, including camera housing.

    Initial convex-mesh overlaps may only diminish (2 mm numerical allowance).
    This is not a complete certificate for the physical scene or flexible cables.
    """
    def __init__(self, model_path, initial):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.baseline = {}
        self._pose(initial)
        for c in self.data.contact:
            if c.dist < -.001:
                self.baseline[tuple(sorted((c.geom1, c.geom2)))] = float(c.dist)

    def _pose(self, q):
        self.data.qpos[:6] = q
        self.data.qpos[6:] = [.021, -.021]
        self.mj.mj_forward(self.model, self.data)

    def check(self, q, minimum_table_clearance=.02):
        self._pose(q)
        m, d = self.model, self.data
        for c in d.contact:
            if c.dist >= -.001: continue
            pair = tuple(sorted((c.geom1, c.geom2)))
            if pair not in self.baseline or c.dist < self.baseline[pair]-.002:
                raise RuntimeError('New or worsening modeled self contact')
        for gid in range(m.ngeom):
            name = self.mj.mj_id2name(m, self.mj.mjtObj.mjOBJ_GEOM, gid) or ''
            if name == 'table' or '/1_base' in name or '/2_shoulder' in name: continue
            rotation = d.geom_xmat[gid].reshape(3, 3)
            if m.geom_type[gid] == self.mj.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[gid]; start = m.mesh_vertadr[mid]; n = m.mesh_vertnum[mid]
                z = (m.mesh_vert[start:start+n]@rotation.T+d.geom_xpos[gid])[:, 2].min()
            elif m.geom_type[gid] == self.mj.mjtGeom.mjGEOM_BOX:
                z = d.geom_xpos[gid, 2]-np.abs(rotation[2])@m.geom_size[gid]
            else: continue
            if z < minimum_table_clearance:
                raise RuntimeError('Modeled geometry/table clearance guard')
