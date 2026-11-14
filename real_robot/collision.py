"""Shared robot self/table geometry checks using the calibrated mesh model."""
import numpy as np

TABLE_CONTACT_FINGERS=frozenset({'vx300s_right/10_left_gripper_finger','vx300s_right/10_right_gripper_finger'})


class GeometryGuard:
    """Check the same mesh model as the offline path, including camera housing.

    Initial convex-mesh overlaps may only diminish (2 mm numerical allowance).
    This is not a complete certificate for the physical scene or flexible cables.
    """
    def __init__(self, model_path, initial, profile='hover'):
        if profile not in ('hover','grasp'):raise ValueError('Unknown geometry profile')
        self.profile=profile
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
        grasp=self.profile=='grasp'
        # Learned grasps permit fingertip/table contact. The legacy 20mm hover
        # margin is inappropriate for picking a marker from the table. Other
        # moving links and the wrist camera must still remain above the plane.
        clearance=0. if grasp else minimum_table_clearance
        names=[self.mj.mj_id2name(m,self.mj.mjtObj.mjOBJ_GEOM,gid)or'' for gid in range(m.ngeom)]
        for c in d.contact:
            if c.dist >= -.001: continue
            pair_names={names[c.geom1],names[c.geom2]}
            if grasp and 'table' in pair_names and bool(pair_names & TABLE_CONTACT_FINGERS):continue
            pair = tuple(sorted((c.geom1, c.geom2)))
            if pair not in self.baseline or c.dist < self.baseline[pair]-.002:
                raise RuntimeError(f'New or worsening modeled self contact: {names[c.geom1]} / {names[c.geom2]} at {c.dist:.4f}m')
        for gid in range(m.ngeom):
            name = names[gid]
            if name == 'table' or '/1_base' in name or '/2_shoulder' in name: continue
            if grasp and name in TABLE_CONTACT_FINGERS:continue
            rotation = d.geom_xmat[gid].reshape(3, 3)
            if m.geom_type[gid] == self.mj.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[gid]; start = m.mesh_vertadr[mid]; n = m.mesh_vertnum[mid]
                z = (m.mesh_vert[start:start+n]@rotation.T+d.geom_xpos[gid])[:, 2].min()
            elif m.geom_type[gid] == self.mj.mjtGeom.mjGEOM_BOX:
                z = d.geom_xpos[gid, 2]-np.abs(rotation[2])@m.geom_size[gid]
            else: continue
            if z < clearance-(.001 if grasp else 0.):
                raise RuntimeError(f'Modeled geometry/table clearance guard: {name} at {z:.4f}m; minimum {clearance:.4f}m')
