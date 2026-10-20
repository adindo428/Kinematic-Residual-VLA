"""Isaac Lab adapter. Import ONLY after AppLauncher has started the simulator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .config import file_hash
from .math3d import point_jacobian, quaternion, rotation
from .storage import atomic_json
from .usd_geometry import visual_bounds

CAMERAS = ("wrist_cam", "opst_cam", "side_cam")


def _np(value):
    return value.detach().cpu().numpy().copy()


def dependency_check(path):
    from pxr import UsdUtils
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    if unresolved:
        raise ValueError(f"Unresolved USD references/textures in {path}: {unresolved[:8]}")
    return {"layers": len(layers), "assets": len(assets)}


def assemble_layout(cfg, audit):
    """Select a table edge in a fresh training scene. Never edit the asset USD."""
    import omni.usd
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
    errors = []
    object_name = sorted(audit["eligible_objects"])[0]
    obj = audit["eligible_objects"][object_name]
    dependency_check(obj)
    for scene_name, scene_file in sorted(audit["eligible_scenes"].items()):
        try:
            spawn = sim_utils.UsdFileCfg(usd_path=scene_file)
            spawn.func("/World/InspectionHouse", spawn, orientation=(2**-0.5, 2**-0.5, 0, 0))
            root = stage.GetPrimAtPath("/World/InspectionHouse")
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            tables = [p for p in Usd.PrimRange(root) if "Table" in p.GetName() and p.HasAttribute("height")]
            for table in sorted(tables, key=lambda p: str(p.GetPath())):
                box = cache.ComputeWorldBound(table).ComputeAlignedBox()
                lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
                height = float(table.GetAttribute("height").Get())
                if height <= 0 or not 0.3 <= height <= 1.3:
                    continue
                # Robot +X points inward; sample both long edges.
                if hi[0]-lo[0] >= hi[1]-lo[1]:
                    anchors = [([(lo[0]+hi[0])/2, lo[1]+0.08, height], np.pi/2),
                               ([(lo[0]+hi[0])/2, hi[1]-0.08, height], -np.pi/2)]
                    depth, width = hi[1]-lo[1], hi[0]-lo[0]
                else:
                    anchors = [([lo[0]+0.08, (lo[1]+hi[1])/2, height], 0),
                               ([hi[0]-0.08, (lo[1]+hi[1])/2, height], np.pi)]
                    depth, width = hi[0]-lo[0], hi[1]-lo[1]
                center, half = np.array(cfg["trajectory"]["center"]), np.array(cfg["trajectory"]["half_extents"])
                if depth < center[0]+half[0]+0.16 or width/2 < abs(center[1])+half[1]+0.15:
                    continue
                for position, yaw in anchors:
                    # Conservative broad-phase checks against neighboring furniture/walls.
                    parent = table.GetParent()
                    neighbors = [p for p in parent.GetChildren() if p != table]
                    structure = root.GetChild("structure")
                    if structure:
                        neighbors += list(structure.GetChildren())
                    rb = Rotation.from_euler("z", yaw)
                    probes = [np.array(position)+[0, 0, 0.25],
                              np.array(position)+rb.apply(center),
                              np.array(position)+rb.apply([1, 0, 0.6]),
                              np.array(position)+rb.apply([0.5, 1, 0.35])]
                    blocked = False
                    for neighbor in neighbors:
                        bound = cache.ComputeWorldBound(neighbor).ComputeAlignedBox()
                        nlo, nhi = np.array(bound.GetMin()), np.array(bound.GetMax())
                        if any(np.all(p >= nlo-0.12) and np.all(p <= nhi+0.12) for p in probes):
                            blocked = True
                            break
                    if blocked:
                        continue
                    deps = dependency_check(scene_file)
                    # Detect renamed exact copies of reserved assets for the selected pair.
                    sh, oh = file_hash(scene_file), file_hash(obj)
                    for name in audit["reserved_scenes"]:
                        reserved = Path(cfg["assets"]["benchmark"])/"scenes"/name
                        if reserved.exists() and reserved.stat().st_size == Path(scene_file).stat().st_size and file_hash(reserved) == sh:
                            raise ValueError("Selected scene duplicates a reserved asset")
                    for name in audit["reserved_objects"]:
                        reserved = Path(cfg["assets"]["objects"])/name
                        if reserved.exists() and reserved.stat().st_size == Path(obj).stat().st_size and file_hash(reserved) == oh:
                            raise ValueError("Selected object duplicates a reserved asset")
                    layout = dict(scene_usd=scene_file, object_usd=obj, scene_usd_sha256=sh, object_usd_sha256=oh,
                                  audit_sha256=audit["audit_sha256"], robot_position=position,
                                  robot_quaternion=quaternion(rb).tolist(), table_prim=str(table.GetPath()),
                                  table_height=height, table_bounds=[lo.tolist(), hi.tolist()],
                                  dependencies=deps, status="assembled-awaiting-physics-validation")
                    atomic_json(cfg["scene"]["layout"], layout)
                    return layout
        except (ValueError, RuntimeError) as exc:
            errors.append(f"{scene_name}: {exc}")
        finally:
            stage.RemovePrim("/World/InspectionHouse")
    raise ValueError(f"No feasible table/asset layout found. Last failures: {errors[-5:]}")


class IsaacBackend:
    def __init__(self, cfg, layout):
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
        from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
        from isaaclab.utils.math import convert_camera_frame_orientation_convention
        from isaacsim.core.utils.xforms import get_world_pose
        from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
        from pxr import Usd, UsdGeom, UsdPhysics
        import omni.usd

        self.cfg, self.layout = cfg, layout
        self.dt = 1/cfg["collection"]["physics_hz"]
        self.stride = cfg["collection"]["physics_hz"]//cfg["collection"]["fps"]
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=self.dt, render_interval=self.stride, device=cfg["scene"]["device"], use_fabric=True))
        self.device = self.sim.device
        self.base = np.array(layout["robot_position"])
        self.rb = rotation(layout["robot_quaternion"])
        self.episode_step = 0
        house = sim_utils.UsdFileCfg(usd_path=layout["scene_usd"])
        house.func("/World/House", house, orientation=(2**-0.5, 2**-0.5, 0, 0))
        light = sim_utils.DomeLightCfg(intensity=350)
        light.func("/World/DomeLight", light)
        light = sim_utils.DistantLightCfg(intensity=550)
        light.func("/World/KeyLight", light, orientation=(0.92388, 0.382683, 0, 0))
        robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
        robot_cfg.prim_path = "/World/Robot"
        robot_cfg.init_state.pos = tuple(self.base)
        robot_cfg.init_state.rot = tuple(layout["robot_quaternion"])
        if cfg["scene"]["robot_usd"]:
            robot_cfg.spawn.usd_path = cfg["scene"]["robot_usd"]
        robot_cfg.spawn.activate_contact_sensors = True
        robot_cfg.actuators["panda_hand"].effort_limit_sim = 5.0
        robot_cfg.actuators["panda_hand"].velocity_limit_sim = 0.2
        self.robot = Articulation(robot_cfg)
        object_cfg = RigidObjectCfg(prim_path="/World/Target", spawn=sim_utils.UsdFileCfg(
            usd_path=layout["object_usd"], activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(self.base+self.rb.apply(cfg["trajectory"]["center"])),
                                                    rot=tuple(layout["robot_quaternion"])))
        self.object = RigidObject(object_cfg)
        self.contacts = ContactSensor(ContactSensorCfg(prim_path="/World/Robot/.*", update_period=0.0, history_length=0))
        stage = omni.usd.get_context().get_stage()
        target_bodies = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Target"))
                         if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(target_bodies) != 1:
            raise ValueError(f"Expected one target rigid body; found {target_bodies}")
        self.object_contacts = ContactSensor(ContactSensorCfg(prim_path=target_bodies[0], update_period=0.0, history_length=0))
        self.cameras = {}
        import usdrt
        fabric_stage = usdrt.Usd.Stage.Attach(sim_utils.get_current_stage_id())
        class MountedCamera(Camera):
            def _update_poses(sensor, env_ids):
                # Lab 2.3.2's generic child Fabric view can cache a world matrix.
                # Read the renderer's moving rigid parent and compose the authored
                # mount instead; never write a cached camera world pose into Fabric.
                if sensor.cfg.prim_path.endswith("/WristCamera"):
                    parent = fabric_stage.GetPrimAtPath("/World/Robot/panda_hand")
                    matrix = parent.GetAttribute("omni:fabric:worldMatrix").Get()
                    if matrix is None:
                        raise ValueError("Renderer hand transform unavailable")
                    parent_p = np.array(matrix.ExtractTranslation())
                    parent_quat = matrix.ExtractRotationQuat()
                    parent_q = [parent_quat.GetReal(), *parent_quat.GetImaginary()]
                    local = UsdGeom.Xformable(stage.GetPrimAtPath(sensor.cfg.prim_path)).GetLocalTransformation()
                    mount_q = local.ExtractRotationQuat()
                    mount_r = rotation([mount_q.GetReal(), *mount_q.GetImaginary()])
                    p = parent_p+rotation(parent_q).apply(local.ExtractTranslation())
                    q = quaternion(rotation(parent_q)*mount_r)
                else:
                    p, q = get_world_pose(sensor.cfg.prim_path, fabric=False)
                sensor._data.pos_w[env_ids] = torch.as_tensor(p, device=sensor._device, dtype=torch.float32)
                opengl = torch.as_tensor(q, device=sensor._device, dtype=torch.float32).reshape(1, 4)
                sensor._data.quat_w_world[env_ids] = convert_camera_frame_orientation_convention(
                    opengl, origin="opengl", target="world")
        presets = {
            "wrist_cam": ("/World/Robot/panda_hand/WristCamera", [0.065, 0, 0], [0, 2**-0.5, 2**-0.5, 0]),
            "opst_cam": ("/World/OppositeCamera", [1, 0, 0.6], quaternion(Rotation.from_euler("XYZ", [0, 60, 90], degrees=True))),
            "side_cam": ("/World/SideCamera", [0.5, 1, 0.35], quaternion(Rotation.from_euler("XYZ", [-90, 0, 180], degrees=True))),
        }
        for name, (prim, position, q) in presets.items():
            if name != "wrist_cam":
                position = self.base+self.rb.apply(position)
                q = quaternion(self.rb*rotation(q))
            self.cameras[name] = MountedCamera(CameraCfg(prim_path=prim, update_period=0.0,
                update_latest_camera_pose=True,
                height=cfg["camera"]["height"], width=cfg["camera"]["width"], data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=2.3, horizontal_aperture=4.6,
                                               focus_distance=400, clipping_range=(0.01, 10000)),
                offset=CameraCfg.OffsetCfg(pos=tuple(position), rot=tuple(q), convention="opengl")))
        self.sim.reset()
        self.robot.update(self.dt)
        self.object.update(self.dt)
        self.joint_ids, self.joint_names = self.robot.find_joints("panda_joint[1-7]", preserve_order=True)
        self.hand = self.robot.find_bodies("panda_hand")[0][0]
        self.jacobian_index = self.hand-1 if self.robot.is_fixed_base else self.hand
        self.finger_ids = self.robot.find_joints("panda_finger_joint.*")[0]
        self.limits = _np(self.robot.data.joint_pos_limits[0, self.joint_ids])
        self.velocity_limits = np.minimum(_np(self.robot.data.joint_vel_limits[0, self.joint_ids]),
                                         [2.175]*4+[2.61]*3)
        self.contact_indices = [i for i, name in enumerate(self.contacts.body_names) if name != "panda_link0"]
        if not self.contact_indices:
            raise ValueError("No robot contact sensors: collision gate cannot be enforced")
        stage = omni.usd.get_context().get_stage()
        lo, hi = visual_bounds(stage, "/World/Target")
        origin = self.base+self.rb.apply(cfg["trajectory"]["center"])
        self.anchor_world = np.r_[(lo[:2]+hi[:2])/2, hi[2]]-origin
        self.object_half_size = (hi-lo)/2
        self.last_camera_frames = {name: -1 for name in CAMERAS}
        self.preflight()

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)

    def preflight(self):
        t = self.cfg["trajectory"]
        center, half = np.array(t["center"]), np.array(t["half_extents"])
        hover_offset = self.rb.inv().apply(self.anchor_world+[0, 0, self.cfg["controller"]["hover_height"]])
        corners = center + np.array(np.meshgrid(*[[-h, h] for h in half])).T.reshape(-1, 3)
        if max(np.linalg.norm(corners+hover_offset, axis=1)) > 0.78:
            raise ValueError(f"Configured workspace/hover offset exceeds conservative Franka reach; hover_offset={hover_offset.tolist()}")
        if center[2]-half[2]-self.object_half_size[2] < 0.015:
            raise ValueError("Object path may intersect supporting table; raise trajectory.center.z")

    def reset(self, path):
        self.path = path
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos.clone(), self.robot.data.default_joint_vel.clone())
        self.robot.reset()
        self.object.reset()
        self.contacts.reset()
        self.object_contacts.reset()
        for cam in self.cameras.values():
            cam.reset()
        self.last_camera_frames = {name: -1 for name in CAMERAS}
        self.episode_step = 0
        self.set_target(0)

    def set_target(self, time):
        p, v, a = self.path.sample(time)
        self.target_p, self.target_v, self.target_a = p, v, a
        pose = np.r_[self.base+self.rb.apply(p), self.layout["robot_quaternion"]]
        self.object.write_root_pose_to_sim(self.tensor(pose[None]))
        self.object.write_root_velocity_to_sim(self.tensor(np.r_[self.rb.apply(v), [0, 0, 0]][None]))

    def target(self):
        offset = self.rb.inv().apply(self.anchor_world+[0, 0, self.cfg["controller"]["hover_height"]])
        center = np.asarray(self.cfg["trajectory"]["center"])+offset
        half = np.asarray(self.cfg["trajectory"]["half_extents"])
        return dict(position=self.target_p+offset, velocity=self.target_v.copy(),
                    workspace_lower=center-half, workspace_upper=center+half)

    def state(self):
        self.robot.update(0.0)
        data = self.robot.data
        hand_p = _np(data.body_link_pos_w[0, self.hand])
        hand_q = _np(data.body_link_quat_w[0, self.hand])
        hand_v = _np(data.body_link_lin_vel_w[0, self.hand])
        omega = _np(data.body_link_ang_vel_w[0, self.hand])
        offset = rotation(hand_q).apply(self.cfg["controller"]["observation_offset"])
        q = quaternion(self.rb.inv()*rotation(hand_q))
        result = dict(ee_position=self.rb.inv().apply(hand_p+offset-self.base), ee_quaternion=q,
                      ee_velocity=self.rb.inv().apply(hand_v+np.cross(omega, offset)),
                      ee_angular_velocity=self.rb.inv().apply(omega),
                      joint_position=_np(data.joint_pos[0, self.joint_ids]),
                      joint_velocity=_np(data.joint_vel[0, self.joint_ids]),
                      all_joint_position=_np(data.joint_pos[0]), all_joint_velocity=_np(data.joint_vel[0]))
        if not all(np.isfinite(value).all() for value in result.values()):
            raise ValueError("Nonfinite articulation state")
        return result

    def jacobian(self):
        j = _np(self.robot.root_physx_view.get_jacobians()[0, self.jacobian_index])[:, self.joint_ids]
        hand_q = _np(self.robot.data.body_link_quat_w[0, self.hand])
        # PhysX Jacobians are at the body COM; translate to hand origin first.
        com = _np(self.robot.data.body_com_pos_w[0, self.hand])
        origin = _np(self.robot.data.body_link_pos_w[0, self.hand])
        offset = origin-com+rotation(hand_q).apply(self.cfg["controller"]["action_offset"])
        j = point_jacobian(j, offset)
        j[:3], j[3:] = self.rb.inv().as_matrix()@j[:3], self.rb.inv().as_matrix()@j[3:]
        return j

    def apply(self, command):
        self.robot.set_joint_position_target(self.tensor([command["joint_position_target"]]), joint_ids=self.joint_ids)
        self.robot.set_joint_velocity_target(self.tensor([command["joint_velocity_target"]]), joint_ids=self.joint_ids)
        self.robot.set_joint_position_target(self.tensor([[0.04]*len(self.finger_ids)]), joint_ids=self.finger_ids)
        self.robot.write_data_to_sim()

    def step(self, next_time, render=False):
        self.set_target(next_time)
        self.sim.step(render=render)
        self.robot.update(self.dt)
        self.object.update(self.dt)
        self.contacts.update(self.dt, force_recompute=True)
        self.object_contacts.update(self.dt, force_recompute=True)
        self.episode_step += 1

    def contact_force(self):
        arm = _np(self.contacts.data.net_forces_w[0, self.contact_indices])
        obj = _np(self.object_contacts.data.net_forces_w[0])
        return float(max(np.linalg.norm(arm, axis=-1).max(initial=0), np.linalg.norm(obj, axis=-1).max(initial=0)))

    def capture(self, timestamp):
        # Render the CURRENT physics state. No physics advance between RGB and state.
        self.sim.render()
        images, metadata = {}, {}
        target_world = self.base+self.rb.apply(self.target_p)+self.anchor_world
        for name, cam in self.cameras.items():
            cam.update(1/self.cfg["collection"]["fps"], force_recompute=True)
            data = cam.data
            frame = int(cam.frame[0].item())
            if frame <= self.last_camera_frames[name]:
                raise ValueError(f"Stale camera frame: {name}")
            self.last_camera_frames[name] = frame
            rgb = _np(data.output["rgb"][0])[..., :3]
            if rgb.shape != (self.cfg["camera"]["height"], self.cfg["camera"]["width"], 3):
                raise ValueError(f"Camera not ready: {name}")
            images[name] = rgb
            p, q, intrinsic = _np(data.pos_w[0]), _np(data.quat_w_ros[0]), _np(data.intrinsic_matrices[0])
            if name == "wrist_cam":
                hand_p = _np(self.robot.data.body_link_pos_w[0, self.hand])
                hand_q = _np(self.robot.data.body_link_quat_w[0, self.hand])
                expected_p = hand_p+rotation(hand_q).apply([0.065, 0, 0])
                expected_r = rotation(hand_q)*rotation([0, 2**-.5, 2**-.5, 0])*Rotation.from_euler("x", np.pi)
                if np.linalg.norm(p-expected_p) > 0.001 or np.linalg.norm((rotation(q)*expected_r.inv()).as_rotvec()) > .001:
                    raise ValueError(f"Wrist camera pose mismatch: measured_p={p.tolist()} expected_p={expected_p.tolist()} measured_q={q.tolist()} expected_q={quaternion(expected_r).tolist()}")
            local = rotation(q).inv().apply(target_world-p)
            pixel = intrinsic@local
            uv = pixel[:2]/max(pixel[2], 1e-12)
            visible = bool(local[2] > 0.01 and 0 <= uv[0] < rgb.shape[1] and 0 <= uv[1] < rgb.shape[0])
            metadata[name] = dict(timestamp=timestamp, sensor_frame=frame, intrinsic=intrinsic.tolist(),
                                  position_world=p.tolist(), quaternion_world_ros=q.tolist(),
                                  pose_source="renderer_rigid_parent_and_usd_mount" if name == "wrist_cam" else "static_usd",
                                  target_in_frame=visible)
        return images, metadata

    def close(self):
        # Suppress Lab 2.3.2's interactive stop callback for standalone shutdown.
        # Otherwise stopping the timeline enters a render loop, even in headless.
        self.sim._disable_app_control_on_stop_handle = True
        self.sim.stop()
