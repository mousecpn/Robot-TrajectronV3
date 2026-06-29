from pathlib import Path
import time

import numpy as np
import pybullet

from utils_exp.grasp import Label
from utils_exp.perception import *
from experiment import btsim, workspace_lines
from utils_exp.transform import Rotation, Transform
from utils_exp.noise import apply_noise
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Slerp
import pybullet_data as pd
import math
from utils_exp.control import calculate_velocity, arrived, velocity_based_control, NEO_SS
import spatialgeometry as sg
import roboticstoolbox as rtb
import spatialmath as sm
try:
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.geom.types import WorldConfig
    from curobo.rollout.rollout_base import Goal
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import JointState, RobotConfig
    from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
    from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

    from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
    from curobo.types.state import JointState
    from curobo.rollout.rollout_base import Goal
    from utils_exp.MotionPlanner import MotionPlanner
    from curobo.geom.types import Mesh, WorldConfig
    from curobo.wrap.reacher.motion_gen import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
        PoseCostMetric,
    )
    from curobo.geom.sdf.utils import create_collision_checker

except:
    print('load curobo failed')
from experiment.logger import DataCollector
from utils_exp.visual import draw_frame_in_pybullet, draw_trajectory_in_pybullet, visualize_point_cloud_with_normals

def reorder_pose_list(pose):
    """
    qx,qy,qz,qw,x,y,z => x,y,z,qw,qx,qy,qz
    """
    if len(pose) == 7:
        new_pose = [pose[4], pose[5], pose[6], pose[3], pose[0], pose[1], pose[2]]
    elif len(pose) == 4: # qx, qy, qz, qw => qw, qx, qy, qz
        new_pose = [pose[3], pose[0], pose[1], pose[2]]
    return new_pose

def create_point_cloud_from_depth_image(depth, camera, organized=False):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

class ClutterRemovalSim(object):
    def __init__(self, scene, object_set, gui=True, seed=None, add_noise=False, sideview=False, randomview=False, save_file_name=None, save_freq=8, planning='curobo'):
        assert scene in ["pile", "packed"]
        assert planning in ['curobo', 'mppi', 'neo_ss']

        self.planning = planning
        self.urdf_root = Path("data/urdfs")
        # self.urdf_root = Path('/home/pinhao/orbitgrasp/simulator/data_robot/urdfs')
        self.scene = scene
        self.object_set = object_set
        self.discover_objects()

        self.global_scaling = {
            "blocks": 1.67,
            "google": 0.7,
            'google_pile': 0.7,
            'google_packed': 0.7,
            'ycb_packed': 0.8
            
        }.get(object_set, 1.0)
        self.gui = gui
        self.add_noise = add_noise
        self.sideview = sideview
        self.randomview = randomview

        self.rng = np.random.RandomState(seed) if seed else np.random
        self.world = btsim.BtWorld(self.gui, None, 8) # save_freq here doesn't mean anything
        self.gripper = Gripper(self.world)
        self.size = 6 * self.gripper.finger_depth
        # intrinsic = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        self.intrinsic = CameraIntrinsic(848, 480, 426.678, 426.67822265625, 427.2525634765625, 234.44296264648438)
        self.camera = self.world.add_camera(self.intrinsic, 0.1, 2.0)
        if planning == 'mppi':
            controller = 'mpc'
        else:
            controller = 'instantaneous'
        self.robot = FrankaRobot(self.world, debug=gui, controller=controller)

        if save_file_name is not None:
            self.data_collector = DataCollector(Path(save_file_name))
        

    @property
    def num_objects(self):
        return max(0, self.world.p.getNumBodies() - 1)  # remove table from body count

    def discover_objects(self):
        root = self.urdf_root / self.object_set
        self.object_urdfs = [f for f in root.rglob("*") if f.suffix == ".urdf"]

    def save_state(self):
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self._snapshot_id)

    def reset(self, object_count):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.draw_workspace()

        if self.gui:
            # self.world.p.resetDebugVisualizerCamera(
            #     cameraDistance=1.0,
            #     cameraYaw=0.0,
            #     cameraPitch=-45,
            #     cameraTargetPosition=[0.15, 0.50, -0.3],
            # )
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=90,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.15, -0.15],
            )

        table_height = self.gripper.finger_depth
        self.place_table(table_height)

        if 'ycb' in self.object_set:
            self.generate_ycb_scene(object_count, table_height)
        else:
            if self.scene == "pile":
                self.generate_pile_scene(object_count, table_height)
            elif self.scene == "packed":
                self.generate_packed_scene(object_count, table_height)
            else:
                raise ValueError("Invalid scene argument")

    def draw_workspace(self):
        points = workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )

    def place_table(self, height):
        urdf = self.urdf_root / "setup" / "plane.urdf"
        pose = Transform(Rotation.identity(), [0.15, 0.15, height])
        self.table_id = self.world.load_urdf(urdf, pose, scale=0.2)

        # define valid volume for sampling grasps
        lx, ux = 0.02, self.size - 0.02
        ly, uy = 0.02, self.size - 0.02
        lz, uz = height + 0.005, self.size
        # lz, uz =  0.00, self.size
        nlz = 0.005
        self.lower = np.r_[lx, ly, lz]
        self.newlower = np.r_[lx, ly, nlz]
        self.upper = np.r_[ux, uy, uz]

    def generate_pile_scene(self, object_count, table_height):
        # place box
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.3)

        # drop objects
        urdfs = self.rng.choice(self.object_urdfs, size=object_count)
        for urdf in urdfs:
            rotation = Rotation.random(random_state=self.rng)
            xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
            pose = Transform(rotation, np.r_[xy, table_height + 0.2])
            scale = self.rng.uniform(0.8, 1.0)
            self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            self.wait_for_objects_to_rest(timeout=1.0)

        # remove box
        self.world.remove_body(box)
        self.remove_and_wait()

    def generate_packed_scene(self, object_count, table_height):
        attempts = 0
        max_attempts = 12
        name_list = []

        while self.num_objects < object_count and attempts < max_attempts:
            self.save_state()
            urdf = self.rng.choice(self.object_urdfs)
            # if 'ycb' in self.object_set:
            #     if urdf in name_list:
            #         continue
            #     else:
            #         name_list.append(urdf)
            
            x = self.rng.uniform(0.08, 0.22)
            y = self.rng.uniform(0.08, 0.22)
            z = 2.0
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            pose = Transform(rotation, np.r_[x, y, z])
            scale = self.rng.uniform(0.7, 0.9)
            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
            self.world.step()

            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:

                self.remove_and_wait()
            attempts += 1

    def generate_ycb_scene(self, object_count, table_height):
        attempts = 0
        max_attempts = 12
        
        name_set = set()

        while self.num_objects < object_count and attempts < max_attempts:
            self.save_state()
            urdf = self.rng.choice(self.object_urdfs)
            if 'ycb' in self.object_set:
                if str(urdf) in name_set:
                    continue                    
            
            x = self.rng.uniform(0.05, 0.25)
            y = self.rng.uniform(0.05, 0.25)
            z = 2.0
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            pose = Transform(rotation, np.r_[x, y, z])
            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.01
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))
            self.world.step()

            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:
                print(urdf)
                name_set.add(str(urdf))
                self.remove_and_wait()
            attempts += 1
    
    def recovered_scene(self, mesh_list):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.draw_workspace()

        if self.gui:
            # self.world.p.resetDebugVisualizerCamera(
            #     cameraDistance=1.0,
            #     cameraYaw=0.0,
            #     cameraPitch=-45,
            #     cameraTargetPosition=[0.15, 0.50, -0.3],
            # )
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=90,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.15, -0.15],
            )

        table_height = self.gripper.finger_depth
        self.place_table(table_height)
        for (mesh_path, scale, pose) in mesh_list:
            pose = Transform.from_matrix(pose)
            mesh_path = str(Path(mesh_path).with_suffix('.urdf')) \
                if 'visual' not in mesh_path.split('_')[-1] \
                else '_'.join(mesh_path.split('_')[:-1])+'.urdf'
            try:
                body = self.world.load_urdf(mesh_path, pose, scale=self.global_scaling * scale)
            except:
                mesh_path = '/'.join(['data']+mesh_path.split('/')[-4:])
                body = self.world.load_urdf(mesh_path, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            # body.set_pose(pose=pose)
            self.world.step()


    def acquire_tsdf(self, n, N=None, resolution=40, camera_base=False):
        """Render synthetic depth images from n viewpoints and integrate into a TSDF.

        If N is None, the n viewpoints are equally distributed on circular trajectory.

        If N is given, the first n viewpoints on a circular trajectory consisting of N points are rendered.
        """
        tsdf = TSDFVolume(self.size, resolution)
        high_res_tsdf = TSDFVolume(self.size, 120)

        if self.sideview:
            origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, self.size / 3])
            theta = np.pi / 3.0
            # theta = np.pi / 100
        else:
            origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, 0])
            theta = np.pi / 6.0
        r = 2.0 * self.size

        N = N if N else n
        if self.randomview:
            ### debug ###
            r = np.random.uniform(1.6, 2.4) * self.size
            theta = np.random.uniform(np.pi / 4.0, 5.0 * np.pi / 12.0)
            phi_list = [np.random.uniform(0, 2.0 * np.pi)]
            
            # r = np.random.uniform(2, 2.5) * self.size
            # theta = np.random.uniform(np.pi / 4, np.pi / 3)
            # phi_list = [np.random.uniform(0.0, np.pi)]
            # origin = Transform(
            #     Rotation.identity(),
            #     np.r_[self.size / 2, self.size / 2, 0.0 + 0.25],
            # )

            # r = np.random.uniform(1.5, 2) * self.size
            # theta = np.random.uniform(np.pi / 4, np.pi / 2.4)
            # phi_list = [np.random.uniform(0.0, np.pi)]
            # origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, 0.0 + 0.15])
            ### debug ###
        elif self.sideview:
            assert n == 1
            # phi_list = [0.0]
            phi_list = [- np.pi / 2.0]
        else:
            phi_list = 2.0 * np.pi * np.arange(n) / N
        
        extrinsics = [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]

        timing = 0.0 # [x,y,z,qx,qy,qz,qw]: -0.15, 0.1616, 0.5200, -0.866, 0, 0, -0.5
        for extrinsic in extrinsics:
            depth_img = self.camera.render(extrinsic)[1]
  
            # add noise 
            depth_img = apply_noise(depth_img, self.add_noise)

            tic = time.time()
            # if camera_base:
            #     pc = create_point_cloud_from_depth_image(depth_img, self.camera.intrinsic)
            # else:
            tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
            high_res_tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
            timing += time.time() - tic

        bounding_box = o3d.geometry.AxisAlignedBoundingBox(self.lower, self.upper)
        pc = high_res_tsdf.get_cloud()
        pc = pc.crop(bounding_box)
        if camera_base:
            return tsdf, pc, timing, extrinsics
        else:
            return tsdf, pc, timing
            

    def advance_sim(self,frames):
        for _ in range(frames):
            self.world.step()
    
    def execute_grasp(self, grasp, remove=True, allow_contact=False):
        T_world_grasp = grasp.pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        approach = T_world_grasp.rotation.as_matrix()[:, 2]
        angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
        if angle > np.pi / 3.0:
            # side grasp, lift the object after establishing a grasp
            T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
            T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
        else:
            T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
            T_world_retreat = T_world_grasp * T_grasp_retreat

        self.gripper.reset(T_world_pregrasp)

        if self.gripper.detect_contact():
            result = Label.FAILURE, self.gripper.max_opening_width
        else:
            self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
            if self.gripper.detect_contact() and not allow_contact:
                result = Label.FAILURE, self.gripper.max_opening_width
            else:
                self.gripper.move(0.0)
                self.gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
                if self.check_success(self.gripper):
                    result = Label.SUCCESS, self.gripper.read()
                    if remove:
                        contacts = self.world.get_contacts(self.gripper.body)
                        self.world.remove_body(contacts[0].bodyB)
                else:
                    result = Label.FAILURE, self.gripper.max_opening_width

        self.world.remove_body(self.gripper.body)

        if remove:
            self.remove_and_wait()

        return result
    



    # def execute_grasp_robot2(self, grasp, scene_id=None, pc=None, approaching=True, connect_last=False):
    #     T_world_grasp = self.robot.T_base_task * grasp.pose
    #     T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.08])
    #     T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

    #     approach = T_world_grasp.rotation.as_matrix()[:, 2]
    #     angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
    #     if angle > np.pi / 3.0:
    #         # side grasp, lift the object after establishing a grasp
    #         T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
    #         T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
    #     else:
    #         T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
    #         T_world_retreat = T_world_grasp * T_grasp_retreat

    #     # self.gripper.reset(T_world_pregrasp)
    #     self.robot.gripper_homing()
    #     success = self.move_robot(T_world_pregrasp, threshold=0.02)
    #     if success is False:
    #         return

        
    #     success = self.move_robot(T_world_grasp)
    #     if success is False:
    #         return

    #     print("Grasp")
    #     self.robot.grasp_robot()

    #     success = self.move_robot(T_world_retreat)
    #     if success is False:
    #         return

    #     return

    def move_robot(self, pose, threshold=0.01, **kwargs):
        if self.robot.controller == "instantaneous":
            success = self.robot.move_robot_instant(pose, threshold=threshold, **kwargs)
        else:
            success = self.robot.move_robot_mpc(pose, threshold=threshold, **kwargs)
        return success


    def execute_grasp_robot(self, grasp, scene_id=None, pc=None, approaching=True, connect_last=False):
        ee_poses = []
        joint_states = []
        
        T_world_grasp = self.robot.T_base_task * grasp.pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        approach = T_world_grasp.rotation.as_matrix()[:, 2]
        angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
        if angle > np.pi / 3.0:
            # side grasp, lift the object after establishing a grasp
            T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
            T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
        else:
            T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
            T_world_retreat = T_world_grasp * T_grasp_retreat

        # self.gripper.reset(T_world_pregrasp)
        self.robot.gripper_homing()
        # _, pc, _ = self.acquire_tsdf(n=1, N=1, resolution=40)
        # pc = np.asarray(pc.points)
        if pc is not None:
            pc = self.robot.T_base_task.transform_point(pc)
        # visualize_point_cloud_with_normals(pc)

        if self.planning == 'curobo':
            waypoints = self.robot.planning(reorder_pose_list((T_world_pregrasp * self.robot.T_tcp_link8).to_list()), pc) # pc
            # if waypoints is None:
            #     T_world_pregrasp.rotation = T_world_pregrasp.rotation * Rotation.from_euler("z", np.pi)
            #     T_world_grasp.rotation = T_world_grasp.rotation * Rotation.from_euler("z", np.pi)
            #     T_world_retreat.rotation = T_world_retreat.rotation * Rotation.from_euler("z", np.pi)
            #     waypoints = self.robot.planning(reorder_pose_list((T_world_pregrasp * self.robot.T_tcp_link8).to_list()), pc ) # pc

            if waypoints is None:
                return False
            for w_i in range(len(waypoints)):
                waypoint = waypoints[w_i]
                self.robot.set_joint_control(waypoint, mode='position')
                self.robot.step(1 * np.exp(0.01*(w_i-len(waypoints)+40)))
                if self.robot.detect_contact():
                    # print("Contact detected before grasping")
                    self.robot.set_joint_control([0.0]* 7, mode='velocity')
                    return False
                ## data collection
                ee_poses.append(self.robot.get_ee_pose())
                joint_states.append(self.robot.read_joint_state()[0])
        elif self.planning == 'mppi':
            try:
                self.robot.mpc.world_coll_checker.clear_cache()
            except:
                pass
            if pc is not None:
                mesh_obstacle = Mesh.from_pointcloud(pc,              # 必须是 numpy (N,3)
                                                    pose=[0,0,0,1,0,0,0],  # xyz+quaternion(qw,qx,qy,qz)
                                                    name="scene_pc")
                world_cfg = WorldConfig(mesh=[mesh_obstacle],
                                    cuboid=self.robot.table_cfg.cuboid
                                    )
            else:
                world_cfg = self.robot.table_cfg
            self.robot.mpc.update_world(world_cfg)
            success = self.move_robot(T_world_pregrasp, threshold=0.03, ee_poses=ee_poses, joint_states=joint_states)
            if success is False:
                return False
        elif self.planning == 'neo_ss':
            success = self.move_robot(T_world_pregrasp, threshold=0.05,  ee_poses=ee_poses, pcl=pc,  joint_states=joint_states) # 
            if success is False:
                return False


        ### approaching
        if approaching:
            if self.planning == 'curobo':
                plan_config = MotionGenPlanConfig(
                    enable_graph=False,
                    enable_graph_attempt=4,
                    max_attempts=10,
                    enable_finetune_trajopt=False,
                    time_dilation_factor=0.5,
                )
                approach_vector = ((T_world_grasp.translation - T_world_pregrasp.translation)/np.linalg.norm(T_world_grasp.translation - T_world_pregrasp.translation)).tolist()
                pose_cost_metric = PoseCostMetric(
                        hold_partial_pose=True,
                        hold_vec_weight=self.robot.tensor_args.to_device([1, 1, 1]+[0,0,0]), # +approach_vector
                    )
                plan_config.pose_cost_metric = pose_cost_metric
                waypoints = self.robot.planning(reorder_pose_list((T_world_grasp * self.robot.T_tcp_link8).to_list()), no_table=True) # plan_config=plan_config
                if waypoints is None:
                    self.robot.set_joint_control([0.0]* 7, mode='velocity')
                    return False
                for wi in range(len(waypoints)):
                    waypoint = waypoints[wi]
                    self.robot.set_joint_control(waypoint, mode='position')
                    self.robot.step(1/2)
            elif self.planning == 'mppi':
                # mesh_obstacle = Mesh.from_pointcloud(np.zeros((1, 3)),              # 必须是 numpy (N,3)
                #                                     pose=[0,0,0,1,0,0,0],  # xyz+quaternion(qw,qx,qy,qz)
                #                                     name="scene_pc")
                # world_cfg = WorldConfig(mesh=[mesh_obstacle])
                # self.robot.mpc.update_world(world_cfg)
                self.robot.mpc.world_coll_checker.clear_cache()
                success = self.move_robot(T_world_grasp)
                # if success is False:
                #     return False
            elif self.planning == 'neo_ss':
                success = self.move_robot(T_world_grasp, watch_dog_limit=100)
                # if success is False:
                #     return False

                ## data collection
                # if (wi+1) % 2 == 0:
                #     ee_poses.append(self.robot.get_ee_pose())
                #     joint_states.append(self.robot.read_joint_state()[0])
                
            self.robot.grasp_robot()


            ### retreating
            if self.planning == 'curobo':
                waypoints = self.robot.planning(reorder_pose_list((T_world_retreat * self.robot.T_tcp_link8).to_list()),  no_table=True) #  plan_config=plan_config
                if waypoints is None:
                    self.robot.set_joint_control([0.0]* 7, mode='velocity')
                else:
                    for waypoint in waypoints:
                        self.robot.set_joint_control(waypoint, mode='position')
                        self.robot.step(1/2)
                # success = self.robot.move_robot(T_world_retreat)
                # if success is False:
                #     return
                self.robot.set_joint_control([0.0]* 7, mode='velocity')
            else:
                success = self.move_robot(T_world_retreat)
                # if success is False:
                #     return False
                self.robot.set_joint_control([0.0]* 7, mode='velocity')
        
            if not self.robot.check_success():
                if self.gui:
                    print("Grasp failed")
                return False

        if scene_id is not None and connect_last is False:
            self.data_collector.add_trajectory(
                scene_id=scene_id,
                ee_poses=np.stack(ee_poses),
                joint_states=np.stack(joint_states),
                robot_base_pose=self.robot.T_base_task.as_matrix(),
            )

            print("add trajectory with length:", len(ee_poses))
        elif scene_id is not None and connect_last is True:
            self.data_collector.add_current_states(scene_id, np.stack(ee_poses), np.stack(joint_states))

        return True


    def remove_and_wait(self):
        # wait for objects to rest while removing bodies that fell outside the workspace
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest()
            removed_object = self.remove_objects_outside_workspace()

    def wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break

    def remove_objects_outside_workspace(self):
        removed_object = False
        for body in list(self.world.bodies.values()):
            xyz = body.get_pose().translation
            if np.any(xyz < 0.0) or np.any(xyz > self.size):
                self.world.remove_body(body)
                removed_object = True
        return removed_object

    def check_success(self, gripper):
        # check that the fingers are in contact with some object and not fully closed
        contacts = self.world.get_contacts(gripper.body)
        res = len(contacts) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res




class Gripper(object):
    """Simulated Panda hand."""

    def __init__(self, world):
        self.world = world
        self.urdf_path = Path("data/urdfs/panda/hand.urdf")

        self.max_opening_width = 0.08
        self.finger_depth = 0.05
        self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.022])
        # self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.])
        self.T_tcp_body = self.T_body_tcp.inverse()

    def reset(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.body = self.world.load_urdf(self.urdf_path, T_world_body)
        self.body.set_pose(T_world_body)  # sets the position of the COM, not URDF link
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            T_world_body,
        )
        self.update_tcp_constraint(T_world_tcp)
        # constraint to keep fingers centered
        self.world.add_constraint(
            self.body,
            self.body.links["panda_leftfinger"],
            self.body,
            self.body.links["panda_rightfinger"],
            pybullet.JOINT_GEAR,
            [1.0, 0.0, 0.0],
            Transform.identity(),
            Transform.identity(),
        ).change(gearRatio=-1, erp=0.1, maxForce=50)
        self.joint1 = self.body.joints["panda_finger_joint1"]
        self.joint1.set_position(0.5 * self.max_opening_width, kinematics=True)
        self.joint2 = self.body.joints["panda_finger_joint2"]
        self.joint2.set_position(0.5 * self.max_opening_width, kinematics=True)

    def update_tcp_constraint(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
            maxForce=300,
        )
    def grasp_object_id(self):
        contacts = self.world.get_contacts(self.body)
        for contact in contacts:
            # contact = contacts[0]
            # get rid body
            grased_id = contact.bodyB
            if grased_id.uid!=self.body.uid:
                return grased_id.uid
            
    def get_distance_from_hand(self,):
        object_id = self.grasp_object_id()
        pos, _ = pybullet.getBasePositionAndOrientation(object_id)
        dist_from_hand = np.linalg.norm(np.array(pos) - np.array(self.body.get_pose().translation))
        return dist_from_hand
    def set_tcp(self, T_world_tcp):
        T_word_body = T_world_tcp * self.T_tcp_body
        self.body.set_pose(T_word_body)
        self.update_tcp_constraint(T_world_tcp)

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
            if abort_on_contact and self.detect_contact():
                return

    def detect_contact(self, threshold=5):
        if self.world.get_contacts(self.body):
            return True
        else:
            return False

    def move(self, width):
        self.joint1.set_position(0.5 * width)
        self.joint2.set_position(0.5 * width)
        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()

    def read(self):
        width = self.joint1.get_position() + self.joint2.get_position()
        return width
    
    def move_gripper_top_down(self):
        current_pose = self.body.get_pose()
        pos = current_pose.translation + 0.1
        flip = Rotation.from_euler('y', np.pi)
        target_ori = Rotation.identity()*flip
        self.move_tcp_pose(Transform(rotation=target_ori,translation=pos),abs=True)
    
    def move_tcp_pose(self, target, eef_step1=0.002, vel1=0.10, abs=False):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp
        pos_diff = target.translation - T_world_tcp.translation
        n_steps = max(int(np.linalg.norm(pos_diff) / eef_step1),10)
        dist_step = pos_diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel1
        key_rots = np.stack((T_world_body.rotation.as_quat(),target.rotation.as_quat()),axis=0)
        key_rots = Rotation.from_quat(key_rots)
        slerp = Slerp([0.0,1.0],key_rots)
        times = np.linspace(0,1,n_steps)
        orientations = slerp(times).as_quat()
        for ii in range(n_steps):
            T_world_tcp.translation += dist_step
            T_world_tcp.rotation = Rotation.from_quat(orientations[ii])
            if abs is True:
                # todo by haojie add the relation transformation later
                self.constraint.change(
                    jointChildPivot=T_world_tcp.translation,
                    jointChildFrameOrientation=T_world_tcp.rotation.as_quat(),
                    maxForce=300,
                )
            else:
                self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
    
    def shake_hand(self,pre_dist):
        grasp_id = self.grasp_object_id()
        current_pose = self.body.get_pose()
        x,y,z = current_pose.translation[0],current_pose.translation[1],current_pose.translation[2]
        default_position = [x, y, z]
        shake_position = [x, y, z+0.05]
        hand_orientation2 = pybullet.getQuaternionFromEuler([np.pi, 0, -np.pi/2])
        shake_orientation1 = pybullet.getQuaternionFromEuler([np.pi, -np.pi / 12, -np.pi/2])
        shake_orientation2 = pybullet.getQuaternionFromEuler([np.pi, np.pi / 12, -np.pi/2])
        new_trans = current_pose.translation + np.array([0.,0.,0.05])
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(hand_orientation2),translation=new_trans))
        #check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(hand_orientation2), translation=default_position))
        #check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(hand_orientation2), translation=shake_position))
        # check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(hand_orientation2), translation=default_position))
        # check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(shake_orientation1), translation=default_position))
        # check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        self.move_tcp_pose(target=Transform(rotation=Rotation.from_quat(shake_orientation2), translation=default_position))
        # check drop
        if self.is_dropped(grasp_id,pre_dist):
            return False
        else:
            return True
        
    def is_dropped(self,object_id,prev_dist):
        pos,_ = pybullet.getBasePositionAndOrientation(object_id)
        dist_from_hand = np.linalg.norm(np.array(pos) - np.array(self.body.get_pose().translation))
        if np.isclose(prev_dist,dist_from_hand,atol=0.1):
            return False
        else:
            return True



class FrankaRobot(object):
    def __init__(self, world, debug=True, controller='instantaneous'):
        self.panda_control = rtb.models.Panda()
        self.watchdog_limit = 70
        self.controller = controller
        self.world = world
        self.world.p.setAdditionalSearchPath(pd.getDataPath())
        self.T_body_tcp = self.T_tcp_link8 = Transform.from_dict({"rotation": [0.000, 0.000, 0.0, 1.0], "translation": [0.000, 0.000, -0.05]})
        self.T_tcp_body = self.T_body_tcp.inverse()
        # self.T_base_task =  Transform.from_dict({"rotation": [0.000, 0.000, 0.0, 1.0], "translation": [0.3,-0.15, 0.15]}) # 0.15
        self.T_base_task = Transform(
            Rotation.from_quat([0, 0, 0, 1]), [0.311, -0.011, 0.11-0.05]
        )
        # self.shift_base = Transform.from_dict({"rotation": [0.000, 0.000, 0.3826, 0.9238], "translation": [0,0,0]})
        self.shift_base = Transform.from_dict({"rotation": [0.000, 0.000, 0., 1], "translation": [0,0,0]})
        self.T_base_task = self.shift_base * self.T_base_task
        # self.T_tcp_link8 = Transform.from_dict({"rotation": [0.000, 0.000, 0.0, 1.0], "translation": [0.000, 0.000, -0.05]})
        self.q_min = np.array([-2.7, -1.7, -2.8, -3.0, -2.8, 0.01, -2.8]) #* 0.5
        self.q_max = np.array([ 2.7,  1.7,  2.8, -0.1,  2.8, 3.7,  2.8]) #* 0.5
        # table size
        try:
            self.table_size = 0.5
            self.T_task_table = Transform.from_list([0, 0, 0, 1, self.table_size/2, self.table_size/2, 0.05])
            # self.T_task_table = Transform.from_list([0, 0, 0, 1, 0.15, 0.15, 0.05])
            T_base_table = reorder_pose_list((self.T_base_task * self.T_task_table).to_list())
            # world_pose = np.asarray([self.table_size/2, self.table_size/2, 0, 1, 0, 0, 0]) + np.concatenate((self.T_base_task.translation,np.array([0,0,0,0])))
            
            self.table_cfg = WorldConfig.from_dict( {
                    "cuboid": {
                        "table": {
                            "dims": [self.table_size, self.table_size, 0.01],  # x, y, z
                            "pose": T_base_table,  # x, y, z, qw, qx, qy, qz
                        },
                    },
                }
            )

            # planning config
            self.planner = MotionPlanner("franka.yml", debug=debug)
            self.tensor_args = TensorDeviceType()
        except:
            print("loading planner failed")
        
        # instantaneous control
        if controller == 'instantaneous':
            # b0 = sg.Cuboid(scale=(self.table_size, self.table_size, 0.01), pose=sm.SE3(self.T_base_task.translation[0]+0.15-self.table_size//2, self.T_base_task.translation[1]+0.15-self.table_size//2, 0.2), collision=True)
            # self.obstacles = [b0]
            self.neo_ss = NEO_SS()
        elif controller == 'mpc':
            # mpc config
            
            robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
            robot_cfg = RobotConfig.from_dict(robot_cfg, self.tensor_args)

            mpc_config = MpcSolverConfig.load_from_robot_config(
                robot_cfg,
                self.table_cfg,
                store_rollouts=True,
                step_dt=0.05,
            )
            self.mpc = MpcSolver(mpc_config)



        

        # # visualization
        # self.fig = plt.figure()
        # self.ax = plt.axes(projection='3d')
        # plt.ion()

    
    def step(self, i=1): # dt =1/240. 0.05s per robot step
        for _ in range(int(12*i)):
            self.world.step()
    
    def keepstep(self, i=1): # dt =1/240. 0.05s per robot step
        while True:
            if self.autostep:
                for _ in range(int(12*i)):
                    self.world.step()
                time.sleep(0.01)
            else:
                time.sleep(0.01)
    
    def get_ee_pose(self):
        joint_positions = []
        for i in range(7):
            pos_i, _, _, _ = self.world.p.getJointState(self.panda, i)
            joint_positions.append(pos_i)
        joint_positions = np.array(joint_positions)
        T_ee = self.panda_control.fkine(joint_positions)
        return np.array(T_ee)
    
    def add_obstacles(self, meshes, mesh_filenames):
        for mesh_filename in mesh_filenames:
            obs = sg.Mesh(filename=mesh_filename, scale=[1,1,1], pose=sm.SE3(), collision=True)
            self.obstacles.append(obs)
        return


    def add_robot(self, random=True):
        self.T_base_task =  Transform.from_dict({"rotation": [0.000, 0.000, 0.0, 1.0], "translation": [0.3,-0.15, 0.15]})
        # self.T_base_task = Transform(
        #     Rotation.from_quat([0, 0, 0, 1]), [0.311, -0.011, 0.11-0.05]
        # )
        small_offset = np.random.normal(0, 0.05, 3)
        small_offset[0] *= 0.5
        # self.T_base_task.translation = small_offset + self.T_base_task.translation
        self.T_base_task = self.shift_base * self.T_base_task
        # update the table pose
        # world_pose = np.asarray([self.table_size/2, self.table_size/2, 0, 1, 0, 0, 0.0]) + np.concatenate((self.T_base_task.translation,np.array([0,0,0,0])))
        self.T_task_table = Transform.from_list([0, 0, 0, 1, self.table_size/2, self.table_size/2, 0.05])
        T_base_table = reorder_pose_list((self.T_base_task * self.T_task_table).to_list())
        # pos, ori = pybullet.getBasePositionAndOrientation(0)
        try:
            self.table_cfg = WorldConfig.from_dict( {
                    "cuboid": {
                        "table": {
                            "dims": [self.table_size, self.table_size, 0.01],  # x, y, z
                            "pose": T_base_table,  # x, y, z, qw, qx, qy, qz
                        },
                    },
                }
            )
        except:
            pass
        # if self.controller == 'instantaneous':
        #     # b0 = sg.Cuboid(scale=(self.table_size, self.table_size, 0.01), pose=sm.SE3(self.T_base_task.translation[0]+0.15-self.table_size//2, self.T_base_task.translation[1]+0.15-self.table_size//2, 0.2), collision=True)
        #     # b0 = sg.Cuboid(scale=(self.table_size, self.table_size, 0.01), pose=sm.SE3(T_base_table[0], T_base_table[1], T_base_table[2]), collision=True)
        #     # self.obstacles = [b0]
        #     self.neo_ss = NEO_SS()
        # elif self.controller == 'mpc':
        #     # mpc config
            
        #     robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
        #     robot_cfg = RobotConfig.from_dict(robot_cfg, self.tensor_args)

        #     mpc_config = MpcSolverConfig.load_from_robot_config(
        #         robot_cfg,
        #         self.table_cfg,
        #         store_rollouts=True,
        #         step_dt=0.05,
        #     )
        #     self.mpc = MpcSolver(mpc_config)


        T_task_base = self.T_base_task.inverse()
        self.panda = self.world.p.loadURDF("franka_panda/panda.urdf", T_task_base.translation, T_task_base.rotation.as_quat(), useFixedBase=True, flags=self.world.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        index = 0
        ### random init pose ###
        if random:
            init_jointPositions=[0.0, -math.pi/4, 0.0, -3*math.pi/4, 0.0, math.pi/2, math.pi/4, 0.0, 0.0]
            jointPositions = np.random.normal(init_jointPositions[:7], np.array(self.q_max - self.q_min)*np.array([0.1,0.04,0.1,0.04,0.1,0.1,0.1])*0.6 )
            jointPositions = np.clip(jointPositions, self.q_min, self.q_max)
            jointPositions = jointPositions.tolist()+[0.0, 0.0]
        else:
            jointPositions=[0.0, -math.pi/4, 0.0, -3*math.pi/4, 0.0, math.pi/2, math.pi/4, 0.0, 0.0]
        #######################

        for j in range(self.world.p.getNumJoints(self.panda)):
            self.world.p.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.world.p.getJointInfo(self.panda, j)

            jointName = info[1]
            jointType = info[2]
            if (jointType == self.world.p.JOINT_PRISMATIC):
                self.world.p.resetJointState(self.panda, j, jointPositions[index]) 
                index=index+1
            if (jointType == self.world.p.JOINT_REVOLUTE):
                self.world.p.resetJointState(self.panda, j, jointPositions[index]) 
                index=index+1
        self.init_pose, self.init_orn = self.world.p.getLinkState(self.panda,11)[:2]
        self.gripper_homing()
    

    def read_joint_state(self):
        joint_pose = []
        joint_vel = []
        for i in range(7):
            pos_i, vel_i, _, _ = self.world.p.getJointState(self.panda, i)
            joint_pose.append(pos_i)
            joint_vel.append(vel_i)
        return np.array(joint_pose), np.array(joint_vel)
    
    def set_joint_control(self, command, mode='position'):
        """Set joint positions and velocities for the robot."""

        if mode == 'velocity':
            for i in range(len(command)):
                self.world.p.setJointMotorControl2(self.panda, i, self.world.p.VELOCITY_CONTROL, targetVelocity=command[i])
        elif mode == 'position':
            for i in range(len(command)):
                self.world.p.setJointMotorControl2(self.panda, i, self.world.p.POSITION_CONTROL, command[i])
        elif mode == 'torque':
            for i in range(len(command)):
                self.world.p.setJointMotorControl2(self.panda, i, self.world.p.TORQUE_CONTROL, force=command[i])
    
    def move_robot_instant(self, pose, threshold=0.01, watch_dog_limit=30, pcl=None, ee_poses=None, joint_states=None):
        pose = pose * self.T_tcp_body
        watch_dog = 0
        joint_pose, joint_vel = self.read_joint_state()
        last_joint = np.array(joint_pose)
        loop_start = time.time()
        while True:
            joint_pose, joint_vel = self.read_joint_state()
            if isinstance(ee_poses, list) and isinstance(joint_states, list):
                ee_poses.append(self.get_ee_pose())
                joint_states.append(joint_pose)
                
            joint_movement = np.linalg.norm(joint_pose-last_joint)
            last_joint = joint_pose
            if joint_movement < 0.001:
                watch_dog += 1
            else:
                watch_dog = 0
            if time.time() - loop_start > 10:
                break

            # target_vel, arrived = calculate_velocity(self.panda_control, np.array(joint_pose), pose, threshold=threshold, obstacles=self.obstacles+pcl) #
            # target_vel, arrived = self.neo_ss.calculate_velocity_ss(self.panda_control, np.array(joint_pose), pose, threshold=threshold, pcl=pcl) #
            params = {'di':0.4,'ds':0.05,'xi':1.0}
            target_vel = None
            for _attempt in range(3):
                try:
                    target_vel, arrived = self.neo_ss.calculate_velocity_ss(self.panda_control, np.array(joint_pose), pose, Gain=1, threshold=threshold, pcl=pcl, params=params) #
                    break
                except:
                    params['ds'] -= 0.01  # reduce ds to be less conservative
                    # print("controller failed, reduce ds to ", params['ds'])
            if target_vel is None:
                target_vel, arrived = self.neo_ss.calculate_velocity_ss(self.panda_control, np.array(joint_pose), pose, threshold=threshold, params=params) #
            if arrived is True:
                return True
            self.set_joint_control(target_vel, mode='velocity')
            self.step()
            if watch_dog > watch_dog_limit:
                break
        self.set_joint_control([0.0]*7, mode='velocity')
        self.step()

        return False
    
    def planning(self, target_pose, pcl=None, plan_config=None, no_table=False):
        # world_config_dict = {
        #     "cuboid": {
        #         "table": {
        #             "dims": [0.3,0.3,0.01],  # x, y, z
        #             "pose": np.asarray([0.45, 0., 0.15, 1, 0, 0, 0.0]),  # x, y, z, qw, qx, qy, qz
        #         },
        #     },
        # }
        world_cfg = {}
        if pcl is not None:
            mesh_obstacle = Mesh.from_pointcloud(pcl,              # 必须是 numpy (N,3)
                                                pose=[0,0,0,1,0,0,0],  # xyz+quaternion(qw,qx,qy,qz)
                                                name="scene_pc")
            world_cfg = WorldConfig(mesh=[mesh_obstacle],
                                    cuboid=self.table_cfg.cuboid
                                    )
        elif no_table:
            world_cfg = WorldConfig.from_dict({})
        else:
            world_cfg = self.table_cfg

        self.planner.update_world(world_cfg)
        result, success = self.planner.plan(self.read_joint_state()[0], target_pose, plan_config=plan_config)
        if success:
            joint_waypoints = result.get_interpolated_plan().position
            joint_waypoints = joint_waypoints.detach().cpu().numpy()
        else:
            joint_waypoints = None
        return joint_waypoints
    
    def detect_contact(self,):
        points = self.world.p.getContactPoints(self.panda)
        if len(points) == 0:
            return False
        return True
        # contacts = []
        # for point in points:
        #     contact = btsim.Contact(
        #         bodyA=self.panda,
        #         bodyB=self.world.bodies[point[2]],
        #         point=point[5],
        #         normal=point[7],
        #         depth=point[8],
        #         force=point[9],
        #     )
        #     contacts.append(contact)
        # return contacts
    
    def check_success(self):
        # check that the fingers are in contact with some object and not fully closed

        left_finger = self.world.p.getJointState(self.panda, 9)[0]
        right_finger = self.world.p.getJointState(self.panda, 10)[0]
        if left_finger+right_finger > 0.1 * 0.08:
            return True
        return False


    def move_robot_mpc(self, pose, threshold=0.01, watch_dog_limit=30, ee_poses=None, joint_states=None):
        achieved = False
        pose = pose * self.T_tcp_link8
        watch_dog = 0
        joint_pose, joint_vel = self.read_joint_state()
        last_joint = np.array(joint_pose)

        joint_names = self.mpc.rollout_fn.joint_names
        rotation = pose.rotation.as_quat()
        rotation = [rotation[3], rotation[0], rotation[1], rotation[2]]  # x,y,z,w => w,x,y,z
        goal_pose = Pose(
            position=self.tensor_args.to_device(pose.translation),
            quaternion=self.tensor_args.to_device(rotation),
        )

        current_state = JointState.from_numpy(position=last_joint.reshape(-1), joint_names=joint_names)

        goal = Goal(
            current_state=current_state,
            # goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
            goal_pose=goal_pose,
        )
        

        goal_buffer = self.mpc.setup_solve_single(goal, 1)
        self.mpc.update_goal(goal_buffer)
        mpc_result = self.mpc.step(current_state, max_attempts=2)
        cmd_state_full = None
        ### Mpc Solver init ###
        count = 0

        while True:
            joint_pose, joint_vel = self.read_joint_state()
            mpc_trajectory = self.mpc.get_visual_rollouts()
            if isinstance(ee_poses, list) and isinstance(joint_states, list) and count % 2 == 0:
                ee_poses.append(self.get_ee_pose())
                joint_states.append(joint_pose)
            count += 1
            # for i in range(len(mpc_trajectory)):
            # draw_trajectory_in_pybullet(self.world.p, mpc_trajectory[-1].detach().cpu().numpy())
            
            joint_movement = np.linalg.norm(np.array(joint_pose)-last_joint)
            last_joint = np.array(joint_pose)
            if joint_movement < 0.004:
                watch_dog += 1
            else:
                watch_dog = 0
            

            cu_js = JointState.from_numpy(
                position=np.array(joint_pose).reshape(-1),
                # velocity=tensor_args.to_device(joint_pose) * 0.0,
                # acceleration=tensor_args.to_device(joint_pose) * 0.0,
                # jerk=tensor_args.to_device(joint_pose) * 0.0,
                joint_names=self.mpc.joint_names,
            )
            # print("cu_js:", cu_js.position.cpu().numpy())
            cu_js = cu_js.get_ordered_joint_state(self.mpc.rollout_fn.joint_names)

            # if cmd_state_full is None:
            #     current_state.copy_(cu_js)
            # else:
            #     current_state_partial = cmd_state_full.get_ordered_joint_state(
            #         mpc.rollout_fn.joint_names
            #     )
            #     current_state.copy_(current_state_partial)
            #     current_state.joint_names = current_state_partial.joint_names
                # current_state = current_state.get_ordered_joint_state(mpc.rollout_fn.joint_names)
            # common_js_names = []
            current_state.copy_(cu_js)

            current_pose_mpc = self.mpc.compute_kinematics(current_state).ee_pose.get_matrix().cpu().numpy()
            eTep = Transform.from_matrix(pose.inverse().as_matrix() @ current_pose_mpc[0])
            e = np.sum(np.abs(np.r_[eTep.translation, eTep.rotation.as_euler('xyz') * np.pi / 180]))
            if e < threshold:
                achieved = True
                break

            mpc_result = self.mpc.step(current_state, max_attempts=2)

            target_vel = mpc_result.action.velocity.cpu().numpy()
            # draw_trajectory_in_pybullet(self.world.p, self.T_base_task.inverse().transform_point(mpc_trajectory[-1].detach().cpu().numpy()))

            self.world.p.removeAllUserDebugItems()

            self.set_joint_control(target_vel, mode='velocity')
            self.step(0.5)
            if watch_dog > watch_dog_limit:
                break
        self.set_joint_control([0.0]*7, mode='velocity')
        self.step(0.5)
        return achieved
        


    def grasp_robot(self):
        self.world.p.setJointMotorControl2(self.panda, 9, self.world.p.POSITION_CONTROL, 0.0, force=20)
        self.world.p.setJointMotorControl2(self.panda, 10, self.world.p.POSITION_CONTROL, 0.0, force=20.)
        self.step(2)
        for i in range(7):
            self.world.p.setJointMotorControl2(self.panda, i, self.world.p.VELOCITY_CONTROL, targetVelocity=0.0)
        self.step()

    def gripper_homing(self):
        self.world.p.setJointMotorControl2(self.panda, 9, self.world.p.POSITION_CONTROL, 0.04,)
        self.world.p.setJointMotorControl2(self.panda, 10, self.world.p.POSITION_CONTROL, 0.04)
        self.step(2)
        # print(p.getJointState(self.panda,0)[1])
        if abs(self.world.p.getJointState(self.panda,9)[0] - 0.04) < 1e-5:
            return True
        return False

    def get_gripper_width(self):
        left_finger = self.world.p.getJointState(self.panda, 9)[0]
        right_finger = self.world.p.getJointState(self.panda, 10)[0]
        return left_finger + right_finger