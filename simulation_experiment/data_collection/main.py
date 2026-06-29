import argparse
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import sys

import trimesh
import sys
import os

# Add parent directory to path (Robot-TrajectronV3)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from utils_exp.grasp import Grasp, Label
from utils_exp.io import *
from utils_exp.perception import *
from utils_exp.visual import grasp2mesh
from experiment.simulation import ClutterRemovalSim
from utils_exp.transform import Rotation, Transform, select_grasps
import collections
from utils_exp.implicit import get_scene_from_mesh_pose_list
from utils_exp.visual import trimesh_to_open3d, visualize_point_cloud_with_normals
import torch
from PIL import Image
import matplotlib.pyplot as plt
import io as sysio

State = collections.namedtuple("State", ["tsdf", "pc"])

debug = True


def read_mesh(root, scene_id, name="mesh_pose_list"):
    path = root / name / (scene_id + ".npz")
    return np.load(path, allow_pickle=True)['pc']

def main(args):
    global debug
    root = "data/data_packed_train_raw"
    df = read_df(Path(root))
    scene_list = list(set(df.loc[:,"scene_id"].to_numpy()))
    scene_list.sort()

    sim = ClutterRemovalSim("pile", "pile/train", gui=debug, save_file_name=args.save_file_name, randomview=True, planning='curobo')
    
    traj_count = 0
    cur_idx = args.start_scene
    while cur_idx < args.start_scene + (args.num_scenes if args.num_scenes is not None else len(scene_list)):
        scene_id = scene_list[cur_idx]
        mesh_pose_list = read_mesh(Path(root), scene_id)
        sim.recovered_scene(mesh_pose_list)
        sim.save_state()
        tsdf, pc, timing = sim.acquire_tsdf(1, 1, 40)
        sim.robot.add_robot(random=False)
        T_base_task =  torch.tensor(sim.robot.T_base_task.as_matrix(), dtype=torch.float32)

        scene_mask = df.loc[:, "scene_id"] == scene_id
        label_array = df.loc[:, "label"].to_numpy(np.single)#.astype(np.int64)
        pos_mask = (label_array>0.4)
        # print("pos vs neg:", pos_mask.sum(), (~pos_mask).sum())
        mask = scene_mask & pos_mask
        label_array = label_array[mask]

        ori_list = []
        ori_array = df.loc[mask, "qx":"qw"].to_numpy(np.single)
        for term in ori_array:
            ori_list.append(Rotation.from_quat(term))
        # ori_list = np.stack(ori_list,axis=0)
        pos_array = df.loc[mask, "x":"z"].to_numpy(np.single)
        width_array = df.loc[mask, "width"].to_numpy(np.single)

        ### visualize scenes ###
        # grasp_mesh_list = []
        # for i in range(ori_array.shape[0]):
        #     ori_term = ori_array[i]
        #     pos_term = pos_array[i]
        #     g = Grasp(Transform(rotation=Rotation.from_quat(ori_term), translation=pos_term),width=0.08)
        #     grasp_mesh_list.append(grasp2mesh(g, 1))
        #     # grasp_mesh_list.append(trimesh_to_open3d(grasp2mesh(g, 1)))
        # scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list)
        # composed_scene = trimesh.Scene(scene_mesh)
        # for i, g_mesh in enumerate(grasp_mesh_list):
        #     composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
        # composed_scene.show()
        # composed_scene.set_camera(angles=(np.pi / 3.0, 0, - np.pi / 2.0), distance=1.6 * sim.size, center=composed_scene.centroid)
        # data = composed_scene.save_image(resolution=(1280,1080),line_settings= {'point_size': 20})
        # image = np.array(Image.open(sysio.BytesIO(data)))
        # if not os.path.exists("logs"):
        #     os.mkdir("logs")
        # plt.imsave("logs/"+f'round_{cur_idx:03d}'+'.png', image)
        # visualize_point_cloud_with_normals(np.asarray(pc.points))
        # visualize_point_cloud_with_normals(trimesh_to_open3d(scene_mesh), grasps=grasp_mesh_list)
        ### visualize scenes ###

        for j in range(len(pos_array)):
            pos = pos_array[j]
            ori = ori_list[j]
            width = width_array[j]

            T_cur = sim.robot.get_ee_pose()
            g = Transform(ori, pos).as_matrix()
            g = select_grasps(torch.tensor(g, dtype=torch.float32), torch.inverse(T_base_task) @ torch.tensor(T_cur, dtype=torch.float32)).numpy()
            g = Grasp(Transform.from_matrix(g), width)
            # sim.execute_grasp(g,remove=False, allow_contact=True)
            # sim.restore_state()
            for _ in range(1):
                jump = False
                # jump = (np.random.random() > 0.7)
                # jump = True
                success = sim.execute_grasp_robot(g, scene_id=scene_id, pc=np.asarray(pc.points), approaching=(jump is False)) # 
                if jump and success:
                    jump_idx = np.random.randint(0, len(pos_array))
                    T_cur = sim.robot.get_ee_pose()
                    g_jump_pose = Transform(ori_list[jump_idx], pos_array[jump_idx]).as_matrix()
                    g_jump_pose = select_grasps(torch.tensor(g_jump_pose, dtype=torch.float32), torch.inverse(T_base_task) @ torch.tensor(T_cur, dtype=torch.float32)).numpy()
                    g_jump_pose = Grasp(Transform.from_matrix(g_jump_pose), width_array[jump_idx])
                    success = sim.execute_grasp_robot(g_jump_pose, scene_id=scene_id, pc=np.asarray(pc.points), connect_last=True)

                sim.world.p.removeBody(sim.robot.panda)
                sim.restore_state()
                sim.robot.add_robot()
                while sim.robot.detect_contact():
                    sim.world.p.removeBody(sim.robot.panda)
                    sim.robot.add_robot()
                if success:
                    traj_count += 1
                    break
                # g.pose.rotation = g.pose.rotation * Rotation.from_euler("z", np.pi)
        #     grasp_mesh_list.append(trimesh_to_open3d(grasp2mesh(g, label)))
        # scene = trimesh_to_open3d(get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False))
        # o3d.visualization.draw_geometries([scene, ]+ grasp_mesh_list,
        #                                     window_name="Point Cloud with Normals",
        #                                     point_show_normal=True)
        cur_idx += 1
        if traj_count > args.save_interval and debug is False:
            print(f"Saving trajectories at {sim.data_collector.save_file_name}")
            sim.data_collector.save()
            traj_count = traj_count - args.save_interval

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script with save interval and directory as arguments")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Interval (in steps) between saves")
    parser.add_argument("--save_file_name", type=str, default="./data/trajectory/trajectories.npz",
                        help="Directory where checkpoints will be saved")
    parser.add_argument("--no-debug", action='store_true', 
                        help="Disable debug mode for additional output")
    parser.add_argument("--start-scene", type=int, default=0, 
                        help="Start from a specific scene index")
    parser.add_argument("--num-scenes", type=int, default=None,
                        help="Number of scenes to process, if None, process all")
                    
    args = parser.parse_args()
    if args.no_debug:
        debug = False
        import logging
        logging.basicConfig(level=logging.ERROR)

    main(args)