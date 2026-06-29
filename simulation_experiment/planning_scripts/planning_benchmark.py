import argparse
from pathlib import Path

import numpy as np
from spatialmath import SE3
import spatialgeometry as sg
from utils_exp.io import *
from utils_exp.perception import *
from experiment.simulation import ClutterRemovalSim
from utils_exp.transform import Rotation, Transform
import collections
from utils_exp.visual import trimesh_to_open3d, visualize_point_cloud_with_normals, pointcloud_to_meshes, grasp2mesh
import torch
import os
from utils_exp.transform import SO3_R3, matrix_to_euler_angles, quaternion_to_matrix, matrix_to_rotation_6d, select_grasps
import matplotlib.pyplot as plt
State = collections.namedtuple("State", ["tsdf", "pc"])
import random


def set_random_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


debug = True
Viz = False

mode = 'neo_ss' # mppi or neo_ss or curobo

def main():
    set_random_seed(42)
    global debug
    root = "data/data_packed_train_raw"
    df = read_df(Path(root))
    scene_list = list(set(df.loc[:,"scene_id"].to_numpy()))
    scene_list.sort()
    sim = ClutterRemovalSim("pile", "pile/train", gui=debug, save_file_name=None, planning=mode)

    count = 0
    success_count = 0
    collision_count = 0
    cur_idx = 0
    num_test_scenes = 1000
    while cur_idx < num_test_scenes:
        # cur_idx = np.random.randint(0, len(scene_list))
        scene_id = scene_list[cur_idx]
        cur_idx += 1
        mesh_pose_list = read_mesh(Path(root), scene_id)
        sim.recovered_scene(mesh_pose_list)
        sim.save_state()

        ## viz
        # env = swift.Swift()
        # env.launch(headless=True)
        if Viz:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111, projection='3d')
            x_range = (0, 0.3)
            y_range = (0, 0.3)
            z_range = (0, 0.5)
            ax.set_xlim([x_range[0], x_range[1]])
            ax.set_ylim([y_range[0], y_range[1]])
            ax.set_zlim([z_range[0], z_range[1]])
            plt.grid()
            plt.ion()

        sim.robot.add_robot()
        


        T_base_task = torch.tensor(sim.robot.T_base_task.as_matrix(), dtype=torch.float)
        T_task_base = torch.inverse(T_base_task)
        dt = 0.05

        ## load pcl
        tsdf, pc, timing = sim.acquire_tsdf(1, 1, 40)
        pcl_ = np.asarray(pc.points)
        pcl_ = torch.tensor(pcl_, dtype=torch.float32)
        pcl_ = T_base_task @ torch.cat((pcl_,torch.ones_like(pcl_[:,:1])), dim=-1).unsqueeze(-1)


        scene_mask = df.loc[:, "scene_id"] == scene_id
        label_array = df.loc[:, "label"].to_numpy(np.single)#.astype(np.int64)
        pos_mask = (label_array>0.4)
        # print("pos vs neg:", pos_mask.sum(), (~pos_mask).sum())
        mask = scene_mask & pos_mask
        grasps = df.loc[mask, "qx":"label"].to_numpy(np.single)

        if grasps.shape[0] == 0:
            num_test_scenes+=1
            continue
        shuffle_idx = torch.randperm(grasps.shape[0])
        count += 1

        for i in range(2):
            T_cur = sim.robot.get_ee_pose()
            
            ori = grasps[shuffle_idx[i], :4]
            pos = grasps[shuffle_idx[i], 4:7]

            g = Transform(Rotation.from_quat(ori), pos).as_matrix()
            g = select_grasps(torch.tensor(g, dtype=torch.float32), torch.inverse(T_base_task) @ torch.tensor(T_cur, dtype=torch.float32)).numpy()
            g = Grasp(Transform.from_matrix(g), 0.08)

            success = sim.execute_grasp_robot(g, scene_id=None, pc=np.asarray(pc.points), approaching=False)


            if success:
                success_count += 1
                if debug:
                    print("success on scene {}".format(scene_id))
                break
            else:
                sim.world.p.removeBody(sim.robot.panda)
                sim.restore_state()
                sim.robot.add_robot()
        
    print("success rate:", success_count/count)
    print("collision rate:", collision_count/count)
    return

def achieved(cur_pose, goal_poses):
    error_mat = torch.inverse(cur_pose) @ goal_poses
    e = torch.cat((error_mat[..., :3,3], matrix_to_euler_angles(error_mat[...,:3,:3], 'ZYX')), dim=-1).abs().sum(-1)
    idx = torch.argmin(e)
    if torch.min(e) < 0.09:
        return True, idx
    else:
        return False, idx
    



if __name__ == "__main__":
    main()