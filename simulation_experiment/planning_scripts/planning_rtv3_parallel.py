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
import sys
import os

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
import torch
import os
from model.trajectron import Trajectron
from argument_parser import args
from model.transform import SO3_R3, matrix_to_euler_angles, quaternion_to_matrix, matrix_to_rotation_6d, select_grasps
from dataset.se3_preprocessing import se3_derivatives_of, js_derivatives_of,  plot_se3_poses
import matplotlib.pyplot as plt
from utils_exp.control import calculate_velocity, velocity_based_control, NEO_SS
State = collections.namedtuple("State", ["tsdf", "pc"])
import random
import swift
from utils_exp.noise import set_random_seed




debug = False
relative = True
Viz = False


def achieved(cur_pose, goal_poses):
    error_mat = torch.inverse(cur_pose) @ goal_poses
    e = torch.cat((error_mat[..., :3,3], matrix_to_euler_angles(error_mat[...,:3,:3], 'ZYX')), dim=-1).abs().sum(-1)
    idx = torch.argmin(e)
    if torch.min(e) < 0.2:
        return True, idx
    else:
        return False, idx
    

def init_model(config_path=None):
    torch.cuda.set_device('cuda:0')

    # Load hyperparameters from json
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../..', 'config', 'config.json')
    
    with open(config_path, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)
    

    # Add hyperparams from arguments
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval
    hyperparams['frequency'] = 20

    args.checkpoint = "checkpoints/line28.pth"
    trajectron = Trajectron(hyperparams, args.device)
    model = torch.load(args.checkpoint)
    trajectron.model.node_modules = model
    trajectron.set_annealing_params()
    max_hl = hyperparams['maximum_history_length']
    ph = hyperparams['prediction_horizon']
    trajectron.model.to(args.device)
    trajectron.model.eval()
    return trajectron


def process_scene(scene_ids, root, T_body_tcp, T_grasp_pregrasp, args, random_seeds, num_steps_limit=150):
    """
    处理单个场景的抓取模拟和测试。
    返回: (success_count, collision_count, total_count)
    """
    # 局部初始化 sim 和 controller
    # 注意：Sim对象不能在进程间共享，必须在每个进程内创建。
    # 确保ClutterRemovalSim的初始化参数适合并行环境（例如，可能需要禁用GUI）。
    trajectron = init_model()
    global debug
    sim = ClutterRemovalSim("pile", "pile/train", gui=False, save_file_name=None)
    controller = NEO_SS()
    
    # 局部计数器
    success_count = 0
    collision_count = 0
    total_count = len(scene_ids) # 每个场景计为一次总测试

    # 1. 场景加载
    for scene_id, seed in zip(scene_ids, random_seeds):
        torch.manual_seed(seed)
        mesh_pose_list = read_mesh(Path(root), scene_id)
        sim.recovered_scene(mesh_pose_list)
        sim.save_state()

        # 2. 机器人设置
        sim.robot.add_robot(random=False)
        T_base_task = torch.tensor(sim.robot.T_base_task.as_matrix(), dtype=torch.float)
        T_task_base = torch.inverse(T_base_task)
        dt = 0.05
        
        # 3. 感知/PCL加载
        tsdf, pc, timing = sim.acquire_tsdf(1, 1, 40)
        pcl_ = np.asarray(pc.points)
        pcl_ = torch.tensor(pcl_, dtype=torch.float32)
        pcl_ = T_base_task @ torch.cat((pcl_,torch.ones_like(pcl_[:,:1])), dim=-1).unsqueeze(-1)
        
        # 4. 抓取数据加载
        df = read_df(Path(root)) # 在每个进程内重新加载df
        scene_mask = df.loc[:, "scene_id"] == scene_id
        label_array = df.loc[:, "label"].to_numpy(np.single)
        pos_mask = (label_array>0.4)
        mask = scene_mask & pos_mask
        grasps = df.loc[mask, "qx":"label"].to_numpy(np.single)
        rotations = quaternion_to_matrix(torch.tensor(grasps[:, :4], dtype=torch.float))
        translations = torch.tensor(grasps[:, 4:7], dtype=torch.float)
        grasps_viz = SO3_R3(rotations, translations).to_matrix()
        grasps_viz = grasps_viz @ torch.tensor(T_body_tcp, dtype=torch.float) @ torch.tensor(T_grasp_pregrasp, dtype=torch.float)
        shuffle_idx = torch.randperm(grasps.shape[0])

        # 5. 轨迹初始化
        js_traj_ori = []
        ee_traj_ori = []
        cur_js, _ = sim.robot.read_joint_state()
        cur_pos = sim.robot.get_ee_pose()
        for _ in range(7):
            ee_traj_ori.append(cur_pos)
            js_traj_ori.append(cur_js)

        ee_traj_ori = torch.tensor(np.stack(ee_traj_ori, axis=0), dtype=torch.float32)
        js_traj_ori = torch.tensor(np.stack(js_traj_ori, axis=0), dtype=torch.float32)
        
        # 6. 运动规划和控制循环
        step_count = 0
        while step_count < num_steps_limit:
            step_count += 1
            
            # --- 状态更新/历史轨迹构建 --- (与原代码相同)
            cur_js, _ = sim.robot.read_joint_state()
            cur_pos = sim.robot.get_ee_pose()
            ee_traj_ori = torch.cat((ee_traj_ori, torch.tensor(cur_pos, dtype=torch.float32).unsqueeze(0)), dim=0)
            js_traj_ori = torch.cat((js_traj_ori, torch.tensor(cur_js, dtype=torch.float32).unsqueeze(0)),dim=0)

            ee_vel_traj = se3_derivatives_of(ee_traj_ori[-9:], dt=dt)[-8:]
            js_vel_traj = js_derivatives_of(js_traj_ori[-9:], dt=dt)[-8:]

            first_history_index = torch.LongTensor(np.array([0])).to(args.device)
            T_curpose_base = torch.inverse(ee_traj_ori[-1])
            
            if relative:
                ee_traj_rel = T_curpose_base @ ee_traj_ori[-8:]
            else:
                ee_traj_rel = ee_traj_ori[-8:]
                
            ee_traj_logmap = SO3_R3.from_matrix(ee_traj_rel).log_map()
            ee_traj = torch.cat((ee_traj_logmap, ee_vel_traj), dim=-1) 
            js_traj = torch.cat((js_traj_ori[-8:], js_vel_traj), dim=-1)
            ph = 12
            x = ee_traj[-8:,:].unsqueeze(0).to(args.device)
            q = js_traj[-8:,:].unsqueeze(0).to(args.device)
            y = torch.zeros(1, ph, x.shape[-1]).to(args.device)
            
            # --- 抓取和PCL数据处理 ---
            grasps_data = T_base_task @ grasps_viz
            if relative:
                grasps_data = T_curpose_base @ grasps_data
                pcl = T_curpose_base @ pcl_
            else:
                pcl = pcl_
            pcl = pcl[:,:3,0]

            grasps_data = select_grasps(grasps_data, T=ee_traj_ori[-1] if not relative else None)
            grasps_data_ = grasps_data[shuffle_idx][:2]
            grasps_data = SO3_R3.from_matrix(grasps_data_).log_map()
            
            context = {
                'grasp': [grasps_data.to(args.device)],
                'pcl': [pcl.to(args.device)]
            }
            
            # --- 模型预测 ---
            batch = (first_history_index, x, q, y, context)
            with torch.no_grad():
                _, _, predictions = trajectron.predict(batch,
                                        ph=ph,
                                        num_samples=1, z_mode=True, gmm_mode=True, all_z_sep=False, full_dist=False, dist=True)
            
            # --- 动作和控制 ---
            next_action = predictions[0,0,0] # relative pose
            
            if relative:
                next_action = ee_traj_ori[-1].numpy() @ next_action
                success, _ = achieved(torch.tensor(next_action, dtype=torch.float32), ee_traj_ori[-1] @  grasps_data_)
            else:
                success, _ = achieved(torch.tensor(next_action, dtype=torch.float32), grasps_data_)
            
            if success:
                success_count += 1
                print(f"Scene {scene_id} grasp succeeded.")
                break
                
            q0 = sim.robot.read_joint_state()[0]
            next_action = Transform.from_matrix(next_action).as_matrix()
            
            # 运动控制
            j_vel, arrived = calculate_velocity(sim.robot.panda_control, q0, next_action, Gain=15)
            
            # 模拟步进
            sim.robot.set_joint_control(j_vel, 'velocity')
            sim.robot.step()
            
            # 碰撞检测
            collision = sim.robot.detect_contact()
            if collision:
                collision_count = 1
                break
    print(f"Process finished for scenes. Successes: {success_count}, Collisions: {collision_count}, Total: {total_count}")
    
    return success_count, collision_count, total_count

def main():
    set_random_seed(42)
    global debug
    root = "data/data_packed_train_raw"
    df = read_df(Path(root))
    scene_list = list(set(df.loc[:,"scene_id"].to_numpy()))
    scene_list.sort()
    
    T_body_tcp = Transform.from_dict({"rotation": [0.000, 0.000, 0.0, 1.0], "translation": [0.000, 0.000, 0.05]}).as_matrix()
    T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05]).as_matrix()

    num_test_scenes = 100
        
    # ----------------- 使用 Pool 进行并行处理 -----------------
    
    # 确定要使用的CPU核心数，通常建议不超过核心总数
    num_processes = 10
    
    # 确定要测试的场景列表
    scenes_to_test = scene_list[:num_test_scenes]
    L = len(scenes_to_test)
    # 构建 process_scene 函数所需的参数列表
    tasks = []
    for i in range(num_processes):
        idx = list(range(i, L, num_processes))
        scene_ids = scenes_to_test[i::num_processes]
        tasks.append((scene_ids, root, T_body_tcp, T_grasp_pregrasp, args, idx))

    with multiprocessing.Pool(processes=num_processes) as pool:
        # map 或 starmap 用于将任务分发给进程
        # starmap 适用于接收多个参数的函数
        results = pool.starmap(process_scene, tasks)

    # ----------------- 结果聚合 -----------------
    
    total_success_count = 0
    total_collision_count = 0
    total_count = 0

    for success_c, collision_c, total_c in results:
        total_success_count += success_c
        total_collision_count += collision_c
        total_count += total_c

    print("--- 最终结果 ---")
    print(f"Total scenes tested: {total_count}")
    print(f"Success count: {total_success_count}")
    print(f"Collision count: {total_collision_count}")
    if total_count > 0:
        print(f"Success rate: {total_success_count/total_count}")
        print(f"Collision rate: {total_collision_count/total_count}")
    else:
        print("No scenes were successfully tested.")
    
    return

if __name__ == "__main__":
    import multiprocessing
    # 建议在主程序开始时设置 multiprocessing 的 start method
    # 'spawn' 更安全，但在 PyTorch/CUDA 环境中可能需要额外的代码来管理模型和数据传输
    # 在 Linux 上，'fork' 是默认且通常更快的，但可能导致资源泄露，尤其是在 PyTorch 中
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Start method 'spawn' could not be set, using default.")
        pass # 无法设置时跳过

    main()