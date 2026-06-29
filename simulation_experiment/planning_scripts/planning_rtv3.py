from utils_exp.noise import set_random_seed
set_random_seed(42)
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




debug = True
relative = True
Viz = True

def main():
    global debug
    root = "data/data_packed_train_raw"
    df = read_df(Path(root))
    scene_list = list(set(df.loc[:,"scene_id"].to_numpy()))
    scene_list.sort()
    sim = ClutterRemovalSim("pile", "pile/train", gui=debug, save_file_name=None)
    T_body_tcp = Transform.from_dict({"rotation": [0.000, 0.000, 0.0, 1.0], "translation": [0.000, 0.000, 0.05]}).as_matrix()
    T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05]).as_matrix()

    controller = NEO_SS()

    count = 0
    success_count = 0
    collision_count = 0
    cur_idx = 0
    num_test_scenes = 100
    trajectron = init_model()
    while cur_idx < num_test_scenes:
        torch.manual_seed(cur_idx)
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

        sim.robot.add_robot(random=True)
        
        T_base_task = torch.tensor(sim.robot.T_base_task.as_matrix(), dtype=torch.float)
        T_task_base = torch.inverse(T_base_task)
        dt = 0.05

        ## load pcl
        tsdf, pc, timing = sim.acquire_tsdf(1, 1, 40)
        # visualize_point_cloud_with_normals(np.asarray(pc.points))


        # sim.robot.add_obstacles(mesh, mesh_filenames)
        # o3d.visualization.draw_geometries(meshes)
        pcl_ = np.asarray(pc.points)
        pcl_ = torch.tensor(pcl_, dtype=torch.float32)
        pcl_ = T_base_task @ torch.cat((pcl_,torch.ones_like(pcl_[:,:1])), dim=-1).unsqueeze(-1)

        scene_mask = df.loc[:, "scene_id"] == scene_id
        label_array = df.loc[:, "label"].to_numpy(np.single)#.astype(np.int64)
        pos_mask = (label_array>0.4)
        # print("pos vs neg:", pos_mask.sum(), (~pos_mask).sum())
        mask = scene_mask & pos_mask
        grasps = df.loc[mask, "qx":"label"].to_numpy(np.single)
        rotations = quaternion_to_matrix(torch.tensor(grasps[:, :4], dtype=torch.float))
        translations = torch.tensor(grasps[:, 4:7], dtype=torch.float)

        grasps_viz = SO3_R3(rotations, translations).to_matrix()
        grasps_viz = grasps_viz @ torch.tensor(T_body_tcp, dtype=torch.float) @ torch.tensor(T_grasp_pregrasp, dtype=torch.float)


        js_traj_ori = []
        ee_traj_ori = []
        cur_js, _ = sim.robot.read_joint_state()
        cur_pos = sim.robot.get_ee_pose()
        for _ in range(7):
            ee_traj_ori.append(cur_pos)
            js_traj_ori.append(cur_js)

        ee_traj_ori = torch.tensor(np.stack(ee_traj_ori, axis=0), dtype=torch.float32)
        js_traj_ori = torch.tensor(np.stack(js_traj_ori, axis=0), dtype=torch.float32)
        # ee_traj_logmap = SO3_R3.from_matrix(ee_traj_ori).log_map()

        # ee_traj = torch.cat((ee_traj_logmap, ee_vel_traj), dim=-1)  # [timestep, 7]
        curves = None
        next_action = None
        flag = 0
        shuffle_idx = torch.randperm(grasps.shape[0])
        # print(shuffle_idx[:2])

        while True:
            # update current state
            cur_js, _ = sim.robot.read_joint_state()
            # if next_action is None:
            cur_pos = sim.robot.get_ee_pose()
            # else:
                # cur_pos = next_action


            ee_traj_ori = torch.cat((ee_traj_ori, torch.tensor(cur_pos, dtype=torch.float32).unsqueeze(0)), dim=0)
            js_traj_ori = torch.cat((js_traj_ori, torch.tensor(cur_js, dtype=torch.float32).unsqueeze(0)),dim=0)

            ee_vel_traj = se3_derivatives_of(ee_traj_ori[-9:], dt=dt)[-8:]
            js_vel_traj = js_derivatives_of(js_traj_ori[-9:], dt=dt)[-8:]


            first_history_index = torch.LongTensor(np.array([0])).cuda()
            T_curpose_base = torch.inverse(ee_traj_ori[-1])
            if relative:
                ee_traj_rel = T_curpose_base @ ee_traj_ori[-8:]
            else:
                ee_traj_rel = ee_traj_ori[-8:]
            ee_traj_logmap = SO3_R3.from_matrix(ee_traj_rel).log_map()
            ee_traj = torch.cat((ee_traj_logmap, ee_vel_traj), dim=-1) 
            js_traj = torch.cat((js_traj_ori[-8:], js_vel_traj), dim=-1)  # [timestep, 7]

            ph = 12
            x = ee_traj[-8:,:].unsqueeze(0).cuda()
            q = js_traj[-8:,:].unsqueeze(0).cuda()
            y = torch.zeros(1, ph, x.shape[-1]).cuda()
            # ph = data.shape[0]-(j+8)

            
            dim = x.shape[1]

            grasps_data = T_base_task @ grasps_viz

            if relative:
                grasps_data = T_curpose_base @ grasps_data
                pcl = T_curpose_base @ pcl_
            else:
                pcl = pcl_
            pcl = pcl[:,:3,0]

            grasps_data = select_grasps(grasps_data, T=ee_traj_ori[-1] if not relative else None)
            grasps_data_ = grasps_data[shuffle_idx][:2] #

            # if flag == 0:
            #     grasp_mesh_list = []
            #     g = Grasp(Transform.from_matrix(grasps_viz[shuffle_idx][0].numpy()), 0.08)
            #     grasp_mesh_list.append(trimesh_to_open3d(grasp2mesh(g, 1)))
            #     scene = trimesh_to_open3d(get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False))
            #     o3d.visualization.draw_geometries([scene, ]+ grasp_mesh_list,
            #                                         window_name="Point Cloud with Normals",
            #                                         point_show_normal=True)
            #     flag = 1

            
            grasps_data = SO3_R3.from_matrix(grasps_data_).log_map()
            # rotation_6d = matrix_to_rotation_6d(grasps[:,:3,:3])
            # grasps_data = torch.cat((grasps[:,:3,3], rotation_6d), dim=-1)  # [timestep, 9]

            context = {
                'grasp': [grasps_data.cuda()],
                'pcl': [pcl.cuda()]
            }

            
            
            batch = (first_history_index, x, q, y, context)
            with torch.no_grad():
                ################# most likely ##############################
                y_dist, a_dist, predictions = trajectron.predict(batch,
                                        ph=12,
                                        num_samples=1, # doesn't matter when all_z_sep is true
                                        z_mode=True,
                                        gmm_mode=True,
                                        all_z_sep=False,
                                        full_dist=False,
                                        dist=True)
            next_action = predictions[0,0,0] # relative pose
            ### viz ###
            if Viz:
                mode_score = np.exp(trajectron.model.latent.p_dist.logits.detach().cpu().numpy()).reshape(-1)
                
                if curves is not None:
                    if type(curves) == list:
                        for c in curves:
                            c.remove()
                    else:
                        curves.remove()
                vis_data = ee_traj_ori[-2:]
                if flag == 0:
                    plot_se3_poses(T_task_base@vis_data, other_poses=grasps_viz[shuffle_idx][:2], ax=ax, color='green', draw_frame_flag=False) # 
                    flag = 1
                else:
                    plot_se3_poses(T_task_base@vis_data, ax=ax, axis_scale=0.01, color='green',draw_frame_flag=False)

                curves = []
                predictions = np.concatenate((np.eye(4)[None, None, None, :, :], predictions), axis=2)
                for s in range(predictions.shape[0]):
                    if relative:
                        pred_traj_viz = plot_se3_poses(T_task_base@ee_traj_ori[-1]@predictions[s].reshape(-1,4,4), ax=ax, alpha=mode_score.max(), color='red') # 
                    else:
                        pred_traj_viz = plot_se3_poses(T_task_base@predictions[s].reshape(-1,4,4), ax=ax, alpha=mode_score.max(), color='red') # 
                    curves.extend(pred_traj_viz)
                

            
            
            
            if relative:
                next_action = ee_traj_ori[-1].numpy() @ next_action
                success, idx = achieved(torch.tensor(next_action, dtype=torch.float32), ee_traj_ori[-1] @  grasps_data_)
            else:
                success, idx = achieved(torch.tensor(next_action, dtype=torch.float32), grasps_data_)
            
            
            # target_viz = ax.scatter(grasps_viz[idx,0,3], grasps_viz[idx,1,3], grasps_viz[idx,2,3], s=50, color='orange', marker="*")
            # curves.append(target_viz)

            # rel_action = SO3_R3().exp_map(a_dist.mode()[0,0,:1]).to_matrix().cpu()[0]
            # T_task_cur = T_task_base @ ee_traj_ori[-1]
            # abs_action = T_task_cur[:3, :3] @ rel_action[:3,3]
            # ax.quiver(T_task_cur[0,3], T_task_cur[1,3],T_task_cur[2,3],abs_action[0], abs_action[1], abs_action[2], length=0.1, color='black')
            if Viz:
                plt.pause(0.01)
                plt.ioff()
            
            ### viz ###
            if success:
                count += 1
                success_count += 1
                if debug:
                    print("Achieved goal")
                if Viz:
                    plt.close(fig)
                break
            if len(ee_traj_ori) > 150:
                count += 1
                if debug:
                    print("Cannot reach goal, reset scene")
                if Viz:
                    plt.close(fig)
                break
            q0 = sim.robot.read_joint_state()[0]
            # next_action = predictions[0,0,0] # relative pose
            # if relative:
            #     next_action = ee_traj_ori[-1].numpy() @ next_action
            next_action = Transform.from_matrix(next_action).as_matrix()
            # ik_results = ik_solver.solve_batch_ik(np.expand_dims(next_action, 0), initial_joints=q0.reshape(1,7).repeat(12, axis=0).reshape(1,12,7))
            # solution = ik_results['solution'][0,0]
            
            ## velo control ###
            # action = a_dist.mode()[0,0,0].cpu().numpy()
            # pos_vel = action[0, :3, 3].cpu().numpy()
            # ang_vel = matrix_to_euler_angles(action[:,:3,:3], 'ZYX').cpu().numpy()[0][::-1]
            # print('pos_vel:', pos_vel, 'ang_vel:', ang_vel)
            # pos_vel = next_action[:3, 3]#.cpu().numpy()  #*2
            # ang_vel = matrix_to_euler_angles(torch.tensor(next_action[:3,:3],dtype=torch.float32).unsqueeze(0), 'ZYX').cpu().numpy()[0][::-1] # action[0, :3, 3].cpu().numpy()
            # j_vel = velocity_based_control(sim.robot.panda_control, q0, action[:3], action[3:], Gain=1, onbase=False,) #  obstacles=sim.robot.obstacles+obstacles
            # j_pos = sim.robot.panda_control.ik_LM(next_action,q0=q0)[0]
            # env.step(0.01)

            ### velo control ###
            # j_vel = (solution.cpu().numpy()-q0)/dt

            j_vel, arrived = calculate_velocity(sim.robot.panda_control, q0, next_action, Gain=15, ) # obstacles=sim.robot.obstacles+obstacles
            # try:
            # j_vel, arrived = controller.calculate_velocity_ss(sim.robot.panda_control, q0, next_action, Gain=15, ) # pcl=pcl_[:,:3,0].cpu().numpy()
            # except:
            #     j_vel = np.zeros(7)
            
            # sim.robot.set_joint_control(solution.cpu().numpy(), 'position')
            sim.robot.set_joint_control(j_vel, 'velocity')
            sim.robot.step()
            # env.step()
            collision = sim.robot.detect_contact()
            if collision:
                if debug:
                    print("Collision detected, reset scene")
                collision_count += 1
                count += 1
                break
            # time.sleep(0.05)
    print("success rate:", success_count/count)
    print("collision rate:", collision_count/count)
    return

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

    args.checkpoint = "checkpoints/epoch10|20Hz|ade21.05.pth"
    print(f"Loading model from {args.checkpoint}")
    trajectron = Trajectron(hyperparams, args.device)
    model = torch.load(args.checkpoint)
    trajectron.model.node_modules = model
    trajectron.set_annealing_params()
    max_hl = hyperparams['maximum_history_length']
    ph = hyperparams['prediction_horizon']
    trajectron.model.to(args.device)
    trajectron.model.eval()
    return trajectron



if __name__ == "__main__":
    main()