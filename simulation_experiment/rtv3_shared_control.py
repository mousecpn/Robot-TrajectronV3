# Add these imports at the top of your existing file
import rclpy
from utils_exp.viz_node import TrajectoryVisualizationNode
import threading

import argparse
from pathlib import Path

import numpy as np

from utils_exp.io import *
from utils_exp.perception import *
from experiment.simulation import ClutterRemovalSim
from utils_exp.transform import Rotation, Transform
import collections
from utils_exp.implicit import get_scene_from_mesh_pose_list
from utils_exp.visual import trimesh_to_open3d,grasp2mesh, pointcloud_to_meshes
import sys
sys.path.append("/home/u0161364/Robot-TrajectronV3")
import torch
import os
from model.trajectron import Trajectron
from argument_parser import args
from model.transform import SO3_R3, SO3, matrix_to_euler_angles, quaternion_to_matrix, matrix_to_rotation_6d, euler_angles_to_matrix, select_grasps
from dataset.se3_preprocessing import se3_derivatives_of, js_derivatives_of,  plot_se3_poses
import matplotlib.pyplot as plt
from utils_exp.control import calculate_velocity, velocity_based_control
State = collections.namedtuple("State", ["tsdf", "pc"])
from keyboard_control_scripts.teleoperation import keyboard_detection
from threading import Thread, Lock
import torch.distributions as td
import time
import spatialgeometry as sg
from trajectron_node import TrajectronNode
from utils_exp.fps_se3 import fps_se3
debug = True
relative = True
twist_msg = np.zeros((8))

gripper_flip = torch.eye(4, dtype=torch.float32)
gripper_flip[:3,:3] = torch.tensor(Rotation.from_euler("z", np.pi).as_matrix(), dtype=torch.float32)

def keyboard_detection(p, velo=0.2):
    global twist_msg
    while True:
        #Create zero twist message
        twist_msg = np.zeros((8))
        g = p.getKeyboardEvents()

        # if ord('x') in g:
        #     command_msg.command = 5
        # elif 32 in g:
        #     command_msg.command = 6
        # else:
        #     command_msg.command = command_msg.TWIST
        # if len(g.keys()) == 0:
        #     continue
        if p.B3G_UP_ARROW in g:
            twist_msg[0]= -velo
        
        if p.B3G_LEFT_ARROW in g:
            twist_msg[1]= -velo
        
        if p.B3G_DOWN_ARROW in g:
            twist_msg[0] = velo
        
        if p.B3G_RIGHT_ARROW in g:
            twist_msg[1] = velo
        
        if ord('z') in g:
            twist_msg[2] = velo
        
        if ord('x') in g:
            twist_msg[2] = -velo
        

        if ord('j') in g:
            twist_msg[3] = velo*2
        
        if ord('l') in g:
            twist_msg[3] = -velo*2


        if ord('i') in g:
            twist_msg[4] = velo*2
        
        if ord('k') in g:
            twist_msg[4] = -velo*2
        
        if ord('u') in g:
            twist_msg[5] = velo*2
        
        if ord('o') in g:
            twist_msg[5] = -velo*2
        
        if p.B3G_SPACE in g:
            twist_msg[6] = 1

        if ord('m') in g:
            twist_msg[6] = -1

        if ord('h') in g:
            twist_msg[7] = 1
        
        # print(twist_msg)

        time.sleep(0.01)

# Add this global variable for the ROS2 node
ros_viz_node = None
trajectron_node = None

from rclpy.executors import MultiThreadedExecutor

def init_ros2_visualization():
    """Initialize ROS2 visualization nodes in separate thread"""
    global ros_viz_node
    global trajectron_node

    def ros_spin():
        rclpy.init()
        global ros_viz_node, trajectron_node

        ros_viz_node = TrajectoryVisualizationNode()
        trajectron_node = TrajectronNode(checkpoint_path="/home/u0161364/Robot-TrajectronV3/checkpoints/line24.pth")

        executor = MultiThreadedExecutor(4)
        executor.add_node(ros_viz_node)
        executor.add_node(trajectron_node)

        try:
            print("Spinning both nodes...")
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            ros_viz_node.destroy_node()
            trajectron_node.destroy_node()
            rclpy.shutdown()
            print("Shut down ROS 2")

    ros_thread = threading.Thread(target=ros_spin, daemon=True)
    ros_thread.start()
    return ros_thread

def main():
    global debug
    global twist_msg
    global ros_viz_node
    global trajectron_node
    
    # Initialize ROS2 visualization
    ros_thread = init_ros2_visualization()
    
    # Wait a moment for ROS2 node to initialize
    import time
    time.sleep(1)
    
    root = "data/ycb_scene_packed"
    df = read_df(Path(root))
    scene_list = list(set(df.loc[:,"scene_id"].to_numpy()))
    scene_list.sort()
    sim = ClutterRemovalSim("pile", "pile/train", gui=debug, save_file_name=None)
    T_body_tcp = Transform.from_dict({"rotation": [0.000, 0.000, 0.0, 1.0], "translation": [0.000, 0.000, 0.05]}).as_matrix()
    T_body_tcp = torch.tensor(T_body_tcp, dtype=torch.float32).cuda()
    T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05]).as_matrix()
    T_grasp_pregrasp = torch.tensor(T_grasp_pregrasp, dtype=torch.float32).cuda()
    t0 = Thread(target=keyboard_detection,name='keyboard_detection', args=(sim.world.p,))
    t0.start()

    

    traj_count = 0
    cur_idx = 0
    while cur_idx < len(scene_list):
        # cur_idx = np.random.randint(0, len(scene_list))
        scene_id = scene_list[cur_idx]
        cur_idx += 1

        mesh_pose_list = read_mesh(Path(root), scene_id)
        sim.recovered_scene(mesh_pose_list)
        sim.save_state()

        ## load pcl
        tsdf, pc, timing = sim.acquire_tsdf(1, 1, 40)

        sim.robot.add_robot()
        
        T_base_task = torch.tensor(sim.robot.T_base_task.as_matrix(), dtype=torch.float).cuda()
        T_task_base = torch.inverse(T_base_task)
        dt = 0.1

        ros_viz_node.clear()
        trajectron_node.reset()


        pcl_ = np.asarray(pc.points)
        pcl_ = torch.tensor(pcl_, dtype=torch.float32).cuda()
        pcl_ = T_base_task @ torch.cat((pcl_,torch.ones_like(pcl_[:,:1])), dim=-1).unsqueeze(-1)
        
        # # Update pointcloud visualization
        if ros_viz_node is not None:
            ros_viz_node.update_pointcloud(pcl_[:,:3,0].cpu().numpy())
        


        scene_mask = df.loc[:, "scene_id"] == scene_id
        label_array = df.loc[:, "label"].to_numpy(np.single)
        pos_mask = (label_array>0.4)
        mask = scene_mask & pos_mask
        scene_data = df.loc[mask, "qx":"label"].to_numpy(np.single)[:,:7]


        # js_traj_ori = []
        # ee_traj_ori = []
        # cur_js, _ = sim.robot.read_joint_state()
        # cur_pos = sim.robot.get_ee_pose()
        # for _ in range(7):
        #     ee_traj_ori.append(cur_pos)
        #     js_traj_ori.append(cur_js)

        # ee_traj_ori = torch.tensor(np.stack(ee_traj_ori, axis=0), dtype=torch.float32).cuda()
        # js_traj_ori = torch.tensor(np.stack(js_traj_ori, axis=0), dtype=torch.float32).cuda()

        curves = None
        next_action = None
        flag = 0
        grasp_mesh_list = []

        
        # grasps
        rotations = quaternion_to_matrix(torch.tensor(scene_data[:, :4], dtype=torch.float).cuda())
        translations = torch.tensor(scene_data[:, 4:7], dtype=torch.float).cuda()

        grasps_viz = SO3_R3(rotations, translations).to_matrix()
        selected_indices = fps_se3(grasps_viz, k=20)
        grasps_viz = grasps_viz[selected_indices]

        for j in selected_indices:
            label = scene_data[j,-1]
            ori = scene_data[j, :4]
            pos = scene_data[j, 4:7]
            g = Grasp(Transform(Rotation.from_quat(ori), pos), 0.08)
            grasp_mesh_list.append(trimesh_to_open3d(grasp2mesh(g, label)))
        # grasps_viz = grasps_viz @ T_body_tcp @ T_grasp_pregrasp
        grasps = T_base_task @ grasps_viz 

        

        trajectron_node.update_state({
            "grasps": grasps,
            "pcl": pcl_
        })

        # Update grasp poses visualization
        if ros_viz_node is not None:
            grasp_scores = scene_data[:, -1]  # Use the label as score
            ros_viz_node.update_grasp_poses((T_base_task@grasps_viz).cpu().numpy(), grasp_scores)
        

        last_js_velo = np.zeros((7,))
        
        while True:
            if twist_msg[6] == 1:
                cur_pos = sim.robot.get_ee_pose()
                # grasps_viz = SO3_R3(rotations, translations).to_matrix()
                # grasps_viz = grasps_viz 


                _, target_grasp = achieved(torch.tensor(cur_pos, dtype=torch.float).cuda(), grasps)
                # target_grasp = target_grasp @ T_grasp_pregrasp @ T_grasp_pregrasp
                target_grasp = T_task_base @ target_grasp
                target_grasp_matrix = target_grasp.cpu().numpy()
                target_grasp_transform = Transform.from_matrix(target_grasp_matrix)
                g = Grasp(target_grasp_transform, 0.08)
                
                # # Update target grasp visualization
                # if ros_viz_node is not None:
                #     ros_viz_node.update_target_grasp(target_grasp_matrix)

                sim.execute_grasp_robot(g, pc=np.asarray(pc.points))
                
                # sim.robot.move_robot_instant(target_grasp_transform, obstacles=obstacles)
                # sim.robot.move_robot_instant(target_grasp_transform * Transform.from_matrix((T_grasp_pregrasp@T_grasp_pregrasp).cpu().numpy()).inverse())
                # sim.robot.grasp_robot()
                
            elif twist_msg[6] == -1:
                sim.robot.gripper_homing()
            elif twist_msg[7] == 1:
                scene = trimesh_to_open3d(get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False))
                o3d.visualization.draw_geometries([scene, ]+ grasp_mesh_list,
                                                    window_name="Point Cloud with Normals",
                                                    point_show_normal=True)
                break
            elif np.linalg.norm(twist_msg[:6]) > 0.1:
                t1 = time.time()
                command = twist_msg[:6]
                # update current state
                cur_js, _ = sim.robot.read_joint_state()
                cur_pos = sim.robot.get_ee_pose()

                # Update current EE pose visualization
                # if ros_viz_node is not None:
                #     ros_viz_node.update_current_ee_pose(cur_pos)


                # feedforward
                pseudo_js = cur_js#+dt*last_js_velo
                pseudo_cur_pos = sim.robot.panda_control.fkine(pseudo_js).A
                
                trajectron_node.update_state({
                    "js": pseudo_js,
                    "pose": pseudo_cur_pos,
                })
                t2 = time.time()

                # Update current trajectory visualization
                if ros_viz_node is not None and trajectron_node.pose_traj is not None:
                    ros_viz_node.update_current_trajectory(trajectron_node.pose_traj[-20:].cpu().numpy())  # Show last 20 poses
                

                user_velo = torch.tensor(np.array(command), dtype=torch.float32)
                # if relative:
                user_velo[:3] = torch.inverse(torch.tensor(pseudo_cur_pos[:3,:3], dtype=torch.float32)) @ user_velo[:3]
                
                # user_velo[3:6] = SO3(tensor=euler_angles_to_matrix(user_velo[3:6].flip(0), 'ZYX').unsqueeze(0)).log_map().reshape(-1)
                # trajectron_node.user_model = td.MultivariateNormal(user_velo.reshape(-1,6), user_sigma.clone())
                trajectron_node.update_user_model(user_velo.cuda())
                # trajectron_node.last_action = None
                trajectron_node.trajectory_prediction_asyn()
                t3 = time.time()
                # Update predicted trajectory visualization
                if ros_viz_node is not None and trajectron_node.predictions is not None:
                    # Convert predictions to world frame for visualization
                    current_pose_matrix = trajectron_node.pose_traj[-1].cpu().numpy()
                    ros_viz_node.update_predicted_trajectory(trajectron_node.predictions.reshape(-1,trajectron_node.ph, 4, 4), current_pose_matrix, relative=True)

                q0 = sim.robot.read_joint_state()[0]

                
                if trajectron_node.op_count >= 4 and trajectron_node.last_action is not None:
                    # while trajectron_node.last_action is None:
                    #     print('Waiting for model inference...')
                    #     time.sleep(0.01)
                    ## velo control ###
                    # velo_action = a_dist.get_at_time(0).mode()[0,0,:1]
                    action = SO3_R3().exp_map(trajectron_node.last_action*0.05).to_matrix().cpu().numpy()
                    next_action = trajectron_node.pose_traj[-1].cpu().numpy() @ action[0]
                    # next_action = trajectron_node.pose_traj[-1].cpu().numpy() @ trajectron_node.last_action
                    next_action = Transform.from_matrix(next_action).as_matrix()
                    j_vel, arrived = calculate_velocity(sim.robot.panda_control, q0, next_action, Gain=15, obstacles=None) # sim.robot.obstacles+obstacles
                else:
                    j_vel = velocity_based_control(sim.robot.panda_control, q0, command[:3], command[3:6], Gain=1, obstacles=None) # sim.robot.obstacles+obstacles
                last_js_velo = np.array(j_vel)
                t5 = time.time()
                sim.robot.set_joint_control(j_vel, 'velocity')
                sim.robot.step(2)
                t6 = time.time()
                
                print("preprocessing:", t2-t1)
                print("model inference:", t3-t2)
                # print("postprocessing:", t6-t3)
                # print("mode:", t4-t3)
                # print("control:", t5-t4)
                print("sim step:", t6-t5)
                print("loop time:", t6-t1)
            else:
                try:
                    j_vel = j_vel * 0.1
                    sim.robot.set_joint_control(j_vel, 'velocity')
                except:
                    pass
                sim.robot.step()
                last_velo = np.zeros((6,))
                last_js_velo = np.zeros((7,))

    return

def init_model():
    torch.cuda.set_device('cuda:0')

    # Load hyperparameters from json

    with open("/home/u0161364/Robot-TrajectronV3/config/config.json", 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)
    

    # Add hyperparams from arguments
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval
    hyperparams['pcl_encoding'] = True
    hyperparams['frequency'] = 20

    # args.checkpoint = "/home/u0161364/Robot-TrajectronV3/checkpoints/lin8_pcldecoder.pth"
    args.checkpoint = "/home/u0161364/Robot-TrajectronV3/checkpoints/line9.pth"
    trajectron = Trajectron(hyperparams, args.device)
    model = torch.load(args.checkpoint)
    trajectron.model.node_modules = model
    trajectron.set_annealing_params()
    max_hl = hyperparams['maximum_history_length']
    ph = hyperparams['prediction_horizon']
    trajectron.model.to(args.device)
    trajectron.model.eval()
    return trajectron



def achieved(cur_pose, goal_poses):
    error_mat = torch.inverse(cur_pose) @ goal_poses
    e1 = torch.cat((error_mat[..., :3,3], matrix_to_euler_angles(error_mat[...,:3,:3], 'ZYX')/np.pi*0.6), dim=-1).abs().sum(-1)
    idx1 = torch.argmin(e1)

    error_mat = torch.inverse(cur_pose) @ (goal_poses @ gripper_flip.unsqueeze(0).to(goal_poses.device))
    e2 = torch.cat((error_mat[..., :3,3], matrix_to_euler_angles(error_mat[...,:3,:3], 'ZYX') /np.pi*0.6), dim=-1).abs().sum(-1)
    idx2 = torch.argmin(e2)

    if e1[idx1] < e2[idx2]:
        e = e1[idx1]
        target_pose = goal_poses[idx1]
    else:
        e = e2[idx2]
        target_pose = goal_poses[idx2] @ gripper_flip.to(goal_poses.device)

    if e < 0.01:
        return True, target_pose
    else:
        return False, target_pose


if __name__ == "__main__":
    main()