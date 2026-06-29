from pathlib import Path
import numpy as np
import sys
import os

# Add parent directory to path to import from Robot-TrajectronV3
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from experiment.simulation import ClutterRemovalSim
import collections
import torch
import os
import matplotlib.pyplot as plt
State = collections.namedtuple("State", ["tsdf", "pc"])
import torch.distributions as td
from experiment.pseudo_users import *
from trajectron_node_noros2 import TrajectronNode
import argparse
from utils_exp import (
    Rotation, Transform, SO3_R3, SO3, matrix_to_euler_angles, 
    quaternion_to_matrix, achieved, trimesh_to_open3d, grasp2mesh, NEO_SS, Logger, 
    fps_se3, set_random_seed, Grasp, read_df, read_mesh
)


from baselines.hindsight import RobotAssistancePolicy, GoalAssistance as Goal, RobotState, Action, ApplyTwistToTransform

debug = True
method = 'teleop' # 'ho' or 'rt' or 'teleop'

def main(args):
    set_random_seed(42)
    global debug
    
    logger = Logger(log_dir=args.log_dir)
    root = args.data_root
    df = read_df(Path(root))
    scene_list = list(set(df.loc[:,"scene_id"].to_numpy()))
    scene_list.sort()
    sim = ClutterRemovalSim(args.scene_type, args.scene_path, gui=debug, save_file_name=None)
    T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05]).as_matrix()
    T_grasp_pregrasp = torch.tensor(T_grasp_pregrasp, dtype=torch.float32).cuda()
    dt = 0.05


    #### metrics ####
    cur_idx = 0
    count = 0
    success_count = 0
    collision_count = 0
    

    #### define model ####
    if method == 'rt':
        trajectron = TrajectronNode(checkpoint_path=args.checkpoint, 
                                    ood_alpha=args.ood_alpha, 
                                    history_size=args.history_size,
                                    sigma_coff=args.sigma_coff)

    #### define user ####
    if args.user == 'noisy':
        user = NoisyUser(sigma=0.05)
    elif args.user == 'laggy':
        user = LaggyUser()
    elif args.user == 'lowdof':
        user = LowDofUser()
    elif args.user == 'singledof':
        user = SingleDofUser(steps_per_dof=10)
    elif args.user == 'modeswitching':
        user = ModeSwitchingUser(steps_per_mode=20)

    #### define controller ####
    controller = NEO_SS()
    
    if 'ycb' in root:
        idx_list = range(len(scene_list))
    else:
        idx_list = np.random.choice(len(scene_list), 50)
    # print(idx_list)
    for cur_idx in idx_list:
        torch.manual_seed(cur_idx)
    # while cur_idx < len(scene_list):
        # print(cur_idx)
        scene_id = scene_list[cur_idx]
        # cur_idx += 1
        mesh_pose_list = read_mesh(Path(root), scene_id)
        sim.recovered_scene(mesh_pose_list)
        sim.save_state()

        sim.robot.add_robot(random=False)
        
        T_base_task = torch.tensor(sim.robot.T_base_task.as_matrix(), dtype=torch.float).cuda()
        

        ## load pcl
        tsdf, pc, timing = sim.acquire_tsdf(1, 1, 40)
        pcl_ = np.asarray(pc.points)
        pcl_ = torch.tensor(pcl_, dtype=torch.float32).cuda()
        pcl_ = T_base_task @ torch.cat((pcl_, torch.ones_like(pcl_[:,:1])), dim=-1).unsqueeze(-1)
        
        ### load grasps
        scene_mask = df.loc[:, "scene_id"] == scene_id
        label_array = df.loc[:, "label"].to_numpy(np.single)
        pos_mask = (label_array>0.4)
        mask = scene_mask & pos_mask
        scene_data = df.loc[mask, "qx":"label"].to_numpy(np.single)

        grasp_mesh_list = []
        for j in range(len(scene_data)):
            label = scene_data[j,-1]
            ori = scene_data[j, :4]
            pos = scene_data[j, 4:7]
            g = Grasp(Transform(Rotation.from_quat(ori), pos), 0.08)
            grasp_mesh_list.append(trimesh_to_open3d(grasp2mesh(g, label)))
        
        rotations = quaternion_to_matrix(torch.tensor(scene_data[:, :4], dtype=torch.float).cuda())
        translations = torch.tensor(scene_data[:, 4:7], dtype=torch.float).cuda()

        grasps_viz = SO3_R3(rotations, translations).to_matrix()
        grasps = T_base_task @ grasps_viz @ T_grasp_pregrasp
        

        for grasp_idx in range(grasps.shape[0]):
            cur_iter = 0
            
            user.reset()
            user.set_goal(grasps[grasp_idx])
            if grasp_idx != 0:
                sim.recovered_scene(mesh_pose_list)
                sim.save_state()
                sim.robot.add_robot(random=False)
            
            if method == 'rt':
                trajectron.reset()
                trajectron.update_state({
                    "grasps": grasps,
                    "pcl": pcl_
                })

            if method == 'ho':
                goals = [Goal(i, g, None) for i, g in enumerate(grasps.cpu().numpy())]
                robot_state = RobotState(sim.robot.get_ee_pose())
                print_on_file = False
                file = None
                ho = RobotAssistancePolicy(goals, robot_state, print_on_file, file)
                belief_distributions = []


            while True:
                # update current state
                cur_js, _ = sim.robot.read_joint_state()
                cur_pos = sim.robot.get_ee_pose()
                command = user.provide_command(torch.tensor(cur_pos, dtype=torch.float32).cuda())

                if method == 'rt':
                    trajectron.update_state({
                        "js": cur_js,
                        "pose": cur_pos,
                    })
                    trajectron.update_user_model(command)
                    trajectron.last_action = None
                    trajectron.trajectory_prediction(None)
                    action = trajectron.last_action if trajectron.last_action is not None else command[None]
                
                q0 = sim.robot.read_joint_state()[0]

                ## velo control ###
                if method == 'rt':
                    action = SO3_R3().exp_map(action*dt).to_matrix().cpu().numpy()
                    next_pose = cur_pos @ action[0]
                    next_pose = Transform.from_matrix(next_pose).as_matrix()
                    params = {'di':0.4,'ds':0.05,'xi':1.0}
                    j_vel = None
                    for _attempt in range(3):
                        try:
                            j_vel, arrived = controller.calculate_velocity_ss(sim.robot.panda_control, q0, next_pose, Lambda=0.1, Gain=15, params=params,)
                            break
                        except:
                            params['ds'] -= 0.01
                    if j_vel is None:
                        j_vel = np.zeros(7)
                
                if method == 'teleop':
                    action = SO3_R3().exp_map(command.unsqueeze(0)*dt).to_matrix().cpu().numpy()
                    next_pose = cur_pos @ action[0]
                    next_pose = Transform.from_matrix(next_pose).as_matrix()
                    params = {'di':0.4,'ds':0.05,'xi':1.0}
                    j_vel = None
                    for _attempt in range(3):
                        try:
                            j_vel, arrived = controller.calculate_velocity_ss(sim.robot.panda_control, q0, next_pose, Gain=15)
                            break
                        except:
                            params['ds'] -= 0.01
                    if j_vel is None:
                        j_vel = np.zeros(7)
                

                if method == 'ho':
                    command[:3] = torch.tensor(cur_pos[:3,:3], dtype=torch.float32).cuda() @ command[:3]
                    command[3:6] = torch.tensor(cur_pos[:3,:3], dtype=torch.float32).cuda() @ command[3:6]
                    command = command.cpu().numpy()
                    ho.update(sim.robot.get_ee_pose(), user_action=Action(twist=command))
                    if debug:
                        ho.visualize_prob()
                    belief_distribution = ho._goal_predictor.get_distribution()
                    belief_distributions.append(belief_distribution)
                    predict_command = ho.get_action(fix_magnitude_user_command=False).getTwist()
                    predict_pose = ApplyTwistToTransform(predict_command, sim.robot.get_ee_pose(), 0.01)
                    q0 = sim.robot.read_joint_state()[0]
                    params = {'di':0.4,'ds':0.05,'xi':1.0}
                    for _attempt in range(3):
                        try:
                            j_vel = controller.velocity_based_control(sim.robot.panda_control, q0, predict_command[:3] * dt, predict_command[3:6]* dt, Lambda=0.1, Gain=15, params=params)
                            break
                        except:
                            params['ds'] -= 0.01
                    if j_vel is None:
                        j_vel = np.zeros(7)
                sim.robot.set_joint_control(j_vel, 'velocity')
                sim.robot.step()

                cur_iter += 1
                cur_pos = sim.robot.get_ee_pose()

                success, _ = achieved(torch.tensor(cur_pos, dtype=torch.float32).to(grasps[grasp_idx].device), grasps[grasp_idx].unsqueeze(0), threshold=0.1)
                if success:
                    count += 1
                    success_count += 1
                    if debug:
                        print("Achieved goal")
                    logger.iteration_logs.append(cur_iter)
                    break
                    
                if cur_iter >= 150:
                    count += 1
                    if debug:
                        print("Cannot reach goal, reset scene")
                    logger.iteration_logs.append(cur_iter)
                    break

                collision = sim.robot.detect_contact()
                if collision:
                    if debug:
                        print("Collision detected, reset scene")
                    collision_count += 1
                    count += 1
                    break
    
    print("ood_alpha:{}, history_size:{}, sigma_coff:{}".format(args.ood_alpha, args.history_size, args.sigma_coff))
    print("method:{}, user:{}".format(method, args.user))
    print("success rate:", success_count/count)
    print("collision rate:", collision_count/count)
    print("average iteration:", sum(logger.iteration_logs)/len(logger.iteration_logs) if logger.iteration_logs else 0)
    print('total count:', count)
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated shared control benchmark")
    
    # Method and user configuration
    parser.add_argument("--method", type=str, default="teleop", 
                        choices=["rt", "ho", "teleop"],
                        help="Control method: rt (trajectron), ho (hindsight), teleop")
    parser.add_argument("--user", type=str, default="singledof", 
                        choices=["noisy", "laggy", "lowdof", "singledof", "modeswitching"],
                        help="User type for simulation")
    
    # Model parameters
    parser.add_argument("--ood_alpha", type=float, default=0.9,
                        help="Out-of-distribution detection alpha parameter")
    parser.add_argument("--history_size", type=int, default=6,
                        help="History size for trajectory prediction")
    parser.add_argument("--sigma_coff", type=float, default=1.0,
                        help="Sigma coefficient for uncertainty")
    
    # Path configuration
    parser.add_argument("--data_root", type=str, default="data/data_packed_train_raw",
                        help="Root directory for scene data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/line24.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--log_dir", type=str, default="./simulated_shared_benchmark_logs",
                        help="Directory for logging outputs")
    
    # Scene configuration
    parser.add_argument("--scene_type", type=str, default="pile",
                        help="Type of scene for simulation")
    parser.add_argument("--scene_path", type=str, default="pile/train",
                        help="Path to scene configuration")
    
    # Debug mode
    parser.add_argument("--no-debug", action='store_true', 
                        help="Disable debug mode for additional output")

    args = parser.parse_args()
    if args.no_debug:
        debug = False
    method = args.method
    main(args)