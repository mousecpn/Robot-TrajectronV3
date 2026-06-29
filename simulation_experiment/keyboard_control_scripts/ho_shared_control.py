from pathlib import Path

import numpy as np
import open3d as o3d
import scipy.signal as signal
from tqdm import tqdm
import multiprocessing as mp

from utils_exp.grasp import Grasp, Label
from utils_exp.io import *
from utils_exp.perception import *
from utils_exp.visual import grasp2mesh
from experiment.simulation import ClutterRemovalSim
from utils_exp.transform import Rotation, Transform
import collections
from utils_exp.implicit import get_scene_from_mesh_pose_list
from threading import Thread, Lock
from utils_exp.visual import trimesh_to_open3d

from baselines.hindsight.RobotAssistancePolicy import RobotAssistancePolicy
from baselines.hindsight.GoalAssistance import GoalAssistance as Goal
from baselines.hindsight.RobotState import RobotState, Action
from baselines.hindsight.Utils import ApplyTwistToTransform
from utils_exp.control import velocity_based_control

State = collections.namedtuple("State", ["tsdf", "pc"])

twist_msg = np.zeros((8))


def keyboard_detection(p, velo=0.1):
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
            twist_msg[3] = velo
        
        if ord('l') in g:
            twist_msg[3] = -velo


        if ord('i') in g:
            twist_msg[4] = velo
        
        if ord('k') in g:
            twist_msg[4] = -velo
        
        if ord('u') in g:
            twist_msg[5] = velo
        
        if ord('o') in g:
            twist_msg[5] = -velo
        
        if ord('h') in g:
            twist_msg[6] = 1

        if ord('m') in g:
            twist_msg[6] = -1
        
        if 32 in g: # Space key to stop
            twist_msg[7] = 1

        
        # print(twist_msg)

        time.sleep(0.01)

def main():
    global twist_msg
    root = "data/data_packed_train_raw"
    df = read_df(Path(root))
    scene_list = list(set(df.loc[:,"scene_id"].to_numpy()))
    sim = ClutterRemovalSim("pile", "pile/train", gui=True)
    t0 = Thread(target=keyboard_detection,name='keyboard_detection', args=(sim.world.p,))
    t0.start()
    

    for i in range(len(scene_list)):
        scene_id = scene_list[i]
        mesh_pose_list = read_mesh(Path(root), scene_id)
        sim.recovered_scene(mesh_pose_list)
        sim.robot.add_robot()
        sim.save_state()

        scene_mask = df.loc[:, "scene_id"] == scene_id
        label_array = df.loc[:, "label"].to_numpy(np.single)#.astype(np.int64)
        pos_mask = (label_array>0.4)
        mask = scene_mask & pos_mask
        label_array = label_array[mask]

        grasps = []
        grasp_mesh_list = []
        pos_array = df.loc[mask, "x":"z"].to_numpy(np.single)
        ori_array = df.loc[mask, "qx":"qw"].to_numpy(np.single)
        for i in range(ori_array.shape[0]):
            ori_term = ori_array[i]
            pos_term = pos_array[i]
            g = Grasp(Transform(rotation=Rotation.from_quat(ori_term), translation=pos_term),width=0.08)
            grasps.append(sim.robot.T_base_task*Transform(rotation=Rotation.from_quat(ori_term), translation=pos_term))
            # vis
            # add_triad(sim.world.p, -1, Transform(rotation=Rotation.from_quat(ori_term), translation=pos_term))
            # sim.execute_grasp(g,remove=False, allow_contact=True)
            # sim.restore_state()
            grasp_mesh_list.append(trimesh_to_open3d(grasp2mesh(g, 1)))
        scene = trimesh_to_open3d(get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False))
        o3d.visualization.draw_geometries([scene, ]+ grasp_mesh_list,
                                            window_name="Point Cloud with Normals",
                                            point_show_normal=True)

        goals = [Goal(i, g.as_matrix(), None) for i, g in enumerate(grasps)]
        robot_state = RobotState(sim.robot.get_ee_pose())
        print_on_file = False
        file = None
        assistance_policy = RobotAssistancePolicy(goals, robot_state, print_on_file, file)
        belief_distributions = []

        # get point cloud
        sim.robot.gripper_homing()
        _, pc, _ = sim.acquire_tsdf(n=1, N=1, resolution=40)
        pc = np.asarray(pc.points)
        pc = sim.robot.T_base_task.transform_point(pc)
        while True:
            if twist_msg[6] == 1:
                selected_grasp = assistance_policy.get_selected_goal().getCenterMatrix()
                selected_grasp = Transform(Rotation.from_matrix(selected_grasp[0:3,0:3]), selected_grasp[0:3,3])
                T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
                sim.robot.move_robot_instant(selected_grasp*T_grasp_pregrasp)#*sim.robot.T_tcp_body)
                sim.robot.move_robot_instant(selected_grasp)#*sim.robot.T_tcp_body)
                sim.robot.grasp_robot()
                approach = selected_grasp.rotation.as_matrix()[:, 2]
                angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
                if angle > np.pi / 3.0:
                    # side grasp, lift the object after establishing a grasp
                    T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
                    T_world_retreat = T_grasp_pregrasp_world * selected_grasp
                else:
                    T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
                    T_world_retreat = selected_grasp * T_grasp_retreat
                sim.robot.move_robot_instant(T_world_retreat)#*sim.robot.T_tcp_body)
            elif twist_msg[6] == -1:
                sim.robot.gripper_homing()
            if twist_msg[7] == 1:
                break
            if np.linalg.norm(twist_msg[:6]) > 0.01:
                assistance_policy.update(sim.robot.get_ee_pose(), user_action=Action(twist=twist_msg[:6]))
                assistance_policy.visualize_prob()
                belief_distribution = assistance_policy._goal_predictor.get_distribution()
                belief_distributions.append(belief_distribution)
                predict_command = assistance_policy.get_action(fix_magnitude_user_command=False).getTwist()
                print(predict_command)
                predict_pose = ApplyTwistToTransform(predict_command, sim.robot.get_ee_pose(), 0.01)
                ee_tr = Transform(Rotation.from_matrix(assistance_policy._robot_state.ee_trans[:3,:3]), assistance_policy._robot_state.ee_trans[:3,3])
                ee_tr_after = assistance_policy._assist_policy.goal_assist_policies[0].target_assist_policies[0].robot_state_after_action.ee_trans
                ee_tr_after = Transform(Rotation.from_matrix(ee_tr_after[:3,:3]), ee_tr_after[:3,3])
                # add_triad(sim.world.p, -1, sim.robot.T_base_task.inverse()*ee_tr)
                # add_triad(sim.world.p, -1, sim.robot.T_base_task.inverse()*ee_tr_after)
                # sim.robot.velo_control(twist_msg=predict_command)
                # sim.robot.velo_control(twist_msg=predict_command, obstacles=pc)
                q0 = sim.robot.read_joint_state()[0]
                j_vel = velocity_based_control(sim.robot.panda_control, q0, predict_command[:3], predict_command[3:6],  onbase=True)
                sim.robot.set_joint_control(j_vel, 'velocity')
                sim.robot.step()
                # sim.robot.move_robot_instant(Transform(Rotation.from_matrix(predict_pose[0:3,0:3]), predict_pose[0:3,3]))
            else:
                try:
                    j_vel = j_vel * 0.1
                    sim.robot.set_joint_control(j_vel, 'velocity')
                except:
                    pass
                sim.robot.step()
                # assistance_policy.update(sim.robot.get_ee_pose(), user_action=Action())
                # predict_command = assistance_policy.get_action().getTwist()
                # predict_pose = ApplyTwistToTransform(predict_command, sim.robot.get_ee_pose(), 0.05)
                # sim.robot.velo_control(twist_msg=predict_command)
                # sim.robot.move_robot_instant(Transform(Rotation.from_matrix(predict_pose[0:3,0:3]), predict_pose[0:3,3]))
            # print(twist_msg)

    return

if __name__ == "__main__":
    main()