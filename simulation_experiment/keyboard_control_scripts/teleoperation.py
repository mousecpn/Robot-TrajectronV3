import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import scipy.signal as signal
from tqdm import tqdm
import multiprocessing as mp

from utils_exp.grasp import Grasp, Label
from utils_exp.io import *
from utils_exp.perception import *
from utils_exp.visual import grasp2mesh,pointcloud_to_meshes
from experiment.simulation import ClutterRemovalSim
from utils_exp.transform import Rotation, Transform
import collections
from utils_exp.implicit import get_scene_from_mesh_pose_list
from threading import Thread, Lock
import spatialgeometry as sg
import spatialmath as sm
from utils_exp.control import velocity_based_control
from utils_exp.transform import expmap_to_rpyvel
import torch
import swift

State = collections.namedtuple("State", ["tsdf", "pc"])
twist_msg = np.zeros((7))


def keyboard_detection(p, velo=0.2):
    global twist_msg
    while True:
        #Create zero twist message
        twist_msg = np.zeros((7))
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
            twist_msg[-1] = 1

        if ord('m') in g:
            twist_msg[-1] = -1
        
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

        # env = swift.Swift()
        # env.launch() #headless=True

        ## load pcl
        tsdf, pc, timing = sim.acquire_tsdf(1, 1, 40)
        sim.robot.add_robot()
        sim.save_state()

        # env.add(sim.robot.panda_control)
        # env.add(sim.robot.obstacles[0])

        T_base_task = torch.tensor(sim.robot.T_base_task.as_matrix(), dtype=torch.float).cuda()
        T_task_base = torch.inverse(T_base_task)

        pcl_ = np.asarray(pc.points)
        pcl_ = torch.tensor(pcl_, dtype=torch.float32).cuda()
        pcl_ = T_base_task @ torch.cat((pcl_,torch.ones_like(pcl_[:,:1])), dim=-1).unsqueeze(-1)


        while True:
            if twist_msg[-1] == 1:
                sim.robot.grasp_robot()
            elif twist_msg[-1] == -1:
                sim.robot.gripper_homing()
            if np.linalg.norm(twist_msg[:6]) > 0.1:
                command = twist_msg[:6]
                q0 = sim.robot.read_joint_state()[0]
                cur_pose = sim.robot.get_ee_pose()
                command[0:3] = (np.linalg.inv(cur_pose[:3,:3]) @ command[0:3].reshape(3,1)).flatten()
                # print(command)
                # Te = torch.tensor(sim.robot.panda_control.fkine(q0).A, dtype=torch.float32).unsqueeze(0)
                # rpy_vel = expmap_to_rpyvel(Te, torch.tensor(command, dtype=torch.float32), 'ZYX' )[0].numpy()

                # try:
                j_vel = velocity_based_control(sim.robot.panda_control, q0, command[:3], command[3:6], obstacles=None, onbase=False) # obstacles
                # except:
                #     j_vel = np.zeros(7)
                sim.robot.set_joint_control(j_vel, 'velocity')
                sim.robot.step()
                # sim.robot.set_joint_control([0 for _ in range(7)],'velocity')
            else:
                try:
                    j_vel = j_vel * 0.1
                    sim.robot.set_joint_control(j_vel, 'velocity')
                except:
                    pass
                sim.robot.step()
            # env.step()
            # print(twist_msg)

    return

if __name__=="__main__":
    main()