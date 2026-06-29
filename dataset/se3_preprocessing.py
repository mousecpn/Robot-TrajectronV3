import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from torch.utils.data._utils.collate import default_collate
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from model.transform import SO3_R3, quaternion_to_matrix, matrix_to_rotation_6d, Transform, matrix_to_euler_angles, select_grasps
from model.dynamics import SE3Integrator
from model.io import read_point_cloud
import torch.nn.functional as F
import open3d as o3d
def read_df(root):
    return pd.read_csv(root / "grasps.csv")

def make_continuous_copy(alpha):
    alpha = (alpha + np.pi) % (2.0 * np.pi) - np.pi
    continuous_x = np.zeros_like(alpha)
    continuous_x[0] = alpha[0]
    for i in range(1, len(alpha)):
        if not (np.sign(alpha[i]) == np.sign(alpha[i - 1])) and np.abs(alpha[i]) > np.pi / 2:
            continuous_x[i] = continuous_x[i - 1] + (
                    alpha[i] - alpha[i - 1]) - np.sign(
                (alpha[i] - alpha[i - 1])) * 2 * np.pi
        else:
            continuous_x[i] = continuous_x[i - 1] + (alpha[i] - alpha[i - 1])

    return continuous_x


def se3_derivatives_of(x, dt=0.1):
    """
        x: (timestep, 4, 4)
    """
    timestep = x.shape[0]
    T_t = x[1:]
    T_tmin1 = x[:-1]
    DeltaT = torch.bmm(torch.inverse(T_tmin1), T_t)
    twist = SO3_R3.from_matrix(DeltaT).log_map()
    twist = twist/dt
    twist = torch.cat((torch.zeros(1, 6).to(twist.device), twist), dim=0)  # [timestep-1, 6]
    return twist

def js_derivatives_of(x, dt=0.1):
    """
        x: (timestep, 4, 4)
    """
    timestep = x.shape[0]
    js_t = x[1:]
    js_tmin1 = x[:-1]
    js_vel = js_t - js_tmin1
    
    js_vel = js_vel/dt
    js_vel = torch.cat((torch.zeros(1, 7).to(js_vel.device), js_vel), dim=0)  # [timestep-1, 6]
    return js_vel


from multiprocessing import Pool
from functools import partial

def process_scene(scene_id, scene_data_all, scene_file):
    mask = scene_data_all.loc[:, "scene_id"] == scene_id
    term = scene_data_all.loc[mask, "qx":"label"].to_numpy(np.single)
    filtered = term[term[:, -1] > 0.4]
    if len(filtered) == 0:
        print(f"Warning: No data found for scene {scene_id} in {scene_file}")
    return scene_id, filtered

def combine_dict(list_of_dicts):
    combined_dict = {}
    for d in list_of_dicts:
        combined_dict.update(d)
    return combined_dict
    

def load_data(scene_file, trajectory_file, dt=0.05, viz=False):
    if isinstance(trajectory_file, list):
        traj_data = []
        for i in range(0, len(trajectory_file)):
            if os.path.exists(trajectory_file[i]):
                traj_data.append(np.load(trajectory_file[i], allow_pickle=True)['trajectories'].item())
                print(f"Loaded {len(traj_data[i])} scenes from {trajectory_file[i]}")
        if len(traj_data) > 0:
            traj_data = combine_dict(traj_data)
        else:
            print("No saved trajectories found.")
            return {}, [], []
    else:
        if os.path.exists(trajectory_file):
            traj_data = np.load(trajectory_file, allow_pickle=True)['trajectories'].item()
            print(f"Loaded {len(traj_data)} scenes from {trajectory_file}")
        else:
            print("No saved trajectories found.")
            return {}, [], []

    if viz is False:
        scene_indices = list(traj_data.keys())
        train_scenes, test_scenes = train_test_split(scene_indices, test_size=0.1, random_state=42)
        
        
        train_traj_data = []
        for scene_idx in train_scenes:
            for traj in traj_data[scene_idx]:
                traj['scene_id'] = scene_idx
                train_traj_data.append(traj)
        length = 20 * int(dt/0.05)
        test_traj_data = []
        for scene_idx in test_scenes:
            for traj in traj_data[scene_idx]:
                splited_trajs = split_traj(traj, length=length, scene_idx=scene_idx)
                test_traj_data.extend(splited_trajs)
    else:
        all_traj_data = []
        for scene_idx in traj_data.keys():
            for traj in traj_data[scene_idx]:
                traj['scene_id'] = scene_idx
                all_traj_data.append(traj)

        train_traj_data, test_traj_data = train_test_split(all_traj_data, test_size=0.2, random_state=42)

    scene_data_all = read_df(Path(scene_file))

    # 用 partial 预填函数参数
    partial_process = partial(process_scene, scene_data_all=scene_data_all, scene_file=scene_file)

    with Pool(processes=20) as pool:
        results = pool.map(partial_process, traj_data.keys())

    scene_data = dict(results)

    return scene_data, train_traj_data, test_traj_data

# def sub_traj(traj_data, dt):
#     output_traj_data = []
#     freq = traj_data['save_freq']
#     stride = int(freq*dt)
#     for s in range(stride):
#         idx_list = np.array(range(s, traj_data['ee_poses'].shape[0], stride))
#         output_traj_data.append({
#             'joint_states': traj_data['joint_states'][idx_list],
#             'ee_poses': traj_data['ee_poses'][idx_list],
#             'robot_base_pose': traj_data['robot_base_pose'],
#             'save_freq': traj_data['save_freq'],
#         })

def split_traj(traj_data, length, scene_idx):
    splited_trajs = []
    for j in range(traj_data['ee_poses'].shape[0]//length):
        ee_poses = traj_data['ee_poses'][j*length:(j+1)*length]
        joint_states = traj_data['joint_states'][j*length:(j+1)*length]
        if ee_poses.shape[0] < length:
            continue
        splited_trajs.append(
            {
                'ee_poses': ee_poses,
                "joint_states": joint_states,
                "robot_base_pose": traj_data['robot_base_pose'],
                'save_freq': traj_data['save_freq'],
                'scene_id': scene_idx
            }
        )
    splited_trajs.append(
        {
            'ee_poses': traj_data['ee_poses'][-length:],
            "joint_states": traj_data['joint_states'][-length:],
            "robot_base_pose": traj_data['robot_base_pose'],
            'save_freq': traj_data['save_freq'],
            'scene_id': scene_idx
        }
    )
    return splited_trajs


class SE3GraspTrajDataset(Dataset):
    def __init__(self, scene_data, traj_data, pcl_root, max_history_length, min_future_timesteps, frequency, eval=False, pad=False, relative=False, pregrasp=False, loadpcl=False, noise=False, normalize=False):
        self.traj_data = traj_data
        self.scene_data = scene_data
        # self.data = torch.FloatTensor(Data)
        self.max_ht = max_history_length
        self.min_ft = min_future_timesteps
        # self.inseq = self.data[:, :in_length, :]
        # self.outseq = self.data[:, in_length:, :]
        self.eval = eval
        self.pad = pad
        self.map_scale = 1
        self.dt=1/frequency
        self.relative = relative
        self.pregrasp = pregrasp
        self.T_body_tcp = Transform.from_dict({"rotation": [0.000, 0.000, 0.0, 1.0], "translation": [0.000, 0.000, 0.05]}).as_matrix()
        self.T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.045]).as_matrix()
        self.extend_len = 13 # if eval==False else 0
        self.loadpcl = loadpcl
        self.pcl_root = pcl_root
        self.noise = noise
        self.normalize = normalize
        self.gripper_flip = torch.eye(4, dtype=torch.float32)
        self.gripper_flip[:3,:3] = torch.tensor(Rotation.from_euler("z", np.pi).as_matrix(), dtype=torch.float32)
        # self.T_center_base = Transform.from_dict({"rotation": [0.000, 0.000, 0.0, 1.0], "translation": [-0.15, -0.15, -0.1]}).as_matrix()

        # self.dynamics = SE3Integrator(self.dt,{},'cpu')
        return
    
    def __len__(self):
        return len(self.traj_data)
    
    def __getitem__(self, index):
        seq = self.traj_data[index]

        # trajectory
        js_traj = torch.tensor(seq['joint_states'],  dtype=torch.float)
        ee_traj_ori = torch.tensor(seq['ee_poses'], dtype=torch.float)
        T_base_task = torch.tensor(seq['robot_base_pose'], dtype=torch.float)
        save_freq = seq['save_freq']

        stride = int(self.dt * save_freq)
        s = np.random.randint(0, stride)
        idx_list = np.array(range(s, js_traj.shape[0], stride))
        js_traj = js_traj[idx_list]
        ee_traj_ori = ee_traj_ori[idx_list]

        scene_id = seq['scene_id']

        if self.eval:
            t = np.array(8)
        else:
            t = np.random.choice(np.arange(3, len(ee_traj_ori)-self.min_ft+self.extend_len), replace=False)
            # t = n   .array(8)
        timestep_range_x = np.array([t - self.max_ht, t])
        timestep_range_y = np.array([t, t + self.min_ft])
        first_history_index = (self.max_ht - t).clip(0)
        cur_step = timestep_range_x[-1] - 1

        length = timestep_range_x[1] - timestep_range_x[0]



        #### loading grasps ####
        grasps = self.scene_data[scene_id]
        rotations = quaternion_to_matrix(torch.tensor(grasps[:, :4], dtype=torch.float))
        translations = torch.tensor(grasps[:, 4:7], dtype=torch.float)

        grasps = SO3_R3(rotations, translations).to_matrix()
        
        if not self.normalize:
            grasps = T_base_task @ grasps
        grasps = grasps @ torch.tensor(self.T_body_tcp,  dtype=torch.float) 
        if self.pregrasp:
            grasps = grasps @ torch.tensor(self.T_grasp_pregrasp, dtype=torch.float32)
        # if not self.eval:
        # _, goal_grasp = self.achieved(ee_traj_ori[-1], grasps)
        ee_traj_ori = torch.cat((ee_traj_ori, ee_traj_ori[-1].unsqueeze(0).repeat(self.extend_len, 1, 1)), dim=0)
        # ee_traj_ori[-17:-10,:3,3] = moving_average_torch(ee_traj_ori[-17:-10,:3,3], window_size=5)
        js_traj = torch.cat((js_traj, js_traj[-1:].repeat(self.extend_len, 1)), dim=0)
        
        if self.noise and not self.eval:
            noise = torch.rand(ee_traj_ori.shape[0], 6) * torch.tensor([0.005, 0.005, 0.005, 0.0025, 0.0025, 0.0025], dtype=torch.float32) * 1.4
            noisy_transform = SO3_R3().exp_map(noise).to_matrix()
            ee_traj_ori = noisy_transform @ ee_traj_ori

        #### loading pcl ####
        if self.loadpcl:
            pcl = read_point_cloud(self.pcl_root, scene_id)
            
            if pcl.shape[0] == 0:
                return self.__getitem__(np.random.randint(0, len(self.traj_data)-1))
            ## voxel downsample
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcl)
            voxel_size = 0.3/40
            pcl = np.asarray(pcd.voxel_down_sample(voxel_size=voxel_size).points)
            pcl = torch.tensor(pcl, dtype=torch.float32)
            pcl = torch.cat((pcl,torch.ones_like(pcl[:,:1])), dim=-1).unsqueeze(-1)
            if not self.normalize:
                pcl = T_base_task @ pcl
        else:
            pcl = None
        #### loading pcl ####
        
        if self.relative:
            T_curpose_base = torch.inverse(ee_traj_ori[cur_step])
            ee_traj_ori = T_curpose_base @ ee_traj_ori
            grasps = T_curpose_base @ grasps
            
            if self.loadpcl:
                pcl = T_curpose_base @ pcl

        grasps = select_grasps(grasps)#, T=ee_traj_ori[-1])
        if not self.eval and grasps.shape[0] > 10:
            _, true_grasp_idx = achieved(ee_traj_ori[cur_step], grasps)
            # random sample the grasps
            random_num_grasps = np.random.randint(10, grasps.shape[0])
            random_idx = torch.randperm(grasps.shape[0])[:min(grasps.shape[0], random_num_grasps)]
            if true_grasp_idx not in random_idx:
                random_idx[0] = true_grasp_idx
            grasps = grasps[random_idx]
        
        

        if self.normalize and not self.relative:
            grasps[:,:3,3] = (grasps[:,:3,3] - 0.15)/ 0.15
            pcl[:,:3,0] = (pcl[:,:3,0]- 0.15)/ 0.15 if pcl is not None else pcl
        if self.loadpcl:
            pcl = pcl[:,:3,0]

        ee_vel_traj = se3_derivatives_of(ee_traj_ori, dt=self.dt)

        # if not self.eval:
        #     ee_vel_traj[-self.extend_len+2:] = 0.0
        ee_traj_logmap = SO3_R3.from_matrix(ee_traj_ori).log_map()
        js_vel_traj = js_derivatives_of(js_traj, dt=self.dt)

        ee_traj = torch.cat((ee_traj_logmap, ee_vel_traj), dim=-1)  # [timestep, 7]
        js_traj = torch.cat((js_traj, js_vel_traj), dim=-1)  # [timestep, 7]

        
        # ee_traj history
        ee_traj_crop = ee_traj[max(timestep_range_x[0],0):timestep_range_x[1]]
        x = torch.full((length, ee_traj_crop.shape[1]), fill_value=torch.nan)
        x[first_history_index:length] = ee_traj_crop.clone()

        # ee_traj future
        y = ee_traj[max(timestep_range_y[0],0):timestep_range_y[1]]
        if y.shape[0] < self.min_ft:
            y_pad = torch.full((self.min_ft - y.shape[0], y.shape[1]), fill_value=torch.nan)
            y = torch.cat((y, y_pad), dim=0)

        # js_traj
        js_traj_crop = js_traj[max(timestep_range_x[0],0):timestep_range_y[1]]
        q = torch.full((self.max_ht+self.min_ft, js_traj_crop.shape[1]), fill_value=torch.nan)
        q[first_history_index:js_traj_crop.shape[0]+first_history_index] = js_traj_crop.clone()
            
        # plot_se3_poses(ee_traj_ori.numpy(), grasps.numpy(), pcl=pcl) # pcl=pcl
        # plt.show()

        # integrated_traj = self.dynamics.integrate_samples(y[:, 6:].reshape(1,1,12,6), x[-1, :6].reshape(1,1,1,6))
        # plot_se3_poses(integrated_traj.reshape(-1,4,4).numpy(), ee_traj_ori[cur_step+1:cur_step+13].numpy())

        ## log map
        grasps = SO3_R3.from_matrix(grasps).log_map()

        # 6d rotation
        # rotation_6d = matrix_to_rotation_6d(grasps[:,:3,:3])
        # grasps = torch.cat((grasps[:,:3,3], rotation_6d), dim=-1)  # [timestep, 9]
        
        return first_history_index, x, q, y, grasps, pcl

    def achieved(self, cur_pose, goal_poses):
        error_mat = torch.inverse(cur_pose) @ goal_poses
        e1 = torch.cat((error_mat[..., :3,3], matrix_to_euler_angles(error_mat[...,:3,:3], 'ZYX')/np.pi*0.3), dim=-1).abs().sum(-1)
        idx1 = torch.argmin(e1)

        error_mat = torch.inverse(cur_pose) @ (goal_poses @ self.gripper_flip.unsqueeze(0))
        e2 = torch.cat((error_mat[..., :3,3], matrix_to_euler_angles(error_mat[...,:3,:3], 'ZYX') /np.pi*0.3), dim=-1).abs().sum(-1)
        idx2 = torch.argmin(e2)

        if e1[idx1] < e2[idx2]:
            e = e1[idx1]
            target_pose = goal_poses[idx1]
        else:
            e = e2[idx2]
            target_pose = goal_poses[idx2] @ self.gripper_flip

        if e < 0.01:
            return True, target_pose
        else:
            return False, target_pose
    
    def collate(self, batch):
        first_history_indices, x, q, y, grasps, pcl = zip(*batch)
        # for b_idx in range(len(batch)):
        #     first_history_indices.append(batch[b_idx][0])
        #     x_ts.append(batch[b_idx][1])
        #     y_ts.append(batch[b_idx][2])
        #     x_st_ts.append(batch[b_idx][3])
        #     y_st_ts.append(batch[b_idx][4])
        #     goals.append(batch[b_idx][5])
        #     obs.append(batch[b_idx][6])
        #     maps.append(batch[b_idx][7])
        first_history_indices = torch.tensor(first_history_indices)
        x = torch.stack(x)
        q = torch.stack(q)
        y = torch.stack(y)
        context = {
            "grasp": grasps,
        }
        if self.loadpcl:
            context['pcl'] = pcl
        return first_history_indices, x, q, y, context

from mpl_toolkits.mplot3d import Axes3D

def plot_se3_poses(traj_poses, other_poses=None, pcl=None, ax=None, alpha=None, axis_scale=0.03, figsize=(8, 8), color='black', lineform='-', draw_frame_flag=True):
    """
    Plot SE3 poses in 3D.

    Parameters:
    - traj_poses: list or array of shape (N, 4, 4), sequence of SE3 poses for the trajectory.
    - other_poses: optional list or array of shape (M, 4, 4), additional poses to plot.
    - axis_scale: float, length of coordinate axes for each pose.
    - figsize: tuple, size of the figure.

    Each pose is a 4x4 homogeneous transformation matrix [R | t; 0 0 0 1].
    """
    components = []
    # Convert inputs to arrays
    traj_poses = np.asarray(traj_poses)
    if other_poses is not None:
        other_poses = np.asarray(other_poses)
    
    flag = 0
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        flag = 1
    if alpha is None:
        alpha = [1 for _ in range(len(traj_poses))]
    elif isinstance(alpha, np.float32) or isinstance(alpha, np.int64) or isinstance(alpha, float) or isinstance(alpha, int):
        alpha = [alpha for _ in range(len(traj_poses))]

    # Extract and plot trajectory points
    traj_points = traj_poses[:, :3, 3]
    curve = ax.plot(traj_points[:, 0], traj_points[:, 1], traj_points[:, 2], lineform, label='Trajectory', color=color, alpha=alpha[0], linewidth=3.0)
    components.extend(curve)

    # Function to draw a coordinate frame at a pose
    def draw_frame(pose, scale, alpha=1):
        R = pose[:3, :3]
        t = pose[:3, 3]
        # axes in world coords
        axes = R * scale
        colors = ['r', 'g', 'b']
        frames = []
        for i in range(3):
            frame = ax.quiver(t[0], t[1], t[2],
                      axes[0, i], axes[1, i], axes[2, i],
                      length=1.0, color=colors[i], alpha=alpha)
            frames.append(frame)
        return frames

    # Draw frames along trajectory
    pose = traj_poses[-1]
    if draw_frame_flag:
        frames = draw_frame(pose, axis_scale*0.5, alpha=alpha[-1])
        components.extend(frames)

    # Draw other poses if provided
    if other_poses is not None:
        # for pose in other_poses:
        for i in range(len(other_poses)):
            pose = other_poses[i]
            frames = draw_frame(pose, axis_scale)
            components.extend(frames)
    if pcl is not None:
        ax.scatter(pcl[:, 0], pcl[:,1], pcl[:, 2])

    # Set labels and aspect
    if flag == 1:
        # x_range = (-0.15, 0.15)
        # y_range = (-0.15, 0.15)
        # z_range = (-0.15, 0.15)
        # ax.set_xlim([x_range[0], x_range[1]])
        # ax.set_ylim([y_range[0], y_range[1]])
        # ax.set_zlim([z_range[0], z_range[1]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_box_aspect((1, 1, 1))  # equal aspect
    return components

# Example usage:
# traj = [np.eye(4) for _ in range(5)]\# shift traj for demo
# for i in range(len(traj)): traj[i][:3, 3] = [i * 0.1, 0, 0]
# others = [np.eye(4)]
# plot_se3_poses(traj, others)
def achieved(cur_pose, goal_poses):
    error_mat = torch.inverse(cur_pose) @ goal_poses
    e = torch.cat((error_mat[..., :3,3], matrix_to_euler_angles(error_mat[...,:3,:3], 'XYZ') *0.3/np.pi), dim=-1).abs().sum(-1)
    idx = torch.argmin(e)
    if torch.min(e) < 0.01:
        return True, idx
    else:
        return False, idx
    

def moving_average_torch(traj: torch.Tensor, window_size: int = 5) -> torch.Tensor:
    """
    traj: (T, 3) tensor
    """
    traj = traj.unsqueeze(0).permute(0, 2, 1)  # (1,3,T)
    kernel = torch.ones((1, 1, window_size), device=traj.device) / window_size

    # 用 replicate padding 替代 zero padding
    pad = window_size // 2
    smoothed = []
    for d in range(traj.shape[1]):
        padded = F.pad(traj[:, d:d+1, :], (pad, pad), mode="replicate")
        sm = F.conv1d(padded, kernel)
        smoothed.append(sm)
    smoothed = torch.cat(smoothed, dim=1)  # (1,3,T)
    return smoothed.permute(0, 2, 1).squeeze(0)  # (T,3)
    
if __name__ == "__main__":
    import time
    dt = 0.05
    frequency = 1/dt

    scene_file = Path("data/data_scene_raw")
    trajectory_file = Path("data/trajectory/trajectories_pregrasp.npz")
    pcl_root = Path("data/scene_data")
    

    scene_data, train_traj_data, test_traj_data = load_data(scene_file, trajectory_file, dt)
    
    print(f"Trainset length: {len(train_traj_data)}", f"Testset length: {len(test_traj_data)}")
    dataset = SE3GraspTrajDataset(scene_data, train_traj_data, pcl_root, max_history_length=8, min_future_timesteps=12, frequency=frequency, relative=True, pregrasp=True, loadpcl=True, noise=True, normalize=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True,collate_fn=dataset.collate, num_workers=8)
    t1 = time.time()
    for data in dataset:
        print("Processing time per batch:", time.time() - t1)
        t1 = time.time()
        pass
