import numpy as np
from torch.utils.data import Dataset,DataLoader

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from model.io import read_point_cloud
from tqdm import tqdm

class SceneDataset(Dataset):
    def __init__(self, scene_data, traj_data, max_history_length, min_future_timesteps, frequency, eval=False, pad=False, relative=False, num_point_occ=2048, normalize=False):
        self.traj_data = traj_data
        self.scene_data = scene_data
        # self.data = torch.FloatTensor(Data)
        self.max_ht = max_history_length
        self.min_ft = min_future_timesteps
        self.num_point_occ = num_point_occ
        self.eval = eval
        self.pad = pad
        self.map_scale = 1
        self.dt=1/frequency
        self.relative = relative
        self.pcl_root = Path("data/scene_packed")
        self.normalize = normalize

        return
    
    def __len__(self):
        return len(self.traj_data)
    
    def __getitem__(self, index):
        seq = self.traj_data[index]

        # trajectory

        ee_traj_ori = torch.tensor(seq['ee_poses'], dtype=torch.float)
        T_base_task = torch.tensor(seq['robot_base_pose'], dtype=torch.float)
        save_freq = seq['save_freq']

        stride = int(self.dt * save_freq)
        s = np.random.randint(0, stride)
        idx_list = np.array(range(s, ee_traj_ori.shape[0], stride))
        ee_traj_ori = ee_traj_ori[idx_list]

        scene_id = seq['scene_id']

        t = np.random.choice(np.arange(3, len(ee_traj_ori)-self.min_ft), replace=False)
        timestep_range_x = np.array([t - self.max_ht, t])
        cur_step = timestep_range_x[-1] - 1


        #### loading pcl ####
        pcl = read_point_cloud(self.pcl_root, scene_id)
        pcl = torch.tensor(pcl, dtype=torch.float32)
        pcl = torch.cat((pcl,torch.ones_like(pcl[:,:1])), dim=-1).unsqueeze(-1)
        if not self.normalize:
            pcl = T_base_task @ pcl
        #### loading pcl ####

        #### loading occ ####
        occ_points, occ = self.read_occ(scene_id, self.num_point_occ)
        occ_points = torch.tensor(occ_points, dtype=torch.float32)
        occ_points = torch.cat((occ_points,torch.ones_like(occ_points[:,:1])), dim=-1).unsqueeze(-1)
        occ = torch.tensor(occ, dtype=torch.float32)
        if not self.normalize:
            pcl = T_base_task @ pcl
            occ_points = T_base_task @ occ_points
        #### loading occ ####
        
        if self.relative:
            T_curpose_base = torch.inverse(ee_traj_ori[cur_step])
            ee_traj_ori = T_curpose_base @ ee_traj_ori
            pcl = T_curpose_base @ pcl


        if self.normalize and not self.relative:
            pcl[:,:3,0] = (pcl[:,:3,0]- 0.15)/ 0.15 if pcl is not None else pcl
        pcl = pcl[:,:3,0]
        occ_points = occ_points[:,:3,0]


            
        # plot_se3_poses(ee_traj_ori.numpy(), grasps.numpy(), ) #pcl=pcl
        # plt.show()


        return pcl, occ_points, occ
    
    def read_occ(self, scene_id, num_point):
        occ_paths = list((self.pcl_root / 'occ' / scene_id).glob('*.npz'))
        path_idx = torch.randint(high=len(occ_paths), size=(1,), dtype=int).item()
        occ_path = occ_paths[path_idx]
        occ_data = np.load(occ_path)
        points = occ_data['points']
        occ = occ_data['occ'] 
        points, idxs = sample_point_cloud(points, num_point, return_idx=True)
        occ = occ[idxs]
        return points, occ

    
    def collate(self, batch):
        pcl, occ_points, occ = zip(*batch)
        occ_points = torch.stack(occ_points)
        occ = torch.stack(occ)
        return pcl, occ_points, occ


def sample_point_cloud(pc, num_point, return_idx=False):
    num_point_all = pc.shape[0]
    idxs = np.random.choice(np.arange(num_point_all), size=(num_point,), replace=num_point > num_point_all)
    if return_idx:
        return pc[idxs], idxs
    else:
        return pc[idxs]

if __name__=="__main__":
    import json
    from pathlib import Path
    from dataset.se3_preprocessing import load_data

    with open('config/config.json', 'r') as f:
        hyperparams = json.load(f)
    
    scene_file = Path("data/data_packed_train_raw")
    trajectory_file = Path("data/trajectory/trajectories_pregrasp.npz")
    hyperparams['frequency'] = 20
    dt = 1.0/hyperparams['frequency']

    scene_data, train_traj_data, test_traj_data = load_data(scene_file, trajectory_file, dt)

    train_dataset = SceneDataset(
        scene_data,
        train_traj_data,
        max_history_length=8,
        min_future_timesteps=12,
        frequency=hyperparams['frequency'],
        relative=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=train_dataset.collate
    )

    curr_iter = 0
    train_epochs = 10
    for epoch in range(1, train_epochs + 1):
        pbar = tqdm(train_loader, ncols=80)
        for batch in pbar:
            pcl, occ_points, occ = batch
            break
        

    print("Done")