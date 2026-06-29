import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from model.transform import SO3_R3, matrix_to_euler_angles
import torch

def compute_ade(predicted_trajs, gt_traj):
    size = gt_traj.shape[:-1]
    gt_traj = SO3_R3().exp_map(gt_traj.reshape(-1, 6)).to_matrix().reshape(size + (4, 4)).unsqueeze(0)
    gt_traj_inv = torch.linalg.inv(gt_traj)
    error_mat = gt_traj_inv @ predicted_trajs
    ade = torch.cat((error_mat[..., :3,3], matrix_to_euler_angles(error_mat[...,:3,:3], 'XYZ')/np.pi * 0.3), dim=-1).abs().sum(-1) # sum the 
    return torch.min(ade.mean(-1),dim=0)[0]


def compute_fde(predicted_trajs, gt_traj):
    final_gt_pose = gt_traj[:,-1,:]
    size = final_gt_pose.shape[:-1]
    final_gt_pose = SO3_R3().exp_map(final_gt_pose.reshape(-1, 6)).to_matrix().reshape(size + (4, 4)).unsqueeze(0)
    final_gt_pose_inv = torch.linalg.inv(final_gt_pose)
    predicted_final_pose = predicted_trajs[:,:,-1]
    error_mat = final_gt_pose_inv @ predicted_final_pose
    fde = torch.cat((error_mat[..., :3,3], matrix_to_euler_angles(error_mat[...,:3,:3], 'XYZ')/np.pi * 0.3), dim=-1).abs().sum(-1)
    return torch.min(fde,dim=0)[0]


def compute_kde_nll(predicted_trajs, gt_traj):
    kde_ll = 0.
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[0]
    num_batches = predicted_trajs.shape[0]

    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = gaussian_kde(predicted_trajs[batch_num, :, timestep].T)
                pdf = np.clip(kde.logpdf(gt_traj[timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
                kde_ll += pdf / (num_timesteps * num_batches)
            except np.linalg.LinAlgError:
                kde_ll = np.nan

    return -kde_ll


