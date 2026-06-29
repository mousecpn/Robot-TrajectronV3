import json

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
import numpy as np
from utils_exp.transform import Transform, matrix_to_euler_angles, SO3_R3, select_grasps
from queue import Queue
from std_msgs.msg import Empty
import torch
import sys
import os

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from dataset.se3_preprocessing import se3_derivatives_of, js_derivatives_of, plot_se3_poses
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import time
from rclpy.callback_groups import ReentrantCallbackGroup
import collections

def init_model(checkpoint="epoch10|20Hz|ade20.86.pth"):
    torch.cuda.set_device('cuda:0')

    # Load hyperparameters from json

    with open("../config/config.json", 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)
    

    # Add hyperparams from arguments
    # hyperparams['batch_size'] = args.batch_size
    # hyperparams['k_eval'] = args.k_eval
    hyperparams['pcl_encoding'] = True
    hyperparams['frequency'] = 20
    device = torch.device('cuda:0')

    print(f"Loading model from {checkpoint}")
    trajectron = Trajectron(hyperparams, device)
    model = torch.load(checkpoint)
    trajectron.model.node_modules = model
    trajectron.set_annealing_params()
    max_hl = hyperparams['maximum_history_length']
    ph = hyperparams['prediction_horizon']
    trajectron.model.to(device)
    trajectron.model.eval()
    return trajectron


class MultivariateNormal:
    def __init__(self, mean: torch.Tensor, covariance_matrix: torch.Tensor):
        self.mean = mean 
        self.covariance_matrix = covariance_matrix 
        
    def update_mean(self, new_mean: torch.Tensor):
        self.mean = new_mean
    
    def update_covariance(self, new_cov: torch.Tensor):
        self.covariance_matrix = new_cov

class TrajectronNode(Node):
    def __init__(self, checkpoint_path, activate_ros=True):
        super().__init__('trajectron_node')
        # QoS profile for visualization
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        self.callback = ReentrantCallbackGroup()
        self.device = torch.device('cuda:0')

        # output
        self.a_dist = None
        self.last_action = None
        self.predictions = None

        self.base_sigma_val = [0.03]*3 + [0.1, 0.1, 0.1]
        self.base_sigma_vec = torch.tensor(self.base_sigma_val, dtype=torch.float32).to(self.device) #* 2
        sigma_mat = torch.diag(self.base_sigma_vec).unsqueeze(0) 
        self.user_model = MultivariateNormal(
            torch.zeros(1, 6).to(self.device), 
            sigma_mat
        )

        self.subscriber = self.create_subscription(Empty, '/traj_pred', self.trajectory_prediction, 1, callback_group=self.callback)
        self.publisher = self.create_publisher(Empty, '/traj_pred', 1, callback_group=self.callback)
        self.interval = 1
        self.dt = 0.05
        self.relative = True

        self.history_size = 6
        self.vel_history = collections.deque(maxlen=self.history_size)
        self.sigma_alpha = 0.2       # 移动平均系数 (0~1)，越小越平滑
        self.min_sigma = 0.01        # 最小 Sigma 防止数值不稳定
        self.max_sigma = 0.5         # 最大 Sigma 防止分布过散

        # RobotState
        self.js_traj = None
        self.pose_traj = None
        self.grasps = None
        self.pcl = None

        # rt
        self.ph = 6
        self.trajectron = init_model(checkpoint_path)

        self.op_count = 0

    def trajectory_prediction_asyn(self, ):
        self.publisher.publish(Empty())

    def trajectory_prediction(self, msg):
        # t1 = time.time()
        self.op_count += 1
        if self.pose_traj is not None and self.pose_traj.shape[0] >= 4 and self.op_count % self.interval == 0:
            batch = self.prepare_input()
            with torch.no_grad():
                    ################# most likely ##############################
                    y_dist, a_dist, predictions = self.trajectron.predict(batch,
                                            ph=self.ph,
                                            num_samples=1,  # Get multiple samples for visualization
                                            z_mode=False,
                                            gmm_mode=True,
                                            all_z_sep=False,
                                            full_dist=True,
                                            dist=True,
                                            measure=self.user_model)
            self.a_dist = a_dist
            self.predictions = predictions
            ## velo control ###
            self.last_action = a_dist.get_at_time(0).mode()[0,0,:1]
            # action = SO3_R3().exp_map(a_dist.get_at_time(0).mode()[0,0,:1]*self.dt).to_matrix().cpu().numpy()
            # next_action = self.pose_traj[-1].cpu().numpy() @ action[0]
            # next_action = Transform.from_matrix(next_action).as_matrix()
            # self.last_action = next_action
        t2 = time.time()
        # print("latency:", t2-t1)
        return
    
    def reset(self):
        self.js_traj = None
        self.pose_traj = None
        self.grasps = None
        self.pcl = None
        self.op_count = 0
        self.vel_history = collections.deque(maxlen=self.history_size)
        sigma_mat = torch.diag(self.base_sigma_vec).unsqueeze(0) 
        self.user_model = MultivariateNormal(
            torch.zeros(1, 6).to(self.device), 
            sigma_mat
        )
        return
    

    def pose_distance(self, pose1, pose2) -> float:
        # Transform from the end-effector to desired pose
        eTep = np.linalg.inv(pose1) * pose2

        # Spatial error
        e = np.sum(np.abs(np.r_[eTep[:3,3], matrix_to_euler_angles(eTep[:3,:3], "ZYX") / np.pi * 0.3]))
        return e
    
    def update_state(self, data_dict):
        if "js" in data_dict:
            cur_js = torch.tensor(data_dict['js'],dtype=torch.float32).unsqueeze(0).cuda()
            if self.js_traj is None:
                self.js_traj = cur_js
            else:
                self.js_traj = torch.cat((self.js_traj, cur_js),dim=0)
                if self.js_traj.shape[0] > 9:
                    self.js_traj = self.js_traj[-9:]
        
        if "pose" in data_dict:
            cur_pose = torch.tensor(data_dict['pose'], dtype=torch.float32).unsqueeze(0).cuda()
            if self.pose_traj is None:
                self.pose_traj = cur_pose
            else:
                self.pose_traj = torch.cat((self.pose_traj, cur_pose),dim=0)
                if self.pose_traj.shape[0] > 9:
                    self.pose_traj = self.pose_traj[-9:]

        if "grasps" in data_dict:
            if isinstance(data_dict['grasps'], np.ndarray):
                self.grasps = torch.tensor(data_dict['grasps'],dtype=torch.float32).cuda()
            if isinstance(data_dict['grasps'], torch.Tensor):
                self.grasps = data_dict['grasps'].cuda()

        
        if "pcl" in data_dict:
            if isinstance(data_dict['pcl'], np.ndarray):
                self.pcl = torch.tensor(data_dict['pcl'],dtype=torch.float32).cuda().reshape(-1, 3)
                if self.pcl.shape[-2] == 3:
                    self.pcl = torch.cat((self.pcl, torch.ones_like(self.pcl[:,:1])), dim=-1).unsqueeze(-1)
            if isinstance(data_dict['pcl'], torch.Tensor):
                self.pcl = data_dict['pcl'].cuda()
    
    def update_user_model(self, command):
        self.user_model.update_mean(command.reshape(1,6))
        self.vel_history.append(command.reshape(1,6))

        if len(self.vel_history) >= self.history_size:
            # 计算当前速度的方差
            history_stack = torch.cat(list(self.vel_history), dim=0)

            trans_history = history_stack[:,:3]
            rot_history = history_stack[:,3:]

            trans_cov = torch.cov(trans_history.T) * 1.4
            rot_cov = torch.cov(rot_history.T) * 1.4

            cur_cov = self.user_model.covariance_matrix[0].clone()
            # print('trans cov:',(torch.det(trans_cov)**(1/3)).item())
            # print('rot cov:',(torch.det(rot_cov)**(1/3)).item())
            ## translation update ##
            if torch.det(trans_cov)**(1/3) > self.min_sigma:
                cur_cov[:3, :3] = ((1.0 - self.sigma_alpha) * cur_cov[:3, :3] + self.sigma_alpha * trans_cov)
            # if torch.sum(trans_history.abs()) < 0.001:
            #     cur_cov[:3, :3] = (1.0 - self.sigma_alpha) * cur_cov[:3, :3] + self.sigma_alpha * torch.eye(3).to(self.device)
            ## rotation update ##
            if torch.det(rot_cov)**(1/3) > self.min_sigma:
                cur_cov[3:, 3:] = ((1.0 - self.sigma_alpha) * cur_cov[3:, 3:] + self.sigma_alpha * rot_cov)
            # if torch.sum(rot_history.abs()) < 0.001:
            #     cur_cov[3:, 3:] = (1.0 - self.sigma_alpha) * cur_cov[3:, 3:] + self.sigma_alpha * torch.eye(3).to(self.device)
            # print('updated cov:', (torch.det(cur_cov)**(1/6)).item())
            

            # measured_std = torch.std(history_stack, dim=0) # (6,)
            # measured_mean = torch.mean(history_stack, dim=0)
            # update_mask = (measured_std > self.min_sigma) & (measured_mean > 0.01) # (6,)
            # update_mask[:3] = update_mask[:3].any()
            # update_mask[3:] = update_mask[3:].any()

            # cur_sigma_vec = self.user_model.covariance_matrix.diagonal().clone()
            # updated_sigma_vec[update_mask] = ((1.0 - self.sigma_alpha) * cur_sigma_vec + \
            #                             self.sigma_alpha * measured_std)[update_mask]
            
            # new_cov_mat = torch.diag(updated_sigma_vec).unsqueeze(0)
            self.user_model.update_covariance(cur_cov.unsqueeze(0))
        return

    

    def prepare_input(self,):
        ee_vel_traj = se3_derivatives_of(self.pose_traj, dt=self.dt)[-8:]
        js_vel_traj = js_derivatives_of(self.js_traj, dt=self.dt)[-8:]

        first_history_index = torch.LongTensor(np.array([0])).cuda()
        T_curpose_base = torch.inverse(self.pose_traj[-1])
        if self.relative:
            ee_traj_rel = T_curpose_base @ self.pose_traj[-8:]
        else:
            ee_traj_rel = self.pose_traj[-8:]
        ee_traj_logmap = SO3_R3.from_matrix(ee_traj_rel).log_map()
        ee_traj = torch.cat((ee_traj_logmap, ee_vel_traj), dim=-1) 
        js_traj = torch.cat((self.js_traj[-8:], js_vel_traj), dim=-1)

        x = ee_traj[-8:,:].unsqueeze(0).cuda()
        q = js_traj[-8:,:].unsqueeze(0).cuda()
        y = torch.zeros(1, self.ph, x.shape[-1]).cuda()

        dim = x.shape[1]
        # pcl_ = torch.cat((self.pcl, torch.ones_like(pcl[:,:1])), dim=-1).unsqueeze(-1)
        if self.relative:
            grasps = T_curpose_base @ self.grasps
            pcl_ = T_curpose_base @ self.pcl
        pcl = pcl_[:,:3,0]
        
        grasps = select_grasps(grasps)
        grasps_data = SO3_R3.from_matrix(grasps).log_map()

        context = {
            'grasp': [grasps_data],
            'pcl': [pcl]
        }

        
        batch = (first_history_index, x, q, y, context)

        return batch