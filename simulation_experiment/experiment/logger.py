import numpy as np
import os

class DataCollector:
    def __init__(self, save_file_name, save_freq=20):
        self.data = {}
        self.save_file_name = save_file_name
        self.save_freq = save_freq
        self.total_traj_count = 0
        """
            data: 
                scene_id -> List of trajectories
                    - 'joint_states': List of joint states
                    - 'ee_poses': List of end-effector poses
                    - 'robot_base_pose': Pose of the robot base
                    - 'save_freq': Frequency of saving this trajectory
        """
    
    def add_trajectory(self, scene_id, joint_states, ee_poses, robot_base_pose=None, save_freq=None):
        """
        Add a trajectory to the collector.
        
        joint_states: List of joint states (each state is a list of joint angles)
        ee_poses: List of end-effector poses (each pose is a list of [x, y, z, qx, qy, qz, qw])
        """
        trajectory = {
            'joint_states': joint_states,
            'ee_poses': ee_poses,
            'robot_base_pose': robot_base_pose if robot_base_pose is not None else [0, 0, 0, 0, 0, 0, 1],
            'save_freq': save_freq if save_freq is not None else self.save_freq
        }
        if scene_id not in self.data:
            self.data[scene_id] = []
        self.data[scene_id].append(trajectory)
    
    def new_trajectory(self, scene_id):
        """
        Start a new trajectory.
        """
        self.data[scene_id].append({'joint_states': [], 'ee_poses': [], 'robot_base_pose': [0, 0, 0, 0, 0, 0, 1]})
    
    def new_scene(self, scene_id):
        """
        Start a new scene.
        """
        if scene_id not in self.data:
            self.data[scene_id] = []

    def save(self):
        """
        Save the collected trajectories to a file.
        """
        if not self.data:
            print("No trajectories to save.")
            return
        # if os.path.exists(self.save_file_name) is False:
        #     os.makedirs(self.save_file_name)
        if os.path.exists(self.save_file_name):
            recorded_data = np.load(self.save_file_name, allow_pickle=True)["trajectories"].item()
            self.data.update(recorded_data)
        # Save trajectories to a .npz file
        np.savez(self.save_file_name, trajectories=np.array(self.data, dtype=object))
        print(f"Saved {len(self.data)} scenes to {self.save_file_name}")
    
    def load(self):
        """
        Load trajectories from a file.
        """
        if os.path.exists(self.save_file_name):
            data = np.load(self.save_file_name, allow_pickle=True)
            self.data = data['trajectories'].item()
            print(f"Loaded {len(self.data)} trajectories from {self.save_file_name}")
        else:
            print("No saved trajectories found.")
    
    def add_current_states(self, scene_id, ee_poses, joint_states):
        """
        Add the current joint state and end-effector pose to the last trajectory.
        
        joint_state: Current joint state (list of joint angles)
        ee_pose: Current end-effector pose (list of [x, y, z, qw, qx, qy, qz])
        """
        if scene_id not in self.data or not self.data[scene_id]:
            print(f"No trajectories found for scene {scene_id}.")
            return
        last_trajectory = self.data[scene_id][-1]
        last_trajectory['joint_states'] = np.concatenate((last_trajectory['joint_states'], joint_states), axis=0)
        last_trajectory['ee_poses'] = np.concatenate((last_trajectory['ee_poses'], ee_poses), axis=0)