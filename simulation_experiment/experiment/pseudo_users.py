import torch
from utils_exp.transform import SO3_R3, select_grasps

import numpy as np

class SingleDofUser:
    """
    A pseudo-user that acts towards the SE(3) goal by commanding movement 
    in only one degree of freedom (DOF) at a time for a fixed period.
    
    The DOF is selected only when the period expires, switching to the DOF 
    that has the LARGEST absolute error (twist component) at that moment.
    
    The command is a 6D twist vector w.r.t the current frame.
    """
    def __init__(self, steps_per_dof=5):
        """
        Initializes the SingleDofUser.

        :param steps_per_dof: The number of steps to stay focused on a single DOF 
                              before re-evaluating the largest error and switching.
        """
        self.steps_per_dof = steps_per_dof
        self.goal = None
        # State variables to manage the fixed-period switching
        self.step_count = 0
        # -1 indicates the DOF has not been selected yet (or needs recalculation)
        self.current_dof_index = -1 
        
    def set_goal(self, goal):
        """Sets the SE(3) goal pose (4x4 matrix or batched equivalent)."""
        self.goal = goal
    
    def reset(self):
        """Resets the internal state for a new episode."""
        self.step_count = 0
        self.current_dof_index = -1
        return
    
    def _calculate_error_twist(self, current_pose):
        """
        Calculates the full 6D twist vector (error) needed to reach the goal.
        The twist is in the SE(3) Lie algebra (se(3)) w.r.t. the current frame:
        [v_x, v_y, v_z, omega_x, omega_y, omega_z].
        """
        if self.goal is None:
            # Return zero command if no goal is set
            return torch.zeros(6, dtype=torch.float32).cuda() 
            
        # Select the best grasp and compute the error matrix
        goal = select_grasps(self.goal.unsqueeze(0), current_pose).squeeze(0)
        # T_error = T_current_to_world^{-1} @ T_goal_to_world
        error_mat = torch.inverse(current_pose) @ goal
        
        # Compute the 6D twist 'e' using the Lie logarithm map on SE(3)
        # e = [v_x, v_y, v_z, omega_x, omega_y, omega_z]
        e = SO3_R3.from_matrix(error_mat.unsqueeze(0)).log_map().reshape(-1)
        
        return e

    def provide_command(self, current_pose):
        """
        Provides the 6D twist command, activating only the selected DOF 
        for the current period.

        :param current_pose: The current SE(3) pose (4x4 matrix).
        :return: A 6D tensor representing the command twist vector.
        """
        # 1. Calculate the full error twist command
        e = self._calculate_error_twist(current_pose)
        adjusted_e = e.clone()
        adjusted_e[3:] = adjusted_e[3:] * 0.1  # Scale rotational part if needed

        # 2. Check if it's time to switch DOF (start of episode or period expired)
        if self.current_dof_index == -1 or self.step_count >= self.steps_per_dof:
            
            # Find the index of the DOF with the largest absolute error
            self.current_dof_index = torch.argmax(torch.abs(adjusted_e)).item()
            
            # Reset step count for the new DOF period
            self.step_count = 0
        
        # 3. Create the constrained command based on the currently selected DOF
        command = torch.zeros_like(e)
        
        # Only set the command component for the selected DOF index
        # We use the error magnitude of the current step (e) for the command value, 
        # but the index is fixed for the duration of the period.
        command[self.current_dof_index] = e[self.current_dof_index] * 2

        # 4. Increment step count for the current period
        self.step_count += 1
            
        return command

# Define constants for the modes
TRANSLATION_MODE = 0
ROTATION_MODE = 1

class ModeSwitchingUser:
    """
    A pseudo-user that alternates between providing only translation commands 
    and only rotation commands for a fixed number of steps in each mode.
    
    The command is a 6D twist vector (linear velocity, angular velocity) 
    in the current frame, derived from the SE(3) error.
    """
    def __init__(self, steps_per_mode=20):
        """
        Initializes the ModeSwitchingUser.

        :param steps_per_mode: The number of command steps to stay in one mode 
                                (translation or rotation) before switching.
        """
        self.steps_per_mode = steps_per_mode
        self.mode = TRANSLATION_MODE
        self.step_count = 0
        self.goal = None
        
    def set_goal(self, goal):
        """Sets the SE(3) goal pose (4x4 matrix)."""
        self.goal = goal
    
    def reset(self):
        """Resets the internal state to start a new sequence."""
        self.mode = TRANSLATION_MODE
        self.step_count = 0
        return
    
    def _calculate_error_twist(self, current_pose):
        """
        Calculates the 6D twist vector (error) needed to reach the goal.
        
        The twist is in the SE(3) Lie algebra (se(3)) w.r.t. the current frame.
        """
        # Ensure goal is set
        if self.goal is None:
            # Return zero command if no goal is set
            return torch.zeros(6, dtype=torch.float32).cuda() 
            
        # 1. Calculate the error matrix (T_current_to_goal)
        # error_mat = T_current_to_world^{-1} @ T_goal_to_world
        # Use select_grasps to handle multiple potential goals if necessary, 
        # mimicking the structure of the other users.
        goal = select_grasps(self.goal.unsqueeze(0), current_pose).squeeze(0)
        error_mat = torch.inverse(current_pose) @ goal
        
        # 2. Compute the 6D twist 'e' using the Lie logarithm map on SE(3)
        # e = log(error_mat)
        # The result 'e' is a 6D vector (linear velocity v, angular velocity omega)
        # which represents the desired command in the current frame.
        e = SO3_R3.from_matrix(error_mat.unsqueeze(0)).log_map().reshape(-1)
        
        # e has the structure [v_x, v_y, v_z, omega_x, omega_y, omega_z]
        return e

    def provide_command(self, current_pose):
        """
        Provides the 6D twist command based on the current mode and pose error.

        :param current_pose: The current SE(3) pose (4x4 matrix).
        :return: A 6D tensor representing the command twist vector.
        """
        # 1. Calculate the full error twist command
        e = self._calculate_error_twist(current_pose)
        
        # 2. Modify command based on the current mode
        command = e.clone()
        if self.mode == TRANSLATION_MODE:
            # Translation Mode: Only provide linear velocity (first 3 DOFs)
            # Set rotational components (last 3 DOFs) to zero.
            command[3:] = 0.0
        elif self.mode == ROTATION_MODE:
            # Rotation Mode: Only provide angular velocity (last 3 DOFs)
            # Set translational components (first 3 DOFs) to zero.
            command[:3] = 0.0

        # 3. Update the mode and step count
        self.step_count += 1
        if self.step_count >= self.steps_per_mode:
            # Time to switch modes
            if self.mode == TRANSLATION_MODE:
                self.mode = ROTATION_MODE
            else: # self.mode == ROTATION_MODE
                self.mode = TRANSLATION_MODE
            self.step_count = 0
            
        return command

class NoisyUser:
    def __init__(self, sigma):
        self.noise_free = False
        if isinstance(sigma, (float, int)):
            if sigma == 0:
                self.noise_free = True
            self.sigma = torch.diag(torch.tensor([sigma, sigma, sigma, sigma, sigma, sigma], dtype=torch.float32)).cuda()
        else:
            if sigma.max() == 0:
                self.noise_free = True
            self.sigma = sigma
        if not self.noise_free:
            self.model = torch.distributions.MultivariateNormal(torch.zeros(6, dtype=torch.float32).cuda(), self.sigma)

    def set_goal(self, goal):
        self.goal = goal

    def reset(self):
        return
    
    def provide_command(self, current_pose):
        goal = select_grasps(self.goal.unsqueeze(0), current_pose).squeeze(0)
        error_mat = torch.inverse(current_pose) @ goal
        e = SO3_R3.from_matrix(error_mat.unsqueeze(0)).log_map().reshape(-1)

        # e = torch.cat((error_mat[:3,3], matrix_to_euler_angles(error_mat[:3,:3], 'ZYX') * 0.2), dim=-1)
        # print('distance:',e.abs().sum().item())
        command_mean = e
        if self.noise_free:
            return command_mean
        command = command_mean + self.model.sample()
        return command

class LaggyUser:
    def __init__(self, lag_steps):
        self.lag_steps = lag_steps
        self.last_action = None
    
    def set_goal(self, goal):
        self.goal = goal
    
    def reset(self):
        self.last_action = None
        return
    
    def provide_command(self, current_pose):
        if self.last_action is not None and np.random.random() < 0.8:
            delayed_command = self.last_action
        else:
            goal = select_grasps(self.goal.unsqueeze(0), current_pose).squeeze(0)
            error_mat = torch.inverse(current_pose) @ goal
            e = SO3_R3.from_matrix(error_mat.unsqueeze(0)).log_map().reshape(-1)
            delayed_command = e
            self.last_action = delayed_command
            
        return delayed_command

class LaggyUser:
    def __init__(self, ):
        self.last_action = None
    
    def set_goal(self, goal):
        self.goal = goal
    
    def reset(self):
        self.last_action = None
        return
    
    def provide_command(self, current_pose):
        if self.last_action is not None and np.random.random() < 0.8:
            delayed_command = self.last_action
        else:
            goal = select_grasps(self.goal.unsqueeze(0), current_pose).squeeze(0)
            error_mat = torch.inverse(current_pose) @ goal
            e = SO3_R3.from_matrix(error_mat.unsqueeze(0)).log_map().reshape(-1)
            delayed_command = e
            self.last_action = delayed_command
        
        return delayed_command

class LowDofUser:
    def __init__(self,):
        return
    
    def set_goal(self, goal):
        self.goal = goal
    
    def reset(self):
        return
    
    def provide_command(self, current_pose):
        goal = select_grasps(self.goal.unsqueeze(0), current_pose).squeeze(0)
        error_mat = torch.inverse(current_pose) @ goal
        e = SO3_R3.from_matrix(error_mat.unsqueeze(0)).log_map().reshape(-1)
        e[3:] = 0  # only provide translation command
        return e