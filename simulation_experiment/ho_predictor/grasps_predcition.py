import numpy as np
from object_grasp import ObjectGrasp
from prediction_utils import ApplyTwistToTransform
from scipy.special import logsumexp
from collections import deque
from typing import Tuple
import transformations as transmethods


weight_sc = 0.95
max_prob_any_goal = 0.99
log_max_prob_any_goal = np.log(max_prob_any_goal)
window_size = 10
ACTION_APPLY_TIME = 0.05  # used when calculating future state


class GraspsPredictor:
    def __init__(self, grasps_list, eef_pose=None):
        self.objects_grasp = []
        for id, grasp in enumerate(grasps_list):
            self.objects_grasp.append(ObjectGrasp(id, grasp))
        self._init_prediction_policies(eef_pose)
    
    def _init_prediction_policies(self, eef_pose=None):
        self.ind_max = 0
        self.log_goal_distribution = np.array([])
        if len(self.objects_grasp) <= 0:
            return
        distances = []
        # initialize prediction by distance
        for object_grasp in self.objects_grasp:
            if eef_pose is not None:
                obj_pos = object_grasp.grasp_pose[0:3, 3]
                dist = np.linalg.norm(eef_pose[:3, 3] - obj_pos)
                distances.append(dist)
        if eef_pose is None:
            self.log_goal_distribution = np.log((1./len(self.objects_grasp)) * 
                                                np.ones(len(self.objects_grasp)))  # log( p(g|xi) )'
        else:
            distances = np.array(distances)
            weights = np.exp(-distances)
            probs = weights / np.sum(weights)
            self.log_goal_distribution = np.log(probs)

    def update_prediction_policies(self, eef_pose, user_action):
        eef_pose_after_action = ApplyTwistToTransform(user_action, eef_pose, ACTION_APPLY_TIME)
        for object_grasp in self.objects_grasp:
            object_grasp.update_prediction_policy(eef_pose, eef_pose_after_action)
        # update values and q_values
        v_values = np.ndarray(len(self.objects_grasp))
        q_values = np.ndarray(len(self.objects_grasp))
        for idx, object_grasp in enumerate(self.objects_grasp):
            v_values[idx] = object_grasp.get_value()
            q_values[idx] = object_grasp.get_qvalue()
        # update log distribution: V(x; g) - Q(x, u; g) = V(x; g) - C(x, u) - V(x'; g)
        # log( p(xi|g) ) = log( p(xi|g) )@t-1 + log( exp(V - Q) )@t
        self.log_goal_distribution *= weight_sc
        self.log_goal_distribution += v_values - q_values
        # self.vq_history.append(v_values - q_values)
        # self.log_goal_distribution = np.mean(self.vq_history, axis=0)
        # normalize log distribution
        # log( sum( p(xi|g) ) )
        log_normalization_val = logsumexp(self.log_goal_distribution)
        # log( p(g|xi) )_i = log( p(xi|g) )_i - log( sum( p(xi|g) ) )
        self.log_goal_distribution -= log_normalization_val
        self._clip_prob()
        first_max, second_max = self._get_index_max()
        self.ind_max = first_max
    
    def _clip_prob(self):
        if len(self.log_goal_distribution) <= 1:
            return
        #check if any too high
        max_prob_ind = np.argmax(self.log_goal_distribution)
        if self.log_goal_distribution[max_prob_ind] > log_max_prob_any_goal:
            #see how much we will remove from probability
            diff = np.exp(self.log_goal_distribution[max_prob_ind]) - max_prob_any_goal
            #want to distribute this evenly among other goals
            diff_per = diff / (len(self.log_goal_distribution) - 1.)
            #distribute this evenly in the probability space...this corresponds to doing so in log space
            # e^x_new = e^x_old + diff_per, and this is formulate for log addition
            self.log_goal_distribution += np.log( 1. + diff_per/np.exp(self.log_goal_distribution))
            #set old one
            self.log_goal_distribution[max_prob_ind] = log_max_prob_any_goal
    
    def _get_index_max(self):
        amax = np.argmax(self.log_goal_distribution)
        mask = np.zeros_like(self.log_goal_distribution, dtype=bool)
        mask[amax] = True
        second_max = np.argmax(np.ma.masked_array(self.log_goal_distribution, mask=mask))
        return amax, second_max

    def get_goal_distribution(self):
        goal_distribution = np.exp(self.log_goal_distribution)
        distribution_with_ids = {object_grasp.id: goal_distribution[i] for i, object_grasp in enumerate(self.objects_grasp)}
        return distribution_with_ids
    
    def get_goal_grasp(self):
        if len(self.objects_grasp) == 0 or self.ind_max not in self.objects_grasp:
            return None, None
        goal_object_grasp = self.objects_grasp[self.ind_max]
        return self.ind_max, goal_object_grasp

    def get_nearest_grasp(self, eef_pose):
        nearest_grasp_id = None
        if len(self.objects_grasp) == 0:
            return None, None
        nearest_distance = np.inf
        for grasp_id, object_grasp in enumerate(self.objects_grasp):
            distance = np.linalg.norm(object_grasp.grasp_pose[:3, 3] - eef_pose[:3, 3])
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_grasp_id = grasp_id
        return nearest_grasp_id, self.objects_grasp[nearest_grasp_id]
    
    def print_probability(self):
        for i, object_grasp in enumerate(self.objects_grasp):
            prob = np.exp(self.log_goal_distribution[i])
            print(f"Grasp ID: {object_grasp.id} has probability: {prob:.4f}")

    def __repr__(self):
        return "\n".join(f"ID={object_grasp.id}, {object_grasp}" for object_grasp in self.objects_grasp)
    

def generate_random_pose():
    """Generate a random 4x4 pose matrix"""
    # Random position
    position = np.random.uniform(-1, 1, 3)
    
    # Random rotation (using random quaternion)
    quat = np.random.randn(4)
    quat = quat / np.linalg.norm(quat)
    
    pose = np.eye(4)
    pose[0:3, 3] = position
    # Convert quaternion to rotation matrix
    pose[0:3, 0:3] = transmethods.quaternion_matrix(quat)[0:3, 0:3]
    
    return pose

def generate_random_twist():
    """Generate a random 6D twist (linear and angular velocity)"""
    linear_vel = np.random.uniform(-0.1, 0.1, 3)
    angular_vel = np.random.uniform(-0.2, 0.2, 3)
    return np.concatenate([linear_vel, angular_vel])


if __name__ == "__main__":
    print("=== Grasp Prediction Test ===")
    # 1. Generate random grasp poses
    print("\n1. Generating random grasp poses...")
    grasps_list = []
    for i in range(5):
        grasp_pose = generate_random_pose()
        grasps_list.append(grasp_pose)
        print(f"Grasp {i}: position = {grasp_pose[0:3, 3]}")
    
    # 2. Generate random initial EEF position
    print("\n2. Generating random initial EEF pose...")
    eef_pose = generate_random_pose()
    print(f"Initial EEF position: {eef_pose[0:3, 3]}")
    
    # 3. Create GraspsPredictor
    print("\n3. Creating GraspsPredictor...")
    predictor = GraspsPredictor(grasps_list, eef_pose)
    
    # Print initial prediction
    print("\nInitial prediction probabilities:")
    predictor.print_probability()
    goal_id, goal_grasp = predictor.get_goal_grasp()
    if goal_grasp:
        print(f"Current predicted goal: Grasp ID {goal_id}")
    
    # 4. Apply 3 random twists and update predictions
    print("\n4. Applying 3 random twists and updating predictions...")
    current_eef_pose = eef_pose.copy()
    
    for i in range(3):
        print(f"\n--- Action {i+1} ---")
        
        # Generate random twist
        twist = generate_random_twist()
        print(f"Generated twist: linear={twist[0:3]}, angular={twist[3:6]}")
        
        # Update prediction
        predictor.update_prediction_policies(current_eef_pose, twist)
        
        # Update EEF pose for next iteration
        current_eef_pose = ApplyTwistToTransform(twist, current_eef_pose, ACTION_APPLY_TIME)
        print(f"New EEF position: {current_eef_pose[0:3, 3]}")
        
        # Print prediction results
        print("Updated prediction probabilities:")
        predictor.print_probability()
        
        goal_id, goal_grasp = predictor.get_goal_grasp()
        if goal_grasp:
            print(f"Current predicted goal: Grasp ID {goal_id}")
        
        # Print goal distribution
        goal_dist = predictor.get_goal_distribution()
        print(f"Goal distribution: {goal_dist}")
    
    print("\n=== Test completed ===")
    