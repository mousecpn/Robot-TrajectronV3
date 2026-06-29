import numpy as np
from utils_exp.transform import Transform, Rotation
from spatialmath import SE3, base


class ClosestGrasp:
    def __init__(self, grasps: list[Transform]):
        """
        grasps: List of Grasp objects
        """
        self.grasps = grasps


    def pose_distance(self, pose1: Transform, pose2: Transform) -> float:
        # Transform from the end-effector to desired pose
        eTep = pose1.inverse() * pose2

        # Spatial error
        e = np.sum(np.abs(np.r_[eTep.translation, eTep.rotation.as_euler('xyz') * np.pi / 180]))
        return e

    
    def get_closest_grasp(self, pose: Transform) -> Transform:
        """
        Returns the grasp closest to the given pose.
        
        pose: Transform object representing the pose to compare against
        """
        closest_grasp = None
        min_distance = float('inf')
        best_idx = -1
        
        for g_idx in range(len(self.grasps)):
            grasp = self.grasps[g_idx]
            distance1 = self.pose_distance(grasp, pose)
            distance2 = self.pose_distance(grasp* Transform(Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0]), [0,0,0]), pose)
            distance = min(distance1, distance2)
            if distance < min_distance:
                best_idx = g_idx
                
                min_distance = distance
                closest_grasp = grasp
        print(f"Grasp {best_idx} distance: {distance}")
        return closest_grasp
    
    def linear_blending(self, pose: Transform, user_command: np.ndarray, alpha: float = 0.5):
        """
        Blends the user command with the closest grasp's pose.
        
        user_command: The command from the user (e.g., a desired pose).
        alpha: Blending factor (0.0 = only user command, 1.0 = only closest grasp).
        """
        
        closest_grasp = self.get_closest_grasp(pose)
        # eTep = pose.inverse() * closest_grasp
        predicted_velo = self.moving2pose(closest_grasp, pose)
        blended_command = np.zeros(6)
        blended_command = predicted_velo * (1 - alpha) +  user_command * alpha

        return blended_command

    
    def moving2pose(self, grasp_pose: Transform, pose: Transform):
        """
        Blends the user command with the closest grasp's pose.
        
        user_command: The command from the user (e.g., a desired pose).
        alpha: Blending factor (0.0 = only user command, 1.0 = only closest grasp).
        """
        # eTep = pose.inverse() * grasp_pose

        # Spatial error
        predicted_velo,_ = p_servo(pose.as_matrix(), grasp_pose.as_matrix(), gain=1.0, threshold=0.1)
        predicted_velo[:3] = pose.transform_vector(predicted_velo[:3])
        return predicted_velo
    
def p_servo(wTe, wTep, gain = 1.0, threshold=0.1):

    # Pose difference
    eTep = np.linalg.inv(wTe) @ wTep
    e = np.empty(6)

    # Translational error
    e[:3] = eTep[:3, -1]

    # Angular error
    e[3:] = base.tr2rpy(eTep, unit="rad", order="zyx", check=False)

    if base.isscalar(gain):
        k = gain * np.eye(6)
    else:
        k = np.diag(gain)

    v = k @ e
    arrived = True if np.sum(np.abs(e)) < threshold else False

    return v, arrived