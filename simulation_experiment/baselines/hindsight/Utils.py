import numpy as np
from scipy.spatial.transform import Rotation as R
# from geometry_msgs.msg import Pose, Twist
from . import GoalAssistance

def RotationMatrixDistance(pose1, pose2):
    quat1 = R.from_matrix(pose1[:3,:3]).as_quat()
    quat2 = R.from_matrix(pose2[:3,:3]).as_quat()
    return QuaternionDistance(quat1, quat2)

def QuaternionDistance(quat1, quat2):
    quat_between = (R.from_quat(quat2) * R.from_quat(quat1).inv()).as_quat()
    return AngleFromQuaternionW(quat_between[-1])

def AngleFromQuaternionW(w):
    w = min(0.9999999, max(-0.999999, w))
    phi = 2.*np.arccos(w)
    return min(phi, 2.* np.pi - phi)

def ApplyTwistToTransform(twist, transform, time=1.):
    transform[0:3,3] += time * twist[0:3]

    angular_velocity = twist[3:]
    angular_velocity_norm = np.linalg.norm(angular_velocity)
    if angular_velocity_norm > 1e-3:
        angle = time*angular_velocity_norm
        axis = angular_velocity/angular_velocity_norm
        transform[0:3,0:3] = np.dot(R.from_rotvec(angle*axis).as_matrix(), transform[0:3,0:3])

    return transform

def ApplyAngularVelocityToQuaternion(angular_velocity, quat, time=1.):
    angular_velocity_norm = np.linalg.norm(angular_velocity)
    angle = time*angular_velocity_norm
    axis = angular_velocity/angular_velocity_norm

    #angle axis to quaternion formula
    quat_from_velocity = np.append(np.sin(angle/2.)*axis, np.cos(angle/2.))

    return (R.from_quat(quat_from_velocity)*R.from_quat(quat)).as_quat()


def pose_to_mat(pose):
    """
    Convert a Pose msg to a 4x4 numpy matrix \n
    Args:
        pose: Pose msg
    Return: numpy matrix
    """
    mat = np.zeros((4, 4))
    mat[3, 3] = 1
    mat[0:3, 3] = np.array([pose.position.x, pose.position.y, pose.position.z])
    q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    mat[0:3, 0:3] = R.from_quat(q).as_matrix()[0:3, 0:3]
    return mat


# def mat_to_pose(mat):
#     """
#     Convert 4x4 numpy matrix to a Pose msg \n
#     Args:
#         mat: numpy matrix
#     Return: Pose msg
#     """
#     pose = Pose()
#     pose.position.x = mat[0,3]
#     pose.position.y = mat[1,3]
#     pose.position.z = mat[2,3]
#     quat = transmethods.quaternion_from_matrix(mat)
#     pose.orientation.x = quat[0]
#     pose.orientation.y = quat[1]
#     pose.orientation.z = quat[2]
#     pose.orientation.w = quat[3]
#     return pose


# def arrayToTwistMsg(array):
#   """
#   Convert numpy array into twist message \n
#   Args:
#     array: numpy array
#   Return: twist message 
#   """
#   twist = Twist()
#   twist.linear.x = array[0]
#   twist.linear.y = array[1]
#   twist.linear.z = array[2] 
#   twist.angular.x = array[3]
#   twist.angular.y = array[4]
#   twist.angular.z = array[5]
#   return twist


def getGoal(goal_msg):
    """
    Create and get list of GoalAssistance obj \n
    Args:
        goal_msg: msg of goal
    Return: list of GoalAssistance
    """  
    goal_list = []
    target_pos = []
    for g in goal_msg.goal:
        grasp_points = []
        id = g.id
        center = pose_to_mat(g.center)
        tmp_c = np.array([g.center.position.x, g.center.position.y, g.center.position.z])
        target_pos.append(tmp_c)
        for grasp in g.grasping_points:
            grasp_points.append(grasp)
        goal = GoalAssistance.GoalAssistance(id, center, grasp_points)
        goal_list.append(goal)
    
    return goal_list

def get_closer_symmetry_goal_quat(goal_quat, curr_quat):
    # TODO: find out whether this should also be used in active orientation
    R_goal = R.from_quat(goal_quat)
    R_curr = R.from_quat(curr_quat)

    # 180° about z: quaternion [0, 0, 1, 0]
    R_flip_z180 = R.from_quat([0, 0, 1, 0])

    # Rotate about the goal's *local* z by post-multiplying
    R_goal_alt = R_goal * R_flip_z180

    # Geodesic distances (rotation magnitudes) to current orientation
    ang_orig = (R_goal.inv() * R_curr).magnitude()
    ang_alt  = (R_goal_alt.inv() * R_curr).magnitude()

    return (R_goal_alt if ang_alt < ang_orig else R_goal).as_quat()