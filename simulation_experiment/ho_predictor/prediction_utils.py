import numpy as np
import transformations as transmethods


def ApplyTwistToTransform(twist, transform, time=1.):
    """
    Apply a twist (linear and angular velocity) to a transform over a given time interval.
    """
    twist = np.asarray(twist).reshape(-1)
    assert transform.shape == (4, 4), "Transform must be a 4x4 matrix"
    assert twist.shape == (6,), "Twist must be a 6D vector"
    
    new_transform = transform.copy()
    new_transform[0:3,3] += time * twist[0:3]
    angular_velocity = twist[3:]
    angular_velocity_norm = np.linalg.norm(angular_velocity)
    if angular_velocity_norm > 1e-3:
        angle = time*angular_velocity_norm
        axis = angular_velocity/angular_velocity_norm
        new_transform[0:3,0:3] = np.dot(transmethods.rotation_matrix(angle, axis), new_transform)[0:3,0:3]
    return new_transform

def ApplyAngularVelocityToQuaternion(angular_velocity, quat, time=1.):
    """
    Apply angular velocity to a quaternion over a given time interval.
    """
    angular_velocity_norm = np.linalg.norm(angular_velocity)
    angle = time*angular_velocity_norm
    axis = angular_velocity/angular_velocity_norm
    #angle axis to quaternion formula
    quat_from_velocity = np.append(np.sin(angle/2.)*axis, np.cos(angle/2.))
    return transmethods.quaternion_multiply(quat_from_velocity, quat)

def AngleFromQuaternionW(w):
    w = min(0.9999999, max(-0.999999, w))
    phi = 2.*np.arccos(w)
    return min(phi, 2.* np.pi - phi)

def QuaternionDistance(quat1, quat2):
    quat_between = transmethods.quaternion_multiply(  quat2, transmethods.quaternion_inverse(quat1) )
    return AngleFromQuaternionW(quat_between[-1])

def pose_to_mat(pose):
    """
    Convert a Pose msg to a 4x4 numpy matrix
    """
    mat = np.zeros((4, 4))
    mat[3, 3] = 1
    mat[0:3, 3] = np.array([pose.position.x, pose.position.y, pose.position.z])
    q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    mat[0:3, 0:3] = np.array(transmethods.quaternion_matrix(q))[0:3, 0:3] 
    return mat

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)


class HuberCost():
  """
  Huber Assistance Policy \n
  Args: 
    pose: matrix of goal
  """
  def __init__(self, goal_pose):
    self.goal_pos = goal_pose[0:3, 3]
    self.goal_quat = transmethods.quaternion_from_matrix(goal_pose)
    self._set_parameters()

  def update(self, eef_pose, robot_state_after_action):
    self.dist_translation = np.linalg.norm(eef_pose[0:3,3] - self.goal_pos)
    self.dist_translation_aftertrans = np.linalg.norm(robot_state_after_action[0:3,3] - self.goal_pos)
    eef_quat = transmethods.quaternion_from_matrix(eef_pose)
    eef_quat_after_action = transmethods.quaternion_from_matrix(robot_state_after_action)
    self.dist_rotation = QuaternionDistance(eef_quat, self.goal_quat)
    self.dist_rotation_aftertrans = QuaternionDistance(eef_quat_after_action, self.goal_quat)

  # ====== Cost Function ======
  def get_cost_translation(self, dist_translation=None):
    if dist_translation is None:
      dist_translation = self.dist_translation

    if dist_translation > self.TRANSLATION_DELTA_SWITCH:
      return self.ACTION_APPLY_TIME * (self.TRANSLATION_LINEAR_COST_MULT_TOTAL)
    else:
      return self.ACTION_APPLY_TIME * (self.TRANSLATION_QUADRATIC_COST_MULTPLIER * dist_translation + self.TRANSLATION_CONSTANT_ADD)
  
  def get_cost_rotation(self, dist_rotation=None):
    if dist_rotation is None:
      dist_rotation = self.dist_rotation

    if dist_rotation > self.ROTATION_DELTA_SWITCH:
      return self.ACTION_APPLY_TIME * self.ROTATION_MULTIPLIER * self.ROTATION_LINEAR_COST_MULT_TOTAL
    else:
      return self.ACTION_APPLY_TIME * self.ROTATION_MULTIPLIER * (self.ROTATION_QUADRATIC_COST_MULTPLIER * dist_rotation + self.ROTATION_CONSTANT_ADD)

  # ====== Value Function ======
  def get_value_translation(self, dist_translation=None):
    if dist_translation is None:
      dist_translation = self.dist_translation

    if dist_translation <= self.TRANSLATION_DELTA_SWITCH:
      return self.TRANSLATION_QUADRATIC_COST_MULTPLIER_HALF * dist_translation*dist_translation + self.TRANSLATION_CONSTANT_ADD*dist_translation
    else:
      return 0.6 *(self.TRANSLATION_LINEAR_COST_MULT_TOTAL * dist_translation - self.TRANSLATION_LINEAR_COST_SUBTRACT)
    
  def get_value_rotation(self, dist_rotation=None):
    if dist_rotation is None:
      dist_rotation = self.dist_rotation

    if dist_rotation <= self.ROTATION_DELTA_SWITCH:
      return self.ROTATION_MULTIPLIER * (self.ROTATION_QUADRATIC_COST_MULTPLIER_HALF * dist_rotation*dist_rotation + self.ROTATION_CONSTANT_ADD*dist_rotation)
    else:
      return self.ROTATION_MULTIPLIER*(self.ROTATION_LINEAR_COST_MULT_TOTAL * dist_rotation - self.ROTATION_LINEAR_COST_SUBTRACT)
  
  # ====== Q-Value Function ======
  def get_qvalue_translation(self):
    return self.get_cost_translation() + self.get_value_translation(self.dist_translation_aftertrans)
  
  def get_qvalue_rotation(self):
    return self.get_cost_rotation() + self.get_value_rotation(self.dist_rotation_aftertrans)
  
  # C(x, u)
  def get_cost(self):
    return self.get_cost_translation() + self.get_cost_rotation()

  # V(x; g)
  def get_value(self):
    return self.get_value_translation() + self.get_value_rotation()

  # Q(x, u; g) = C(x, u) + V(x'; g)
  def get_qvalue(self):
    return self.get_qvalue_translation() + self.get_qvalue_rotation()
  
  def _set_parameters(self):
    self.ACTION_APPLY_TIME = 0.05
    # Translation parameters
    self.TRANSLATION_LINEAR_MULTIPLIER = 1.5
    self.TRANSLATION_DELTA_SWITCH = 0.10
    self.TRANSLATION_CONSTANT_ADD = 0.1
    self.TRANSLATION_LINEAR_COST_MULT_TOTAL = self.TRANSLATION_LINEAR_MULTIPLIER + self.TRANSLATION_CONSTANT_ADD
    self.TRANSLATION_LINEAR_COST_SUBTRACT = self.TRANSLATION_LINEAR_MULTIPLIER * self.TRANSLATION_DELTA_SWITCH * 0.5
    self.TRANSLATION_QUADRATIC_COST_MULTPLIER = self.TRANSLATION_LINEAR_MULTIPLIER / self.TRANSLATION_DELTA_SWITCH
    self.TRANSLATION_QUADRATIC_COST_MULTPLIER_HALF = 0.5 * self.TRANSLATION_QUADRATIC_COST_MULTPLIER
    # Rotation parameters
    self.ROTATION_LINEAR_MULTIPLIER = 0.20
    self.ROTATION_DELTA_SWITCH = np.pi/32.
    self.ROTATION_CONSTANT_ADD = 0.01
    self.ROTATION_MULTIPLIER = 0.07
    self.ROTATION_LINEAR_COST_MULT_TOTAL = self.ROTATION_LINEAR_MULTIPLIER + self.ROTATION_CONSTANT_ADD
    self.ROTATION_QUADRATIC_COST_MULTPLIER = self.ROTATION_LINEAR_MULTIPLIER/self.ROTATION_DELTA_SWITCH
    self.ROTATION_QUADRATIC_COST_MULTPLIER_HALF = 0.5 * self.ROTATION_QUADRATIC_COST_MULTPLIER
    self.ROTATION_LINEAR_COST_SUBTRACT = self.ROTATION_LINEAR_MULTIPLIER * self.ROTATION_DELTA_SWITCH * 0.5