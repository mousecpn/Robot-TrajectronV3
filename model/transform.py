import numpy as np
import scipy.spatial.transform
import torch
import torch.nn.functional as F
import torch

import numpy as np
import torch
import math
import json
import functools

import theseus as th
from theseus.geometry import SO3

def relative_pose_logmap_se3(T_A_logmap: torch.Tensor, T_B_logmap: torch.Tensor) -> torch.Tensor:
    """
    计算相对位姿的 logmap 表示：xi_rel = log(T_B^{-1} T_A)

    Args:
        T_A_logmap (torch.Tensor): 批处理的N个位姿 logmap (B, N, 6)
        T_B_logmap (torch.Tensor): 批处理的参考位姿 logmap (B, 6)

    Returns:
        torch.Tensor: 相对位姿 logmap (B, N, 6)
    """
    B, N, _ = T_A_logmap.shape
    
    # --- 步骤 1: 将 logmap 转化为 SE(3) 矩阵 ---
    
    # 1.1 展平 T_A，方便批处理 exp_map
    T_A_flat_logmap = T_A_logmap.reshape(B * N, 6)
    
    # 计算 T_A (B*N, 4, 4)
    T_A_flat = SO3_R3().exp_map(T_A_flat_logmap).to_matrix()    
    
    # 计算 T_B (B, 4, 4)
    T_B = SO3_R3().exp_map(T_B_logmap).to_matrix()
    
    # --- 步骤 2: 计算逆矩阵和相对位姿 ---
    
    # 2.1 计算 T_B 的逆 T_B_inv (B, 4, 4)
    R_B_inv = T_B[:, 0:3, 0:3].transpose(-1, -2) # R^T
    t_B = T_B[:, 0:3, 3].unsqueeze(2)            # (B, 3, 1)
    t_B_inv = -torch.bmm(R_B_inv, t_B).squeeze(2) # -R^T * t
    
    T_B_inv = torch.eye(4, dtype=T_B.dtype, device=T_B.device).unsqueeze(0).repeat(B, 1, 1)
    T_B_inv[:, 0:3, 0:3] = R_B_inv
    T_B_inv[:, 0:3, 3] = t_B_inv
    
    # 2.2 扩展 T_B_inv 以匹配 T_A 的形状 (B*N, 4, 4)
    T_B_inv_expanded = T_B_inv.unsqueeze(1).repeat(1, N, 1, 1).reshape(B * N, 4, 4)
    
    # 2.3 计算相对位姿 T_rel = T_B_inv @ T_A (B*N, 4, 4)
    T_rel_flat = torch.bmm(T_B_inv_expanded, T_A_flat)
    
    # --- 步骤 3: 将 SE(3) 矩阵转化为 logmap ---
    
    # 计算相对位姿 logmap (B*N, 6)
    T_rel_logmap_flat = SO3_R3.from_matrix(T_rel_flat).log_map()
    
    # 还原形状 (B, N, 6)
    T_rel_logmap = T_rel_logmap_flat.reshape(B, N, 6)
    
    return T_rel_logmap


class SO3_R3():
    def __init__(self, R=None, t=None):
        self.R = SO3()
        if R is not None:
            self.R.update(R)
        self.w = self.R.log_map()
        if t is not None:
            self.t = t

    def log_map(self):
        return torch.cat((self.t, self.w), -1)

    @staticmethod
    def from_matrix(matrix):
        """
        Convert a 4x4 matrix to SO3_R3.
        :param matrix: 4x4 matrix
        :return: SO3_R3 object
        """
        assert matrix.shape[-2:] == (4, 4), "Input must be a 4x4 matrix."
        R = SO3(tensor=matrix[:,:3, :3])
        t = matrix[:, :3, 3]
        return SO3_R3(R=R, t=t)
    
    def exp_map(self, x):
        self.t = x[..., :3]
        self.w = x[..., 3:]
        self.R = SO3().exp_map(self.w)
        return self

    def to_matrix(self):
        H = torch.eye(4).unsqueeze(0).repeat(self.t.shape[0], 1, 1).to(self.t)
        H[:, :3, :3] = self.R.to_matrix()
        H[:, :3, -1] = self.t
        return H
        

    # The quaternion takes the [w x y z] convention
    def to_quaternion(self):
        return self.R.to_quaternion()

    def sample(self, batch=1):
        R = SO3().rand(batch)
        t = torch.randn(batch, 3)
        H = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1).to(t)
        H[:, :3, :3] = R.to_matrix()
        H[:, :3, -1] = t
        return H
    

def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2

def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles, convention: str):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)

def matrix_to_euler_angles(matrix, convention: str):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.
    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.
    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])

    return torch.stack((o1, o2, o3, o0), -1)

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


class Rotation(scipy.spatial.transform.Rotation):
    @classmethod
    def identity(cls):
        return cls.from_quat([0.0, 0.0, 0.0, 1.0])


class Transform(object):
    """Rigid spatial transform between coordinate systems in 3D space.

    Attributes:
        rotation (scipy.spatial.transform.Rotation)
        translation (np.ndarray)
    """

    def __init__(self, rotation, translation):
        assert isinstance(rotation, scipy.spatial.transform.Rotation)
        assert isinstance(translation, (np.ndarray, list))

        self.rotation = rotation
        self.translation = np.asarray(translation, np.double)

    def as_matrix(self):
        """Represent as a 4x4 matrix."""
        return np.vstack(
            (np.c_[self.rotation.as_matrix(), self.translation], [0.0, 0.0, 0.0, 1.0])
        )

    def to_dict(self):
        """Serialize Transform object into a dictionary."""
        return {
            "rotation": self.rotation.as_quat().tolist(),
            "translation": self.translation.tolist(),
        }

    def to_list(self):
        return np.r_[self.rotation.as_quat(), self.translation]

    def __mul__(self, other):
        """Compose this transform with another."""
        rotation = self.rotation * other.rotation
        translation = self.rotation.apply(other.translation) + self.translation
        return self.__class__(rotation, translation)

    def transform_point(self, point):
        return self.rotation.apply(point) + self.translation

    def transform_vector(self, vector):
        return self.rotation.apply(vector)

    def inverse(self):
        """Compute the inverse of this transform."""
        rotation = self.rotation.inv()
        translation = -rotation.apply(self.translation)
        return self.__class__(rotation, translation)

    @classmethod
    def from_matrix(cls, m):
        """Initialize from a 4x4 matrix."""
        rotation = Rotation.from_matrix(m[:3, :3])
        translation = m[:3, 3]
        return cls(rotation, translation)

    @classmethod
    def from_dict(cls, dictionary):
        rotation = Rotation.from_quat(dictionary["rotation"])
        translation = np.asarray(dictionary["translation"])
        return cls(rotation, translation)

    @classmethod
    def from_list(cls, list):
        rotation = Rotation.from_quat(list[:4])
        translation = list[4:]
        return cls(rotation, translation)

    @classmethod
    def identity(cls):
        """Initialize with the identity transformation."""
        rotation = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
        translation = np.array([0.0, 0.0, 0.0])
        return cls(rotation, translation)

    @classmethod
    def look_at(cls, eye, center, up):
        """Initialize with a LookAt matrix.

        Returns:
            T_eye_ref, the transform from camera to the reference frame, w.r.t.
            which the input arguments were defined.
        """
        eye = np.asarray(eye)
        center = np.asarray(center)

        forward = center - eye
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        up = np.asarray(up) / np.linalg.norm(up)
        up = np.cross(right, forward)

        m = np.eye(4, 4)
        m[:3, 0] = right
        m[:3, 1] = -up
        m[:3, 2] = forward
        m[:3, 3] = eye

        return cls.from_matrix(m).inverse()

gripper_flip = torch.eye(4, dtype=torch.float32)
gripper_flip[:3,:3] = torch.tensor(Rotation.from_euler("z", np.pi).as_matrix(), dtype=torch.float32)

def select_grasps(grasps, T=None):
    """
    Select the grasp with the smallest e metric between original and flipped version
    grasps: [N, 4,4] in current base frame
    """
    if T is not None:
        grasps = torch.inverse(T) @ grasps
    flip_grasps = grasps @ gripper_flip.to(grasps.device)
    e = torch.cat((grasps[..., :3,3], matrix_to_euler_angles(grasps[...,:3,:3], 'XYZ')), dim=-1).abs().sum(-1)

    e2 = torch.cat((flip_grasps[..., :3,3], matrix_to_euler_angles(flip_grasps[...,:3,:3], 'XYZ')), dim=-1).abs().sum(-1)
    
    selected_grasps = torch.where((e2 < e).unsqueeze(-1).unsqueeze(-1), flip_grasps, grasps)
    if T is not None:
        selected_grasps = T @ selected_grasps
    return selected_grasps