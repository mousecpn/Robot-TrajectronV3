import numpy as np
import scipy.spatial.transform
try:
    import torch
    import torch.nn.functional as F
except:
    print("torch is not installed, some functions will not work.")
from theseus import SO3
import functools

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
    
def _index_from_letter(letter: str) -> int:
    """Return index 0/1/2 for X/Y/Z or raise ValueError."""
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError(f"Invalid axis letter '{letter}'. Expected one of 'X','Y','Z'.")

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Create a rotation matrix (or batch of matrices) for rotation about axis by `angle`.
    axis: "X", "Y" or "Z"
    angle: tensor of any shape (...), returns tensor of shape (..., 3, 3)
    """
    # ensure tensor
    angle = torch.as_tensor(angle)
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    # prepare an output tensor of shape angle.shape + (3,3)
    out_shape = angle.shape + (3, 3)
    R = torch.zeros(out_shape, dtype=angle.dtype, device=angle.device)

    if axis == "X":
        R[..., 0, 0] = 1
        R[..., 1, 1] = cos
        R[..., 1, 2] = -sin
        R[..., 2, 1] = sin
        R[..., 2, 2] = cos
    elif axis == "Y":
        R[..., 1, 1] = 1
        R[..., 0, 0] = cos
        R[..., 0, 2] = sin
        R[..., 2, 0] = -sin
        R[..., 2, 2] = cos
    elif axis == "Z":
        R[..., 2, 2] = 1
        R[..., 0, 0] = cos
        R[..., 0, 1] = -sin
        R[..., 1, 0] = sin
        R[..., 1, 1] = cos
    else:
        raise ValueError(f"Invalid axis '{axis}'. Expected 'X','Y','Z'.")

    return R

def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert Euler angles to rotation matrix.
    euler_angles: tensor shape (..., 3). The order of angles must match `convention`.
    convention: a string of three letters from {'X','Y','Z'}, e.g. "ZYX".
                The resulting rotation is R = R_{convention[0]}(angle0)
                                          @ R_{convention[1]}(angle1)
                                          @ R_{convention[2]}(angle2)
    Returns: tensor shape (..., 3, 3)
    """
    if not (isinstance(convention, str) and len(convention) == 3):
        raise ValueError("convention must be a 3-letter string like 'ZYX'.")

    for ch in convention:
        if ch not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {ch} in convention string.")

    euler_angles = torch.as_tensor(euler_angles)
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("euler_angles must have shape (..., 3).")

    # split into three angle tensors in the same order as convention
    angles = torch.unbind(euler_angles, dim=-1)  # gives tuple (angle0, angle1, angle2)

    # build rotation matrices in order and multiply
    mats = []
    for axis, ang in zip(convention, angles):
        mats.append(_axis_angle_rotation(axis, ang))

    # reduce with matrix multiplication in order: R = mats[0] @ mats[1] @ mats[2]
    R = functools.reduce(torch.matmul, mats)
    return R

def quaternion_to_matrix(quaternions):
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

def _sqrt_positive_part(x):
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


def expmap_to_rpyvel(T_base_cur, expmap, convention='ZYX'):
    """
    inputs are all torch tensors
    to relative euler angles velocity
    """
    if expmap.shape[-1] != 6:
        raise ValueError("expmap must have shape (..., 6).")
    T = SO3_R3().exp_map(expmap.reshape(-1,6)).to_matrix()
    # T = torch.inverse(T_base_cur).reshape(-1,4,4) @ T

    euler_angles = matrix_to_euler_angles(T[...,:3,:3], convention)
    angvel = torch.stack((euler_angles[:, _index_from_letter(convention[0])], euler_angles[:, _index_from_letter(convention[1])],euler_angles[:, _index_from_letter(convention[2])]), dim=-1)
    vel = torch.cat((T[..., :3, 3], angvel), dim=-1)

    return vel

def rotation_6d_to_matrix(d6):
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


def matrix_to_rotation_6d(matrix):
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
    
    @staticmethod
    def from_twist(xi):
        """Fast construction from twist vector [rho(3), omega(3)]"""
        xi = np.asarray(xi, dtype=float)
        rho = xi[:3]
        omega = xi[3:]
        theta = np.linalg.norm(omega)

        if theta < 1e-12:
            R = Rotation.identity()
            t = rho
        else:
            # Rotation part (fast, from rotvec)
            R = Rotation.from_rotvec(omega)

            # Precompute
            omega_unit = omega / theta
            rho = rho.astype(float)

            # Cross products
            omega_cross_rho = np.cross(omega, rho)
            omega_cross_omega_cross_rho = np.cross(omega, omega_cross_rho)

            # Translation using series expansion
            t = (rho
                 + (1 - np.cos(theta)) / (theta**2) * omega_cross_rho
                 + (theta - np.sin(theta)) / (theta**3) * omega_cross_omega_cross_rho)

        return Transform(R, t)

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



def irreps2rot(irrep):
        cos1 = irrep[..., 0:1]
        sin1 = irrep[..., 1:2]
        cos2 = irrep[..., 2:3]
        sin2 = irrep[..., 3:4]
        cos3 = irrep[..., 4:5]
        sin3 = irrep[..., 5:6]
        return torch.cat((cos1, cos2, cos3, sin1, sin2, sin3), dim=-1)

def rot2irreps(rot):
    cos1 = rot[..., 0:1]
    cos2 = rot[..., 1:2]
    cos3 = rot[..., 2:3]
    sin1 = rot[..., 3:4]
    sin2 = rot[..., 4:5]
    sin3 = rot[..., 5:6]
    return torch.cat((cos1, sin1, cos2, sin2, cos3, sin3), dim=-1)

def padding(data, num_grasps):
    """
    data: torch.tensor(n_grasps, dim)
    num_grasps: torch.tensor(bs, )
    """
    if len(data.shape) == 2:
        padded_data = torch.full((len(num_grasps), max(num_grasps), data.shape[-1]), float('nan'), device=data.device, dtype=torch.float32)
    elif len(data.shape) == 3:
        padded_data = torch.full((len(num_grasps), max(num_grasps), data.shape[-2], data.shape[-1]), float('nan'), device=data.device, dtype=torch.float32)
    count = 0
    for i in range(len(num_grasps)):
        padded_data[i, :num_grasps[i]] = data[count:count+num_grasps[i]]
        count += num_grasps[i]
    return padded_data

def negative_sampling(grasp, label=None, rotation=True):
    """
    grasp: (bs, ns ,dim)
    """
    neg_samples = torch.zeros_like(grasp[:,:0,:]) # (bs, 0, dim)
    # neg_sample = grasp.clone()
    bs, ns = grasp.shape[0], grasp.shape[1]
    sample_type = np.random.choice([0,1])
    # if rotation is False:
    #     trans_perturb_level = 0.3
    # else:
    trans_perturb_level = 0.1
    rot_perturb_level = 0.5
    num_trans_samples = 10
    num_rotations = 5
    # neg_label = label.clone()

    if rotation is True:
        yaws = np.linspace(0.0, np.pi, num_rotations)
        for yaw in yaws[1:-1]:
            neg_sample = grasp.clone()
            z_rot = Rotation.from_euler("z", yaw)
            R = Rotation.from_matrix(rotation_6d_to_matrix(neg_sample[..., 3:]).reshape(-1,3,3).detach().cpu().numpy())
            # R = Rotation.from_quat(neg_sample[..., 3:].reshape(-1,4).detach().cpu().numpy())

            neg_rot = (R*z_rot).as_matrix()
            neg_rot = torch.from_numpy(neg_rot.astype('float32')).to(grasp.device)

            # noise = torch.randn_like(grasp[...,3:]) * rot_perturb_level
            # neg_sample[..., 3:] += noise
            neg_sample[..., 3:] = matrix_to_rotation_6d(neg_rot.reshape(bs,ns,3,3))
            neg_samples = torch.cat((neg_samples, neg_sample), dim=1)

    for i in range(num_trans_samples):
        neg_sample = grasp.clone()
        noise = torch.randn_like(grasp[...,:3]) * trans_perturb_level
        neg_sample[..., :3] += noise
        neg_samples = torch.cat((neg_samples, neg_sample), dim=1)
        if rotation is True:
            yaws = np.linspace(0.0, np.pi, num_rotations)
            yaw = np.random.choice(yaws[1:-1])
            neg_sample = grasp.clone()
            z_rot = Rotation.from_euler("z", yaw)
            R = Rotation.from_matrix(rotation_6d_to_matrix(neg_sample[..., 3:]).reshape(-1,3,3).detach().cpu().numpy())
            # R = Rotation.from_quat(neg_sample[..., 3:].reshape(-1,4).detach().cpu().numpy())

            neg_rot = (R*z_rot).as_matrix()
            neg_rot = torch.from_numpy(neg_rot.astype('float32')).to(grasp.device)

            # noise = torch.randn_like(grasp[...,3:]) * rot_perturb_level
            # neg_sample[..., 3:] += noise
            neg_sample[..., 3:] = matrix_to_rotation_6d(neg_rot.reshape(bs,ns,3,3))
            neg_samples = torch.cat((neg_samples, neg_sample), dim=1)

    return neg_samples, torch.zeros_like(neg_samples[...,0])


def eulerZYX_vel_to_twist(T, euler_vel):
    """
    输入:
        T : 4x4 位姿矩阵 (SE(3))
        euler_rates : [dphi, dtheta, dpsi] (欧拉角速度, rad/s)
    输出:
        xi : 6x1 李代数速度向量 (twist) = [v; ω]
    """
    # 提取旋转矩阵
    R = T[:3, :3]

    # 从旋转矩阵反解欧拉角 (ZYX 顺序: yaw-pitch-roll)
    psi = np.arctan2(R[1,0], R[0,0])                     # yaw
    theta = np.arcsin(-R[2,0])                           # pitch
    phi = np.arctan2(R[2,1], R[2,2])                     # roll

    dphi, dtheta, dpsi = euler_vel
    dq = np.array([dphi, dtheta, dpsi])

    # ZYX 欧拉角速率 -> body 角速度
    T_map = np.array([
        [1, 0, -np.sin(theta)],
        [0, np.cos(phi), np.sin(phi)*np.cos(theta)],
        [0, -np.sin(phi), np.cos(phi)*np.cos(theta)]
    ])
    omega_body = T_map @ dq

    # 线速度（这里没有输入，先设为 0）
    v_body = np.zeros(3)

    # twist = [v; ω] in body frame
    xi = np.hstack((v_body, omega_body))
    return xi


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

gripper_flip = torch.eye(4, dtype=torch.float32)
gripper_flip[:3,:3] = torch.tensor(Rotation.from_euler("z", np.pi).as_matrix(), dtype=torch.float32)

def select_grasps(grasps, T=None):
    """
    Select the grasp with the smallest e metric between original and flipped version
    grasps: [N, 4, 4] in current base frame
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



def achieved(cur_pose, goal_poses, threshold=0.1):
    """
    Determine if the current pose has achieved any of the goal poses.
    Args:
        cur_pose (torch.Tensor): Current end-effector pose of shape (4, 4).
        goal_poses (torch.Tensor): Goal poses of shape (N, 4, 4).
    Returns:
        bool: True if achieved, False otherwise.
        torch.Tensor: The target goal pose that was achieved.
    """
    error_mat = torch.inverse(cur_pose) @ goal_poses
    e1 = torch.cat((error_mat[..., :3,3], matrix_to_euler_angles(error_mat[...,:3,:3], 'ZYX')/np.pi*0.6), dim=-1).abs().sum(-1)
    idx1 = torch.argmin(e1)

    error_mat = torch.inverse(cur_pose) @ (goal_poses @ gripper_flip.unsqueeze(0).to(goal_poses.device))
    e2 = torch.cat((error_mat[..., :3,3], matrix_to_euler_angles(error_mat[...,:3,:3], 'ZYX') /np.pi*0.6), dim=-1).abs().sum(-1)
    idx2 = torch.argmin(e2)

    if e1[idx1] < e2[idx2]:
        e = e1[idx1]
        target_pose = goal_poses[idx1]
    else:
        e = e2[idx2]
        target_pose = goal_poses[idx2] @ gripper_flip.to(goal_poses.device)

    if e < threshold:
        return True, target_pose
    else:
        return False, target_pose