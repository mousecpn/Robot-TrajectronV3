import numpy as np
import torch
import math

def _dh_transform_batch(a, d, alpha, theta):
    """
    Batch DH transform using the convention from the referenced gist.
    a,d,alpha: scalars (floats) or tensors broadcastable to theta
    theta: (bs,) tensor
    returns: (bs,4,4)
    """
    # ensure theta is tensor
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, dtype=torch.get_default_dtype())
    ct = torch.cos(theta)
    st = torch.sin(theta)
    ca = math.cos(alpha) if isinstance(alpha, float) else torch.cos(alpha)
    sa = math.sin(alpha) if isinstance(alpha, float) else torch.sin(alpha)

    # build 4x4 matrix per batch
    # Using the same layout as the gist:
    # [[ cos t, -sin t, 0, a],
    #  [ sin t * cos a, cos t * cos a, -sin a, -sin a * d],
    #  [ sin t * sin a, cos t * sin a,  cos a,  cos a * d],
    #  [ 0, 0, 0, 1]]
    bs = theta.shape[0]
    A = torch.zeros((bs,4,4), dtype=theta.dtype, device=theta.device)
    A[:,0,0] = ct
    A[:,0,1] = -st
    A[:,0,2] = 0.0
    A[:,0,3] = a

    A[:,1,0] = st * ca
    A[:,1,1] = ct * ca
    A[:,1,2] = -sa
    A[:,1,3] = -sa * d

    A[:,2,0] = st * sa
    A[:,2,1] = ct * sa
    A[:,2,2] = ca
    A[:,2,3] = ca * d

    A[:,3,3] = 1.0
    return A

def manipulability_panda(qs: torch.Tensor, trans_eps: float = 1e-12) -> torch.Tensor:
    """
    Compute Yoshikawa manipulability for Franka-Emika Panda.

    Args:
        qs: (bs,7) joint angles in radians, torch tensor
        trans_eps: small epsilon for numerical stability before sqrt

    Returns:
        mani: (bs,) manipulability scalar (torch tensor, same device/dtype as qs)
    """
    assert qs.ndim == 2 and qs.shape[1] == 7, "qs must be (bs,7)"
    device = qs.device
    dtype = qs.dtype

    # DH params taken from community gist (modified DH style)
    # Each entry: [a, d, alpha, theta_fixed]
    # first 7 entries are revolute joints (theta comes from qs),
    # the last 3 are fixed frames/tool offsets to get end-effector pose.
    dh = [
        (0.0,    0.333,  0.0,       0.0),
        (0.0,    0.0,   -math.pi/2, 0.0),
        (0.0,    0.316,  math.pi/2, 0.0),
        (0.0825, 0.0,    math.pi/2, 0.0),
        (-0.0825,0.384, -math.pi/2, 0.0),
        (0.0,    0.0,    math.pi/2, 0.0),
        (0.088,  0.0,    math.pi/2, 0.0),
        (0.0,    0.107,  0.0,       0.0),
        (0.0,    0.0,    0.0,      -math.pi/4),
        (0.0,    0.1034, 0.0,       0.0),
    ]
    n_frames = len(dh)  # 10

    bs = qs.shape[0]

    # Initialize batch identity transform
    T = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(bs,1,1)  # (bs,4,4)

    p_list = []  # will collect (bs,3) for each joint origin (before its transform)
    z_list = []  # will collect (bs,3) for each joint axis (z) in base frame

    # iterate frames; for frames i<7 the theta is qs[:,i]; for later frames use fixed theta
    for i in range(n_frames):
        a, d, alpha, theta_fixed = dh[i]
        # record joint origin & axis for joint frames (only for first 7 joints)
        if i < 7:
            p_i = T[:, 0:3, 3].clone()   # (bs,3)
            z_i = T[:, 0:3, 2].clone()   # (bs,3) third column is z-axis
            p_list.append(p_i)
            z_list.append(z_i)

            theta = qs[:, i] + float(theta_fixed)
        else:
            theta = torch.full((bs,), float(theta_fixed), dtype=dtype, device=device)

        # create A_i (bs,4,4)
        theta_b = theta.to(dtype=dtype, device=device)
        A_i = _dh_transform_batch(a, d, alpha, theta_b)  # (bs,4,4)
        # update T
        T = torch.matmul(T, A_i)

    # end-effector position
    p_end = T[:, 0:3, 3]  # (bs,3)

    # stack p_list and z_list to shapes (bs,7,3)
    p_stack = torch.stack(p_list, dim=1)  # (bs,7,3)
    z_stack = torch.stack(z_list, dim=1)  # (bs,7,3)

    # compute Jv_j = z_j x (p_end - p_j)
    # p_end expand: (bs,1,3)
    r = p_end.unsqueeze(1) - p_stack  # (bs,7,3)
    # cross product along last dim
    Jv = torch.cross(z_stack, r, dim=-1)  # (bs,7,3)
    Jv = Jv.permute(0,2,1)                 # (bs,3,7)

    # Jw is z_stack
    Jw = z_stack.permute(0,2,1)            # (bs,3,7)

    # full Jacobian (bs,6,7)
    J = torch.cat([Jv, Jw], dim=1)

    # compute J * J^T for each batch -> (bs,6,6)
    JJT = torch.matmul(J, J.transpose(-2,-1))

    # determinant per batch (may be slightly negative due to numerical issues) -- clamp
    # use torch.linalg.det (newer) or torch.det
    det = torch.linalg.det(JJT)
    det = torch.clamp(det, min=0.0)
    mani = torch.sqrt(det + trans_eps)  # (bs,)

    return mani


def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))


def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n, device=m.device).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(siz0 + torch.Size(torch.tensor(siz1) * n))


def tile(a, dim, n_tile, device='cpu'):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
    return torch.index_select(a, dim, order_index)


def rotation_angle_from_relative(R_rel: torch.Tensor) -> torch.Tensor:
    """
    R_rel: (...,3,3)
    returns: (...,) rotation angle in radians
    """
    tr = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)  # trace
    val = (tr - 1.0) / 2.0
    val = torch.clamp(val, -1.0, 1.0)  # numerical safety
    return torch.acos(val)

def se3_pairwise_distance(A: torch.Tensor, B: torch.Tensor, trans_weight=1.0, rot_weight=1.0) -> torch.Tensor:
    """
    Compute pairwise SE(3) distances between two sets of poses with arbitrary batch dims.

    Args:
        A: (..., N, 4, 4) homogeneous transforms
        B: (..., M, 4, 4) homogeneous transforms
        trans_weight: weight for translation part
        rot_weight: weight for rotation part
    Returns:
        D: (..., N, M) distance matrix
    """
    assert A.shape[-2:] == (4,4) and B.shape[-2:] == (4,4), "Inputs must be (...,4,4)"

    # unify batch shapes
    batch_shape = torch.broadcast_shapes(A.shape[:-3], B.shape[:-3])
    N, M = A.shape[-3], B.shape[-3]

    # expand to common batch
    A = A.expand(*batch_shape, N, 4, 4)
    B = B.expand(*batch_shape, M, 4, 4)

    tA, RA = A[...,0:3,3], A[...,0:3,0:3]  # (...,N,3), (...,N,3,3)
    tB, RB = B[...,0:3,3], B[...,0:3,0:3]  # (...,M,3), (...,M,3,3)

    # translation distances
    diff = tA.unsqueeze(-2) - tB.unsqueeze(-3)  # (...,N,M,3)
    trans_d = diff.norm(dim=-1)                 # (...,N,M)

    # rotation distances
    R_rel = RA.unsqueeze(-3) @ RB.unsqueeze(-4).transpose(-1,-2)  # (...,N,M,3,3)
    rot_angles = rotation_angle_from_relative(R_rel)              # (...,N,M)

    return trans_weight * trans_d + rot_weight * rot_angles


def furthest_point_sampling(points, num_samples):
    """
    最远点采样 (FPS) 的 PyTorch 实现 (修复维度问题)。
    Args:
        points: (B, P, D) 或 (P, D) 的张量，点集。
        num_samples: 要采样的点数。
    Returns:
        (B, num_samples, D) 或 (num_samples, D) 的张量，采样的点集。
    """
    is_2d_input = (points.dim() == 2)
    if is_2d_input:
        # 添加 Batch 维度
        points = points.unsqueeze(0)
    
    B, P, D = points.shape
    device = points.device
    
    if P < num_samples:
        # 如果点数不够，直接返回
        return points.repeat(1, (num_samples + P - 1) // P, 1)[:, :num_samples, :][0]

    # 1. 初始化
    indices = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    
    # 随机选择第一个点 (B,)
    first_idx = torch.randint(P, (B,), device=device) 
    indices[:, 0] = first_idx
    
    # 构造批次索引 (B,)，用于高级索引
    batch_indices = torch.arange(B, device=device) 
    
    # 获取第一个采样的点 (B, 1, D)
    # 使用高级索引: points[batch_indices, first_idx] 形状为 (B, D)
    initial_sample = points[batch_indices, first_idx].unsqueeze(1) 
    
    # 计算初始距离平方 (B, P)
    distances = torch.sum((points - initial_sample) ** 2, dim=-1) 

    # 2. 迭代采样
    for i in range(1, num_samples):
        # 找到最大距离的点作为下一个采样点
        # farthest_idx 形状为 (B,)
        farthest_idx = torch.argmax(distances, dim=1) 
        
        # 修复点: farthest_idx 形状为 (B,)，indices[:, i] 期望 (B,)，匹配正确
        indices[:, i] = farthest_idx 
        
        # 获取当前采样的点 (B, D)
        # 再次使用高级索引: points[batch_indices, farthest_idx] 形状为 (B, D)
        current_sample = points[batch_indices, farthest_idx]
        
        # 计算新采样点到所有点的距离平方 (B, P)
        # current_sample (B, D) 广播到 points (B, P, D)
        new_distances = torch.sum((points - current_sample.unsqueeze(1)) ** 2, dim=-1) 
        
        # 更新 distances: distances = min(old_distances, new_distances)
        distances = torch.min(distances, new_distances)
    
    # 3. 收集采样的点
    
    # 构造最终的索引张量 (B, num_samples)
    final_indices = indices
    
    # 构造重复的批次索引 (B, num_samples)
    # (0, 0, ..., 1, 1, ..., B-1, B-1, ...)
    # PyTorch 高级索引：首先展平所有索引张量，然后进行采样
    
    # 展平索引，方便高级索引
    flat_indices = final_indices.view(-1)  # (B*num_samples,)
    flat_batch_indices = batch_indices.unsqueeze(1).repeat(1, num_samples).view(-1) # (B*num_samples,)
    
    # 采样出的点 (B*num_samples, D)
    sampled_points_flat = points[flat_batch_indices, flat_indices]
    
    # 重塑为 (B, num_samples, D)
    sampled_points = sampled_points_flat.view(B, num_samples, D)
    
    # 如果输入是 2D，则返回 2D
    if is_2d_input:
        return sampled_points.squeeze(0)
        
    return sampled_points


def collision_avoidance_regularizer(
    robot_trajectory_se3: torch.Tensor, 
    obstacle_pcls: torch.Tensor,
    batch_idx: torch.Tensor = None, 
    margin: float = 0.05, 
    margin_loss_weight: float = 100.0
) -> torch.Tensor:
    BS, N, _ = robot_trajectory_se3.shape
    device = robot_trajectory_se3.device
    
    # 1. 简化机器人表示：只取 3D 坐标
    # (BS, N, 3) - 将整个轨迹视为一组关键点
    robot_points = robot_trajectory_se3[:, :, :3]

    losses = []

    for b_i in range(BS):
        pcl = obstacle_pcls[batch_idx==b_i]
        if pcl.shape[0] == 0:
            losses.append(torch.tensor(0.0, device=device))
            continue
        robot_points_b = robot_points[b_i]  # (N, 3)
        # 2. 计算机器人轨迹点与障碍物点云的距离矩阵
        dist = torch.cdist(robot_points_b, pcl).squeeze(0)  # (N, P)
        loss = torch.relu(margin - dist)  # (N, P)
        losses.append(loss.sum())
    return torch.stack(losses).mean() * margin_loss_weight
    



# --- 示例用法 ---
if __name__ == '__main__':
    # 设定参数
    BS = 4  # Batch Size
    N = 10  # 轨迹长度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 构造机器人轨迹 (BS, N, 6)
    # 假设前三维是 (x, y, z)，数值在 [0, 1] 范围内
    robot_trajectory_se3 = torch.rand(BS, N, 6, device=device, requires_grad=True) 

    # 2. 构造障碍物点云列表
    # 障碍物 A: (1000, 3)
    pcl_A = torch.rand(1000, 3, device=device) * 2 - 1.0  # 假设在 [-1, 1] 之间
    # 障碍物 B: (5000, 3)
    pcl_B = torch.rand(5000, 3, device=device) * 0.5 + 0.5 # 假设在 [0.5, 1.0] 之间
    # 障碍物 C: (100, 3)
    pcl_C = torch.rand(100, 3, device=device) * 0.1 
    
    obstacle_pcls = [pcl_A, pcl_B, pcl_C]

    # 3. 计算碰撞损失
    collision_loss = collision_avoidance_regularizer(
        robot_trajectory_se3=robot_trajectory_se3,
        obstacle_pcls=obstacle_pcls,
        margin=0.1,  # 碰撞安全裕度
        k_nn=1,      # 只考虑最近的一个点
        fps_points=200 # 对每个障碍物最多采样 200 个点
    )

    print(f"Batch Size: {BS}, Trajectory Length: {N}")
    print(f"FPS 采样点数: {200} (B点云从 5000 -> 200)")
    print(f"合并后的有效障碍物点云总数: {pcl_A.shape[0] + 200 + pcl_C.shape[0]}")
    print(f"最终碰撞避免正则化损失: {collision_loss.item():.6f}")

    # 4. 验证反向传播
    # 随机生成一个假的目标损失
    target_loss = torch.tensor(0.5, device=device)
    # 模拟总损失
    total_loss = collision_loss + target_loss
    
    # 反向传播，检查是否能计算梯度
    total_loss.backward()
    
    print(f"机器人轨迹的梯度是否已计算: {robot_trajectory_se3.grad is not None}")
    if robot_trajectory_se3.grad is not None:
        print(f"轨迹梯度范数: {robot_trajectory_se3.grad.norm().item():.6f}")