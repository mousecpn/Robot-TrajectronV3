import torch

def quat_normalize(q):
    return q / (q.norm(dim=-1, keepdim=True) + 1e-8)

def quaternion_to_rotation_matrix(q):
    """
    q: (...,4) as (qx,qy,qz,qw)
    returns: (...,3,3)
    """
    q = quat_normalize(q)
    qx, qy, qz, qw = q.unbind(-1)
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    xw, yw, zw = qx*qw, qy*qw, qz*qw
    
    R = torch.stack([
        1 - 2*(yy + zz), 2*(xy - zw),     2*(xz + yw),
        2*(xy + zw),     1 - 2*(xx + zz), 2*(yz - xw),
        2*(xz - yw),     2*(yz + xw),     1 - 2*(xx + yy)
    ], dim=-1).reshape(q.shape[:-1] + (3,3))
    return R

def extract_t_r_from_poses(poses):
    """
    poses: (N,7) [tx,ty,tz,qx,qy,qz,qw]  OR (N,4,4) homogeneous
    returns: t (N,3), R (N,3,3)
    """
    if poses.ndim == 2 and poses.shape[1] == 7:
        t = poses[:,0:3]
        q = poses[:,3:7]
        R = quaternion_to_rotation_matrix(q)
        return t, R
    elif poses.ndim == 3 and poses.shape[1:] == (4,4):
        t = poses[:,0:3,3]
        R = poses[:,0:3,0:3]
        return t, R
    else:
        raise ValueError("Unsupported pose shape. Use (N,7) or (N,4,4).")

def rotation_angle_from_relative(R_rel):
    # angle = arccos((trace(R)-1)/2)
    tr = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    val = (tr - 1.0) / 2.0
    val = torch.clamp(val, -1.0, 1.0)
    return torch.acos(val)

def se3_pairwise_distance(poses, trans_weight=1.0, rot_weight=1.0):
    t, R = extract_t_r_from_poses(poses)
    diff = t[:,None,:] - t[None,:,:]   # (N,N,3)
    trans_d = diff.norm(dim=-1)        # (N,N)

    R_rel = R[:,None,:,:] @ R[None,:,:,:].transpose(-1,-2)  # (N,N,3,3)
    rot_angles = rotation_angle_from_relative(R_rel)        # (N,N)

    return trans_weight * trans_d + rot_weight * rot_angles

def fps_se3(poses, k, trans_weight=1.0, rot_weight=1.0, start_idx=None, random_seed=None):
    """
    Farthest Point Sampling on SE(3) poses.
    Args:
      poses: (N,7) or (N,4,4), torch tensor
      k: number of samples
      trans_weight, rot_weight: weights
      start_idx: optional int
      random_seed: for reproducibility
    Returns:
      indices: list of selected indices (length k)
    """
    device = poses.device
    N = poses.shape[0]
    if k > N:
        raise ValueError("k must be <= N")
    if random_seed is not None:
        torch.manual_seed(random_seed)
    D = se3_pairwise_distance(poses, trans_weight, rot_weight)  # (N,N)
    if start_idx is None:
        current = torch.randint(N, (1,), device=device).item()
    else:
        current = int(start_idx)
    selected = [current]
    min_dists = D[current].clone()
    for _ in range(1, k):
        idx = torch.argmax(min_dists).item()
        selected.append(idx)
        min_dists = torch.minimum(min_dists, D[idx])
    return selected

# --------- Demo ----------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import numpy as np
    def random_quats(n, device="cpu"):
        u1, u2, u3 = torch.rand(n, device=device), torch.rand(n, device=device), torch.rand(n, device=device)
        qx = torch.sqrt(1-u1) * torch.sin(2*torch.pi*u2)
        qy = torch.sqrt(1-u1) * torch.cos(2*torch.pi*u2)
        qz = torch.sqrt(u1) * torch.sin(2*torch.pi*u3)
        qw = torch.sqrt(u1) * torch.cos(2*torch.pi*u3)
        return torch.stack([qx,qy,qz,qw], dim=1)

    N = 200
    t = torch.empty((N,3), device=device).uniform_(-1,1)
    q = random_quats(N, device=device)
    poses = torch.cat([t, q], dim=1)  # (N,7)

    idxs = fps_se3(poses, k=12, trans_weight=1.0, rot_weight=0.5, random_seed=0)
    print("Selected indices:", idxs)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    poses = poses.cpu()
    ax.scatter(poses[:,0], poses[:,1], poses[:,2], s=10)  # all points
    sel = np.array(idxs)
    ax.scatter(poses[sel,0], poses[sel,1], poses[sel,2], s=80, marker='^')  # selected larger markers
    ax.set_title("SE(3) FPS (translation view)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    plt.show()
