import torch
from model.transform import SO3_R3
from model.gmm6d import GMMSE3


class SE3Integrator():
    def __init__(self, dt, dyn_limits, device):
        self.init_pose = None
        self.dt = dt
        self.device = device
        self.dyn_limits = dyn_limits

    def init_conditions(self, init_pose):
        """
            init_pose: 6d, logmap
        """
        self.init_pose = init_pose
        
    def integrate_one_step(self, xi, T_prev):
        """
        将单个 twist 积分为 SE(3) 变换。

        参数：
        xi:     shape (N,6) 的 ndarray, 每行是一次 twist [vx,vy,vz, ωx,ωy,ωz]
        T_prev: shape (N,6) 的 ndarray, 上一时刻的位姿, logmap 表示
        返回：
        T_next: shape (N,6) 的 ndarray, 下一时刻的位姿
        """
        N = xi.shape[0]
        xi = xi.reshape(-1, 6)
        if T_prev is None:
            T_prev_mat = torch.eye(4).reshape(1,4,4).to(xi.device)
            T_prev_mat = T_prev_mat.repeat(N, 1, 1)
        else:
            T_prev_mat = SO3_R3().exp_map(T_prev.reshape(-1,6)).to_matrix().reshape(-1,4,4)

        dT = SO3_R3().exp_map(xi * self.dt).to_matrix().reshape(N, 4, 4)
        # left‐multiply（body frame）或 right‐multiply（space frame）看你的 twist 定义
        T_next_mat = torch.matmul(T_prev_mat, dT)
        T_next = SO3_R3.from_matrix(T_next_mat).log_map().reshape(N, 6)

        return T_next
    
    def integrate_samples(self, action_seq, T0=None):
        """
        将 twist 序列积分为 SE(3) 轨迹。

        参数：
        xi_seq: shape (N,T,6) 的 ndarray, 每行是一次 twist [vx,vy,vz, ωx,ωy,ωz]
        dt:     时间步长(scalar)
        T0:     初始位姿, 4x4 homogeneous matrix (默认为单位)
        返回：
        Ts:     list 长度 T 的 4x4 ndarray, 分别是 T0, T1, ..., T_T
        """
        N, bs, T = action_seq.shape[:-1]
        action_seq = action_seq.reshape(-1, 6)
        if T0 is None:
            T0 = torch.eye(4).reshape(1,1,4,4).to(action_seq.device)
            T0 = T0.repeat(N, bs, 1, 1)
        else:
            T0 = SO3_R3().exp_map(T0[...,:6].reshape(-1,6)).to_matrix().reshape(1,bs,4,4)
            T0 = T0.repeat(N, 1,  1, 1)

        Ts = [T0.clone()]
        dT = SO3_R3().exp_map(action_seq * self.dt).to_matrix().reshape(N, bs, T, 4, 4)
        for t in range(T):            
            # left‐multiply（body frame）或 right‐multiply（space frame）看你的 twist 定义
            T_next = torch.matmul(Ts[-1], dT[:,:,t])
            Ts.append(T_next)
        traj = torch.stack(Ts, dim=-3).reshape(N, bs, T + 1, 4, 4)[:,:,1:]
        # size = traj.shape[:-2]
        # traj = traj.reshape(-1, 4, 4)
        # traj = SO3_R3.from_matrix(traj).log_map().reshape(size + (T, 6))
        return traj
    

    def integrate_distribution(self, dist, T0=None):
        mus = dist.mus # (bs, bs, T, nc, 6)
        cov = dist.cov # (bs, bs, T, nc, 6)
        N, bs, T, nc = mus.shape[:-1]
        action_seq = mus.reshape(-1, 6)
        if T0 is None:
            T0 = torch.eye(4).reshape(1,1,1,4,4).to(action_seq.device)
            T0 = T0.repeat(N, bs, nc, 1, 1)
        else:
            T0 = SO3_R3().exp_map(T0[...,:6].reshape(-1,6)).to_matrix().reshape(1,1,1,4,4)
            T0 = T0.repeat(N, 1, nc, 1, 1) # bs or 1?
        
        Ts = [T0.clone()]
        P0 = torch.eye(6).to(action_seq.device).reshape(1,1,1,6,6)
        P0 = P0.repeat(N, bs, nc, 1, 1)
        Ps = [P0.clone()]


        dT = SO3_R3().exp_map(action_seq * self.dt).to_matrix().reshape(N, bs, T, nc, 4, 4)
        inv_dT = SO3_R3().exp_map(-action_seq * self.dt).to_matrix().reshape(N, bs, T, nc, 4, 4)
        for t in range(T):            
            # left‐multiply（body frame）或 right‐multiply（space frame）看你的 twist 定义
            T_next = torch.matmul(Ts[-1], dT[:,:,t])
            F_k = adjoint_SE3(inv_dT[:,:,t])
            P_next = F_k.matmul(Ps[-1].matmul(F_k.transpose(-2,-1))) + cov[:,:,t] * self.dt
            Ts.append(T_next)
            Ps.append(P_next)
        traj = torch.stack(Ts, dim=2).reshape(N, bs, T+1, nc, 4, 4)[:,:,1:]
        P_traj = torch.stack(Ps, dim=2).reshape(N, bs, T+1, nc, 6, 6)[:,:,1:]
        integrated_dist = GMMSE3(dist.log_pis, traj, P_traj)
        # samples = integrated_dist.rsample()
        # logprob = integrated_dist.log_prob(samples.repeat(10,1,1,1,1))

        return integrated_dist
    
    def set_initial_condition(self, init_con):
        self.initial_conditions = init_con

def adjoint_SE3(T: torch.Tensor) -> torch.Tensor:
    """
    Compute Adjoint matrix of SE(3) transform(s).
    
    Args:
        T: (..., 4, 4) batch of SE(3) transforms
    
    Returns:
        AdT: (..., 6, 6) batch of Adjoint matrices
    """
    assert T.shape[-2:] == (4, 4), "Input must be (..., 4, 4)"

    R = T[..., :3, :3]   # (..., 3, 3)
    p = T[..., :3, 3]    # (..., 3)

    # skew-symmetric matrix of p
    px = torch.zeros((*p.shape[:-1], 3, 3), dtype=T.dtype, device=T.device)
    px[..., 0, 1] = -p[..., 2]
    px[..., 0, 2] =  p[..., 1]
    px[..., 1, 0] =  p[..., 2]
    px[..., 1, 2] = -p[..., 0]
    px[..., 2, 0] = -p[..., 1]
    px[..., 2, 1] =  p[..., 0]

    # assemble Adjoint
    Ad = torch.zeros((*T.shape[:-2], 6, 6), dtype=T.dtype, device=T.device)
    Ad[..., :3, :3] = R
    Ad[..., 3:, 3:] = R
    Ad[..., 3:, :3] = px @ R
    return Ad

if __name__ == "__main__":
    # 随机 twist 序列
    T = 5
    N = 3
    xi_seq = torch.zeros((1, N, T, 6))
    T0 = torch.eye(4).unsqueeze(0)
    # 例如：沿 x 方向以 0.1m/s，绕 z 轴以 10°/s
    xi_seq[..., 0] = 0.1
    xi_seq[..., 5] = 0.2

    dt = 0.1  # 0.1s 步长
    dynamic = SE3Integrator(dt, None, torch.device('cpu'))
    poses = dynamic.integrate_samples(xi_seq, None)

    for i, T_i in enumerate(poses):
        print(f"Pose at step {i}:\n{T_i}\n")