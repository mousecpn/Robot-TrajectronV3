import torch
import torch.distributions as td
import numpy as np
from model.model_utils import to_one_hot
from model.transform import SO3_R3
from model.gmm_rot import GMM3D

class GMM6D(td.Distribution):
    def __init__(self, log_pis, mus, log_sigmas=None, covmat=None, subgmm=False):
        super(GMM6D, self).__init__(batch_shape=log_pis.shape[0], event_shape=log_pis.shape[1:], validate_args=False)
        self.components = log_pis.shape[-1]
        self.dimensions = 6
        self.device = log_pis.device

        log_pis = torch.clamp(log_pis, min=-1e5)
        self.log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)  # [..., N]
        self.mus = self.reshape_to_components(mus)         # [..., N, 6]
        self.pis_cat_dist = td.Categorical(logits=log_pis)

        if covmat is None:
            self.log_sigmas = self.reshape_to_components(log_sigmas)  # [..., N, 6]
            self.sigmas = torch.exp(self.log_sigmas)                       # [..., N, 6]
            self.L = torch.diag_embed(self.sigmas)
            size = self.L.shape
            self.cov = torch.bmm(self.L.reshape(-1,self.dimensions,self.dimensions), self.L.reshape(-1,self.dimensions,self.dimensions).permute(0,2,1))
            self.cov = self.cov.reshape(size)
        else:
            self.log_sigmas = None
            self.sigmas = None
            self.cov = covmat
            size = self.cov.shape
            self.L = torch.linalg.cholesky(self.cov.reshape(-1,self.dimensions,self.dimensions))
            self.L = self.L.reshape(size)
        self.subgmm = subgmm
        if self.subgmm:
            self.gmm_trans = self.get_trans_gmm()
            self.gmm_rot = self.get_rot_gmm()
    

    # @classmethod
    # def from_log_pis_mus_cov_mats(cls, log_pis, mus, cov_mats):
    #     sigmas = []
    #     for i in range(cov_mats.shape[-1]):
    #         sigmas.append(torch.sqrt(torch.clamp(cov_mats[..., i, i], min=1e-8)))
    #     sigmas = torch.stack(sigmas, dim=-1)
    #     log_sigmas = torch.log(sigmas)
    #     # if kf ==True:
    #     #     return cls(log_pis, mus, log_sigmas, cov_mats.unsqueeze(-1))
    #     return cls(log_pis, mus, log_sigmas, None)
    
    @classmethod
    def from_log_pis_mus_cov_mats(cls, log_pis, mus, cov_mats, subgmm=False):
        # if kf ==True:
        #     return cls(log_pis, mus, log_sigmas, cov_mats.unsqueeze(-1))
        return cls(log_pis, mus, None, cov_mats, subgmm=subgmm)

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.

        :param sample_shape: Shape of the samples
        :return: Samples from the GMM.
        """
        mvn_samples = (self.mus +
                       torch.squeeze(
                           torch.matmul(self.L,
                                        torch.unsqueeze(
                                            torch.randn(size=sample_shape + self.mus.shape, device=self.device),
                                            dim=-1)
                                        ),
                           dim=-1))
        component_cat_samples = self.pis_cat_dist.sample(sample_shape)
        selector = torch.unsqueeze(to_one_hot(component_cat_samples, self.components), dim=-1)
        if torch.sum(torch.isnan(torch.sum(mvn_samples*selector, dim=-2)))>0:
            print('error')

        return torch.sum(mvn_samples*selector, dim=-2)

    def log_prob(self, value):
        r"""
        Calculates the log probability of a value using the PDF for bivariate normal distributions:

        .. math::
            f(x | \mu, \sigma, \rho)={\frac {1}{2\pi \sigma _{x}\sigma _{y}{\sqrt {1-\rho ^{2}}}}}\exp
            \left(-{\frac {1}{2(1-\rho ^{2})}}\left[{\frac {(x-\mu _{x})^{2}}{\sigma _{x}^{2}}}+
            {\frac {(y-\mu _{y})^{2}}{\sigma _{y}^{2}}}-{\frac {2\rho (x-\mu _{x})(y-\mu _{y})}
            {\sigma _{x}\sigma _{y}}}\right]\right)

        :param value: The log probability density function is evaluated at those values.
        :return: Log probability
        """
        # x: [..., 2]
        value = torch.unsqueeze(value, dim=-2)       # [..., 1, 3]
        dx = value - self.mus                       # [..., N, 3]

        # exp_nominator = ((torch.sum((dx/self.sigmas)**2, dim=-1)  # first and second term of exp nominator
        #                   - 2*self.corrs*torch.prod(dx, dim=-1)/torch.prod(self.sigmas, dim=-1)))    # [..., N]

        # component_log_p = -(2*np.log(2*np.pi)
        #                     + torch.log(self.one_minus_rho2)
        #                     + 2*torch.sum(self.log_sigmas, dim=-1)
        #                     + exp_nominator/self.one_minus_rho2) / 2
        
        size = dx.shape
        cov_inv = torch.inverse(self.cov)
        cov_inv = cov_inv.expand(dx.shape+(self.dimensions,))
        # bs = size[-3]
        # cov_inv = cov_inv.repeat(bs,1,1)
        # try:
        exp_nominator = dx.unsqueeze(-2) @ cov_inv
            # exp_nominator = torch.bmm(dx.reshape(-1,1,self.dimensions), cov_inv)
        # except:
        #     bs = size[-3]
        #     cov_inv = cov_inv.repeat(bs,1,1)
        #     exp_nominator = torch.bmm(dx.reshape(-1,1,self.dimensions), cov_inv)
        
        exp_nominator = exp_nominator @ dx.unsqueeze(-1)
        exp_nominator = -exp_nominator.reshape(size[0:-1])/2


        component_log_p = - (self.dimensions/2)*np.log(2*np.pi) - (1/2)*torch.det(self.cov) + exp_nominator
        return torch.logsumexp(self.log_pis + component_log_p, dim=-1)
    
    def CVaR_log_prob(self, value, alpha=0.1):
        value = torch.unsqueeze(value, dim=-2)       # [..., 1, 3]
        dx = value - self.mus                       # [..., N, 3]
        size = dx.shape
        cov_inv = torch.inverse(self.cov)
        cov_inv = cov_inv.expand(dx.shape+(self.dimensions,))
        exp_nominator = dx.unsqueeze(-2) @ cov_inv
        exp_nominator = exp_nominator @ dx.unsqueeze(-1)
        exp_nominator = -exp_nominator.reshape(size[0:-1])/2


        component_log_p = - (self.dimensions/2)*np.log(2*np.pi) - (1/2)*torch.det(self.cov) + exp_nominator
        
        ### CVaR calculation ###
        p = self.log_pis[0, :, 0].exp() # (bs, components)
        q = torch.clamp(p/alpha,max=1.0)
        remain = torch.ones_like(q[:,0])
        for i in range(p.shape[-1]):
            q[:, i] = torch.where(q[:, i]>remain, remain, q[:, i])
            remain = torch.where(q[:, i]<=remain, remain - q[:, i], torch.zeros_like(remain))
        ######### CVaR calculation #########
        if alpha < 0.7:
            return torch.sum(q[None, :, None, :] * component_log_p.exp(), dim=-1).log()
        else:
            max_idx = torch.argmax(p, dim=-1)
            expanded_idx = max_idx.unsqueeze(0).unsqueeze(2).expand(size[0:-2]).unsqueeze(-1)
            loss_term1 = (q[None, :, None, :] * component_log_p.exp())
            loss_term2 = torch.gather(loss_term1.clone(), -1, expanded_idx).squeeze(-1)
            loss_term1.scatter_(
                dim=-1,
                index=expanded_idx,
                src=torch.zeros_like(loss_term2).unsqueeze(-1)
            )
            return (torch.sum(loss_term1, dim=-1).detach() + loss_term2).log()


    def get_for_node_at_time(self, n, t):
        if self.log_sigmas is None:
            return self.__class__(self.log_pis[:, n:n+1, t:t+1], self.mus[:, n:n+1, t:t+1],
                                  None, self.cov[:, n:n+1, t:t+1])
        else:
            return self.__class__(self.log_pis[:, n:n+1, t:t+1], self.mus[:, n:n+1, t:t+1],
                                self.log_sigmas[:, n:n+1, t:t+1], None)
    
    def get_at_time(self, t):
        if self.log_sigmas is None:
            return self.__class__(self.log_pis[:, :, t:t+1], self.mus[:, :, t:t+1],
                                  None, self.cov[:, :, t:t+1])
        return self.__class__(self.log_pis[:, :, t:t+1], self.mus[:, :, t:t+1],
                              self.log_sigmas[:, :, t:t+1], None)
    
    def get_trans_gmm(self):
        return GMM3D(log_pis=self.log_pis, mus=self.mus[...,:3], cov_mats=self.cov[...,:3,:3])

    def get_rot_gmm(self):
        return GMM3D(log_pis=self.log_pis, mus=self.mus[...,3:], cov_mats=self.cov[...,3:,3:])

    def mode(self):
        """
        Calculates the mode of the GMM by calculating probabilities of a 2D mesh grid

        :param required_accuracy: Accuracy of the meshgrid
        :return: Mode of the GMM
        """
        if self.mus.shape[-2] > 1:
            samp, bs, time, comp, _ = self.mus.shape
            assert samp == 1, "For taking the mode only one sample makes sense."
            mode_node_list = []
            for n in range(bs):
                mode_t_list = []
                for t in range(time):
                    nt_gmm = self.get_for_node_at_time(n, t)
                    x_min = self.mus[:, n, t, :, 0].min()
                    x_max = self.mus[:, n, t, :, 0].max()
                    y_min = self.mus[:, n, t, :, 1].min()
                    y_max = self.mus[:, n, t, :, 1].max()
                    z_min = self.mus[:, n, t, :, 2].min()
                    z_max = self.mus[:, n, t, :, 2].max()
                    wx_min = self.mus[:, n, t, :, 3].min()
                    wx_max = self.mus[:, n, t, :, 3].max()
                    wy_min = self.mus[:, n, t, :, 4].min()
                    wy_max = self.mus[:, n, t, :, 4].max()
                    wz_min = self.mus[:, n, t, :, 5].min()
                    wz_max = self.mus[:, n, t, :, 5].max()
                    search_grid = torch.stack(torch.meshgrid([torch.arange(x_min, x_max+0.01, 0.01),
                                                              torch.arange(y_min, y_max+0.01, 0.01),
                                                              torch.arange(z_min, z_max+0.01, 0.01),
                                                              torch.arange(wx_min, wx_max+0.01, 0.1),
                                                              torch.arange(wy_min, wy_max+0.01, 0.1),
                                                              torch.arange(wz_min, wz_max+0.01, 0.1),
                                                              ],indexing='ij'), dim=6
                                              ).view(-1, 6).float().to(self.device)

                    ll_score = nt_gmm.log_prob(search_grid)
                    argmax = torch.argmax(ll_score.squeeze(), dim=0)
                    mode_t_list.append(search_grid[argmax])
                mode_node_list.append(torch.stack(mode_t_list, dim=0))
            return torch.stack(mode_node_list, dim=0).unsqueeze(dim=0)
        return torch.squeeze(self.mus, dim=-2)
    

    def reshape_to_components(self, tensor):
        if len(tensor.shape) == 5:
            return tensor
        return torch.reshape(tensor, list(tensor.shape[:-1]) + [self.components, self.dimensions])

    def get_covariance_matrix(self):
        # cov = self.corrs * torch.prod(self.sigmas, dim=-1)
        # E = torch.stack([torch.stack([self.sigmas[..., 0]**2, cov], dim=-1),
        #                  torch.stack([cov, self.sigmas[..., 1]**2], dim=-1)],
        #                 dim=-2)
        return self.cov


class GMMSE3(td.Distribution):
    def __init__(self, log_pis, T, P):
        super(GMMSE3, self).__init__(batch_shape=log_pis.shape[0], event_shape=log_pis.shape[1:], validate_args=False)
        self.T = T # [..., N, 4, 4]
        self.P = P # [..., N, 6, 6]
        jitter = 1e-12
        self.L = torch.linalg.cholesky(P+ jitter*torch.eye(6).to(P.device)) # [..., N, 6, 6]
        self.Pinv = torch.inverse(self.P)
        self.logdetP = torch.logdet(self.P)
        self.components = log_pis.shape[-1]
        self.dimensions = 6
        self.device = log_pis.device

        log_pis = torch.clamp(log_pis, min=-1e5)
        self.log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)  # [..., N]
        self.pis_cat_dist = td.Categorical(logits=log_pis)
        return
    
    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.

        :param sample_shape: Shape of the samples
        :return: Samples from the GMM.
        """
        mvn_samples = torch.squeeze(
                           torch.matmul(self.L,
                                        torch.unsqueeze(
                                            torch.randn(size=sample_shape + self.P.shape[:-1], device=self.device),
                                            dim=-1)
                                        ),
                           dim=-1)
        shape = mvn_samples.shape
        mvn_samples = self.T @ SO3_R3().exp_map(mvn_samples.reshape(-1, 6)).to_matrix().reshape(shape[:-1] + (4,4))
        component_cat_samples = self.pis_cat_dist.sample(sample_shape)
        selector = to_one_hot(component_cat_samples, self.components).unsqueeze(-1).unsqueeze(-1)
        if torch.sum(torch.isnan(torch.sum(mvn_samples*selector, dim=-3)))>0:
            print('error')

        return torch.sum(mvn_samples*selector, dim=-3)
    
    def log_prob(self, T):
        """
        Right perturbation log-prob on SE(3).
        Args:
            T: (...,4,4) poses
            T_bar: (4,4) mean pose
            P: (6,6) covariance
        Returns:
            log_probs: (...,)
        """
        const = -0.5 * (6 * torch.log(torch.tensor(2*torch.pi, dtype=T.dtype, device=T.device)) + self.logdetP)

        T = T.unsqueeze(-3)
        
        T_bar_inv = torch.linalg.inv(self.T)
        dT = T_bar_inv @ T
        size = dT.shape[:-2]
        xi = SO3_R3().from_matrix(dT.reshape(-1,4,4)).log_map().reshape(size + (6,))
        quad = (xi.unsqueeze(-2) @ (self.Pinv @ xi.unsqueeze(-1))).reshape(size)
        component_log_p = const - 0.5 * quad  # (...,N)
        
        log_prob = torch.logsumexp(self.log_pis + component_log_p, dim=-1)
        return log_prob
    
    def get_for_node_at_time(self, n, t):
        return
    
    def get_at_time(self, t):
        return
    
    def mode(self):
        return
