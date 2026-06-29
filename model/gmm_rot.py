import torch
import torch.distributions as td
import numpy as np
from model.model_utils import to_one_hot
from theseus import SO3
from scipy.stats import chi2

class GMM3D(td.Distribution):
    r"""
    Gaussian Mixture Model using 2D/3D Multivariate Gaussians each of as N components:
    Cholesky decompesition and affine transformation for sampling:

    .. math:: Z \sim N(0, I)

    .. math:: S = \mu + LZ

    .. math:: S \sim N(\mu, \Sigma) \rightarrow N(\mu, LL^T)

    where :math:`L = chol(\Sigma)` and

    .. math:: \Sigma = \left[ {\begin{array}{cc} \sigma^2_x & \rho \sigma_x \sigma_y \\ \rho \sigma_x \sigma_y & \sigma^2_y \\ \end{array} } \right]

    such that

    .. math:: L = chol(\Sigma) = \left[ {\begin{array}{cc} \sigma_x & 0 \\ \rho \sigma_y & \sigma_y \sqrt{1-\rho^2} \\ \end{array} } \right]

    :param log_pis: Log Mixing Proportions :math:`log(\pi)`. [..., N]
    :param mus: Mixture Components mean :math:`\mu`. [..., N * dim]
    :param log_sigmas: Log Standard Deviations :math:`log(\sigma_d)`. [..., N * dim]
    :param corrs: Cholesky factor of correlation :math:`\rho`. [..., N, dim+1]
    :param clip_lo: Clips the lower end of the standard deviation.
    :param clip_hi: Clips the upper end of the standard deviation.
    """
    def __init__(self, log_pis, mus, log_sigmas=None, corrs=None, dim=3, cov_mats=None):
        super(GMM3D, self).__init__(batch_shape=log_pis.shape[0], event_shape=log_pis.shape[1:], validate_args=False)
        self.components = log_pis.shape[-1]
        self.dim = self.dimensions = dim
        self.device = log_pis.device

        log_pis = torch.clamp(log_pis, min=-1e5)
        self.log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)  # [..., N]
        self.mus = self.reshape_to_components(mus)         # [..., N, 2]

        #### rotation representation ######
        if cov_mats is None:
            assert corrs is not None
            self.log_sigmas = self.reshape_to_components(log_sigmas)  # [..., N, 2]
            self.sigmas = torch.exp(self.log_sigmas)                       # [..., N, 2]
            # self.sigmas = torch.clamp(self.sigmas, max=10)
            self.corrs = self.reshape_to_components(corrs, self.dim+1)  # [..., N, 3]
            self.corrs = torch.nn.functional.normalize(self.corrs, dim=-1)
            corr_shape = self.corrs.shape[:-1] # (..., 4)
            self.corr_matrix = SO3(self.corrs.reshape(-1, self.dim+1)).to_matrix() # (N, 3, 3)
            sqrt_cov = torch.bmm(self.corr_matrix, torch.diag_embed(self.sigmas.reshape(-1, 3))) # (N, 3, 1)
            self.cov = torch.bmm(sqrt_cov, sqrt_cov.permute(0,2,1))
            # try:
            self.L = torch.linalg.cholesky(self.cov+torch.eye(3).to(self.cov.device)*1e-5)
            # except:
            #     print()

            self.cov = self.cov.reshape(list(corr_shape)+[self.dim,self.dim,])
            self.L = self.L.reshape(list(corr_shape)+[self.dim,self.dim,])
        else:
            self.cov = cov_mats
            self.L = torch.linalg.cholesky(self.cov)
            self.sigmas = None
            self.log_sigmas = None

        self.pis_cat_dist = td.Categorical(logits=log_pis)
    

    @classmethod
    def from_log_pis_mus_cov_mats(cls, log_pis, mus, cov_mats):
        return cls(log_pis, mus, dim=mus.shape[-1], cov_mats=cov_mats)

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

        size = dx.shape
        cov_inv = torch.inverse(self.cov.reshape(-1,self.dim, self.dim))

        try:
            exp_nominator = torch.bmm(dx.reshape(-1,1,self.dim), cov_inv)
        except:
            bs = value.shape[0]
            cov_inv = cov_inv.repeat(bs,1,1)
            exp_nominator = torch.bmm(dx.reshape(-1,1,self.dim), cov_inv)
        exp_nominator = torch.bmm(exp_nominator, dx.reshape(-1,1,self.dim).permute(0,2,1))
        exp_nominator = -exp_nominator.reshape(size[0:-1])/2


        component_log_p = - (self.dim/2)*np.log(2*np.pi) - (1/2)*torch.det(self.cov) + exp_nominator
        return torch.logsumexp(self.log_pis + component_log_p, dim=-1)
    
    def component_log_p(self, value):
        value = torch.unsqueeze(value, dim=-2)       # [..., 1, 3]
        dx = value - self.mus                       # [..., N, 3]


        
        size = dx.shape
        cov_inv = torch.inverse(self.cov.reshape(-1,self.dim, self.dim))

        try:
            exp_nominator = torch.bmm(dx.reshape(-1,1,self.dim), cov_inv)
        except:
            bs = value.shape[0]
            cov_inv = cov_inv.repeat(bs,1,1)
            exp_nominator = torch.bmm(dx.reshape(-1,1,self.dim), cov_inv)
        exp_nominator = torch.bmm(exp_nominator, dx.reshape(-1,1,self.dim).permute(0,2,1))
        exp_nominator = -exp_nominator.reshape(size[0:-1])/2
        component_log_p = - (self.dim/2)*np.log(2*np.pi) - (1/2)*torch.det(self.cov) + exp_nominator
        return component_log_p
    
    def statistical_test(self, value):
        value = torch.unsqueeze(value, dim=-2)       # [..., 1, 3]
        dx = value - self.mus                       # [..., N, 3]
        cov_inv = torch.inverse(self.cov.reshape(-1,self.dim, self.dim))
        try:
            exp_nominator = torch.bmm(dx.reshape(-1,1,self.dim), cov_inv)
        except:
            bs = value.shape[0]
            cov_inv = cov_inv.repeat(bs,1,1)
            exp_nominator = torch.bmm(dx.reshape(-1,1,self.dim), cov_inv)
        exp_nominator = torch.bmm(exp_nominator, dx.reshape(-1,1,self.dim).permute(0,2,1)).reshape(-1)
        return exp_nominator.min().cpu().detach().item()
        # if exp_nominator.min() < chi2.ppf(0.95, df=3): # chi2.ppf(0.95, df=3) ≈ 7.815
        #     # print("exp_nominator:",exp_nominator)
        #     return True
        # return False




    def get_for_node_at_time(self, n, t):
        if self.log_sigmas is None:
            return self.__class__(self.log_pis[:, n:n+1, t:t+1], self.mus[:, n:n+1, t:t+1],
                                  None, cov_mats=self.cov[:, n:n+1, t:t+1])
        else:
            return self.__class__(self.log_pis[:, n:n+1, t:t+1], self.mus[:, n:n+1, t:t+1],
                                self.log_sigmas[:, n:n+1, t:t+1], None)
    
    def get_at_time(self, t):
        if self.log_sigmas is None:
            return self.__class__(self.log_pis[:, :, t:t+1], self.mus[:, :, t:t+1],
                                  None, cov_mats=self.cov[:, :, t:t+1])
        return self.__class__(self.log_pis[:, :, t:t+1], self.mus[:, :, t:t+1],
                              self.log_sigmas[:, :, t:t+1], None)

    def mode(self):
        """
        Calculates the mode of the GMM by calculating probabilities of a 2D mesh grid

        :param required_accuracy: Accuracy of the meshgrid
        :return: Mode of the GMM
        """
        if self.mus.shape[-2] > 1:
            samp, bs, time, comp, dim = self.mus.shape
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
                    search_grid = torch.stack(torch.meshgrid([torch.arange(x_min, x_max, 0.01),
                                                            torch.arange(y_min, y_max, 0.01),
                                                            torch.arange(z_min, z_max, 0.01)
                                                            ],indexing='ij'), dim=3
                                            ).view(-1, dim).float().to(self.device)

                    ll_score = nt_gmm.log_prob(search_grid)
                    argmax = torch.argmax(ll_score.squeeze(), dim=0)
                    mode_t_list.append(search_grid[argmax])
                mode_node_list.append(torch.stack(mode_t_list, dim=0))
            return torch.stack(mode_node_list, dim=0).unsqueeze(dim=0)
        return torch.squeeze(self.mus, dim=-2)

    def reshape_to_components(self, tensor, dimensions=None):
        if len(tensor.shape) == 5:
            return tensor
        if dimensions is None:
            dimensions = self.dim
        return torch.reshape(tensor, list(tensor.shape[:-1]) + [self.components, dimensions])

    def get_covariance_matrix(self):
        return self.cov
