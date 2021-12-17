from loguru import logger

import torch
from torch import nn
from torch.nn import functional as F
from torch import transpose as T
from einops import rearrange

from .features import GaussianMapping, Favor


# @torch.no_grad()
def build_uniform_prior(b, m, n, device, dtype):
    one = torch.tensor(1., device=device, dtype=dtype)
    ms, ns = m*one, n*one
    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)
    return log_mu, log_nu, norm


def build_normalized_prior(prior0, prior1):
    """
    prior0 (torch.Tensor): [B, M, 1]
    prior1 (torch.Tensor): [B, N, 1]
    """
    prior0, prior1 = map(lambda x: x[..., 0], [prior0, prior1])
    log_norm = - (prior0.sum(-1) + prior1.sum(-1)).log()[:, None]
    log_mu = torch.cat([prior0, prior1.sum(-1, keepdim=True)], -1).log() + log_norm
    log_nu = torch.cat([prior1, prior0.sum(-1, keepdim=True)], -1).log() + log_norm
    return log_mu, log_nu, log_norm


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability
    TODO: entropic regularization != 1
    """
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    Z = Z + u.unsqueeze(2) + v.unsqueeze(1)
    return Z


def log_paritial_optimal_transport(scores, alpha=None, prior0=None, prior1=None, iters=20):
    """
    Args:
        scores (torch.Tensor): [B, M, N] if alpha is not None else [B, M+1, N+1]
        alpha: learnable parameter for dustbin score
        prior0 (torch.Tensor): [B, M, 1] (optional)
        prior1 (torch.Tensor): [B, N, 1] (optional)
    """
    (b, m, n), device = scores.shape, scores.device
    m, n = (m-1, n-1) if alpha is None else (m, n)

    if alpha is not None:
        bins0 = alpha.expand(b, m, 1)
        bins1 = alpha.expand(b, 1, n)
        alpha = alpha.expand(b, 1, 1)
        scores = torch.cat([torch.cat([scores, bins0], -1),
                            torch.cat([bins1, alpha], -1)], 1)
    if prior0 is not None:
        log_mu, log_nu, log_norm = build_normalized_prior(prior0, prior1)
    else:  # uniform prior
        log_mu, log_nu, log_norm = build_uniform_prior(b, m, n, device, torch.float32)
    
    Z = log_sinkhorn_iterations(scores, log_mu, log_nu, iters)
    Z = Z - log_norm[..., None] if prior0 is not None else Z - log_norm
    return Z


def log_sinkhorn_iterations_linear_dustbin(feat0, feat1, alpha, log_mu, log_nu, iters, reg=1.0):
    """
    Args:
        feat0: (B, C, M)
        feat1: (B, C, N)
        alpha: (1,)
        log_mu: (B, M+1)
        log_nu: (B, N+1)
        reg: entropic regularization coefficient
    Returns:
        Z: (B, M+1, N+1) unnormalized assignment matrix in log-domain
    """
    b, m, n = feat0.shape[0], feat0.shape[-1], feat1.shape[-1]
    _phi_x, _phi_y = feat0, feat1  # feat already transformerd.
    _alpha = torch.exp(alpha)

    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = (log_mu - torch.log(torch.cat([
            (T(_phi_x, 1, 2) @ (_phi_y @ (v[:, :-1, None] / reg).exp())
             )[..., 0] + _alpha * (v[:, [-1]] / reg).exp(),  # (B, M)
            (_alpha * (v / reg).exp()).sum(-1, keepdims=True)
        ], -1) * (u / reg).exp() + 1e-6)) * reg + u

        v = (log_nu - torch.log(torch.cat([
            (((u[:, None, :-1] / reg).exp() @ T(_phi_x, 1, 2)) @ _phi_y)[:, 0] + _alpha * (u[:, [-1]] / reg).exp(),
            (_alpha * (u / reg).exp()).sum(-1, keepdims=True)
        ], -1) * (v / reg).exp() + 1e-6)) * reg + v

    # The final transformation is still O(MN)
    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    # The cost matrix is approximated, but the dustbin is exact.
    M = (torch.cat([torch.cat([(T(_phi_x, 1, 2) @ _phi_y).log(), bins0], -1),
                    torch.cat([bins1, alpha], -1)], 1) + 1e-6)
    Z = M + u.unsqueeze(2) + v.unsqueeze(1)
    return Z


def log_sinkhorn_iterations_linear(feat0, feat1, log_mu, log_nu, iters, reg=1.0):
    """
    Args:
        feat0: (B, C, M)
        feat1: (B, C, N)
        log_mu: (B, M+1)
        log_nu: (B, N+1)
        reg: entropic regularization coefficient
    Returns:
        Z: (B, M+1, N+1) unnormalized assignment matrix in log-domain
    """
    _phi_x, _phi_y = feat0, feat1

    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = (log_mu - torch.log(
                (T(_phi_x, 1, 2) @ (_phi_y @ (v[..., None] / reg).exp()))[..., 0]
                * (u / reg).exp() + 1e-6)) * reg + u

        v = (log_nu - torch.log(
                (((u[:, None] / reg).exp() @ T(_phi_x, 1, 2)) @ _phi_y)[:, 0]
                * (v / reg).exp() + 1e-6)) * reg + v

    M = (T(_phi_x, 1, 2) @ _phi_y).log() + 1e-6
    Z = M + u.unsqueeze(2) + v.unsqueeze(1)
    return Z


def log_partial_optimal_transport_linear(feat0, feat1, alpha=None, prior0=None, prior1=None, iters=20):
    """
    Args:
        feat0 (torch.Tensor): [B, M, C] if alpha is not None else [B, M+1, C]
        feat1 (torch.Tensor): [B, N, C] if alpha is not None else [B, N+1, C]
        alpha: learnable parameter for dustbin score
        prior0 (torch.Tensor): [B, M, 1] (optional)
        prior1 (torch.Tensor): [B, N, 1] (optional)
    NOTE: feat0, feat1 should already be transformed to use linear sinkhorn.
    """
    feat0, feat1 = map(lambda x: x.permute(0, 2, 1), [feat0, feat1])  # (B, C, M)
    b, m, n, device = feat0.shape[0], feat0.shape[-1], feat1.shape[-1], feat0.device
    m, n = (m-1, n-1) if alpha is None else (m, n)

    if prior0 is not None:
        log_mu, log_nu, log_norm = build_normalized_prior(prior0, prior1)
    else:
        log_mu, log_nu, log_norm = build_uniform_prior(b, m, n, device, torch.float32)
    
    if alpha is not None:
        Z = log_sinkhorn_iterations_linear_dustbin(feat0, feat1, alpha, log_mu, log_nu, iters, reg=1.0)
    else:
        Z = log_sinkhorn_iterations_linear(feat0, feat1, log_mu, log_nu, iters, reg=1.0)
    
    Z = Z - log_norm[..., None] if prior0 is not None else Z - log_norm
    return Z


class OptimalTransport(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.skh_iters = config['iters']
        self.partial_impl = config['partial_impl']
        self.bin_score = torch.nn.Parameter(torch.tensor(config['init_bin_score'], requires_grad=True)) \
            if self.partial_impl == 'dustbin' else None
        self.bin_prototype = torch.nn.Parameter(torch.randn(config['d_model'], requires_grad=True)) \
            if self.partial_impl == 'prototype' and config['prototype_impl'] == 'learned' else None
        self.with_prior = config['with_prior']
        self.skh_linear = config['linear']['enable']

        if self.skh_linear:
            self.mapping = self.build_mapping()
    
    def build_mapping(self):
        mapping_dim = self.config['linear']['mapping_dim']
        if self.config['linear']['mapping'] == 'favor':  # orthogonal positive random feature
            mapping = Favor(self.config['d_model'], n_dims=mapping_dim, softmax_temp=1., orthogonal=True)
            mapping.new_feature_map()  # TODO: periodically redraw the feature map to avoid degenerate samples
        elif self.config['linear']['mapping'] == 'gauss':  # mapping proposed for gaussian kernel
            mapping = GaussianMapping(self.config['d_model'], n_rand_feature=mapping_dim, learnable=False)
        else:
            raise NotImplementedError()
        return mapping

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """ 
        Args:
            feat_c0 (torch.Tensor): [N, L, C] (assumed to be properly normalized)
            feat_c1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        """
        N, L, S = feat_c0.shape[0], feat_c0.shape[1], feat_c1.shape[1]
        # build prototype
        if self.partial_impl == 'prototype':
            feat_c0, feat_c1 = map(self.cat_proto, [feat_c0, feat_c1])  # (N, L+1, C)
        # build marginal prior
        prior0, prior1 = self.build_prior(data, mask_c0, mask_c1) if self.with_prior else (None, None)
        
        # run log-domain sinkhorn / linear-sinkhorn iterations
        if self.skh_linear:
            # TODO: Masking features from padded regions.
            feat_c0, feat_c1 = map(self.mapping, [feat_c0, feat_c1])
            log_assign_matrix = log_partial_optimal_transport_linear(feat_c0, feat_c1, self.bin_score, prior0, prior1, self.skh_iters)
        else:
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1)
            if mask_c0 is not None:
                sim_matrix[:, :L, :S].masked_fill_(~(mask_c0[..., None] * mask_c1[:, None]).bool(), float('-inf'))
                if self.partial_impl == 'prototype':  # avoid nan when the whole col / row are all `-inf`
                    sim_matrix[:, :-1, S][~mask_c0.bool()] = 0  # padded regions have `-inf` similarity with all elements except for dustbin-prototype
                    sim_matrix[:, L, :-1][~mask_c1.bool()] = 0
            log_assign_matrix = log_paritial_optimal_transport(sim_matrix, self.bin_score, prior0, prior1, self.skh_iters)
        return log_assign_matrix

    def build_prior(self, data, mask_c0, mask_c1, eps=1e-9):
        prior0, prior1 = data['prior0'], data['prior1']
        if mask_c0 is not None:  # set padded region to a near zero prior
            # TODO: this is just a quick&dirty solution
            prior0 = prior0 * mask_c0[..., None]
            prior1 = prior1 * mask_c1[..., None]
            prior0[~mask_c0[..., None].bool()] += eps
            prior1[~mask_c1[..., None].bool()] += eps
        return prior0, prior1

    def cat_proto(self, feat):
        if self.config['prototype_impl'] == 'learned':
            proto = self.bin_prototype[None, None].expand(feat.shape[0], -1, -1)
        elif self.config['prototype_impl'] == 'mean':
            proto = feat.mean(1, keepdim=True)
        else:
            raise NotImplementedError()
        return torch.cat([feat, proto], 1)
