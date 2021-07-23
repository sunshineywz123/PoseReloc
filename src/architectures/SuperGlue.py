import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy


def normalize_2d_keypoints(kpts, image_shape):
    """ Normalize 2d keypoints locations based on image image_shape
    kpts: [b, n, 2]
    image_shape: [b, 2]
    """
    height, width = image_shape[0]
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


def normalize_3d_keypoints(kpts):
    """ Normalize 3d keypoints locations based on the tight bbox
    kpts: [b, n, 3]
    """
    width, height, length = kpts[0].max(dim=0).values - kpts[0].min(dim=0).values
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height, one*length])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


def MLP(channels: list, do_bn=True):
    """Multi-layer perceptron"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True)
        )
        if i < n - 1:
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i])) 
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs """
    def __init__(self, inp_dim, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([inp_dim] + list(layers) + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** 0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivity"""
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)

        query, key, value = [
                l(x).view(batch_dim, self.dim, self.num_heads, -1)
                for l, x in zip(self.proj, (query, key, value))
        ]
        x, prob = attention(query, key, value)
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):

    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionPropagation(feature_dim, 4)
            for _ in range(len(layer_names))
        ])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters:int):
    """ Perform Sinkhorn Normalization in Log-space for stability """
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters:int):
    """ Perform Differentiable Optimal Transport in Log-space for stability """
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([
            torch.cat([scores, bins0], -1),
            torch.cat([bins1, alpha], -1)
        ], 1)

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1 # traceable in 1.1


class SuperGlue(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        #self.match_type = 'softmax'
        if 'match_type' in hparams:
            self.match_type = hparams['match_type']
        else:
            self.match_type = 'sinkhorn'

        self.kenc_2d = KeypointEncoder(
                        inp_dim=3,
                        feature_dim=hparams['descriptor_dim'],
                        layers=hparams['keypoints_encoder']
                    )

        self.kenc_3d = KeypointEncoder(
                        inp_dim=4,
                        feature_dim=hparams['descriptor_dim'],
                        layers=hparams['keypoints_encoder']
                    )

        GNN_layers = ['self', 'cross'] * 9 # FIXME: set in config
        self.gnn = AttentionalGNN(
                        feature_dim=hparams['descriptor_dim'],
                        layer_names=GNN_layers
                    )

        self.final_proj = nn.Conv1d(
                        in_channels=hparams['descriptor_dim'],
                        out_channels=hparams['descriptor_dim'],
                        kernel_size=1,
                        bias=True
                    )

        bin_score = nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    def forward(self, data):
        """
        keys of data:
            descriptors2d: [b, feature_dim, n1]
            descriptors3d: [b, feature_dim, n2]
            keypoints2d:   [b, n1, 2]
            keypoints3d:   [b, n2, 3]
            scores2d:      [b, n1, 1]
            scores3d:      [b, n2, 1]
            image:         [b, 1, h, w]
        """
        desc_2d, desc_3d = data['descriptors2d'].float(), data['descriptors3d'].float()
        kpts_2d, kpts_3d = data['keypoints2d'].float(), data['keypoints3d'].float()
        scores_2d, scores_3d = data['scores2d'][:, :, 0].float(), data['scores3d'][:, :, 0].float()

        if kpts_2d.shape[1] == 0 or kpts_3d.shape[1] == 0:
            shape0, shape1 = kpts_2d.shape[:-1], kpts_3d.shape[:-1]
            return {
                'matches0': kpts_2d.new_full(shape0, -1, dtype=torch.int)[0],
                'matches1': kpts_3d.new_full(shape1, -1, dtype=torch.int)[0],
                'matching_scores0': kpts_2d.new_zeros(shape0)[0],
                'matching_scores1': kpts_3d.new_zeros(shape1)[0],
                'skip_train': True
            }

        # keypoint normalization
        kpts_2d = normalize_2d_keypoints(kpts_2d, data['image_size'])
        kpts_3d = normalize_3d_keypoints(kpts_3d) # FIXME: normalize by box scale

        # keypoints MLP encoder
        desc_2d = desc_2d + self.kenc_2d(kpts_2d, scores_2d)
        # desc_3d = desc_3d + self.kenc_3d(kpts_3d, scores_3d) # without 3d positional encoding?

        # Multi-layer Transformer network
        desc_2d, desc_3d = self.gnn(desc_2d, desc_3d)

        # Final MLP projection
        mdesc_2d, mdesc_3d = self.final_proj(desc_2d), self.final_proj(desc_3d)

        # Normalize mdesc to avoid NaN
        mdesc_2d = F.normalize(mdesc_2d, p=2, dim=1)
        mdesc_3d = F.normalize(mdesc_3d, p=2, dim=1)
            
        # Compute matching descriptor distance
        if self.match_type == 'sinkhorn':
            scores = torch.einsum('bdn,bdm->bnm', mdesc_2d, mdesc_3d)
            scores = scores / self.hparams['descriptor_dim']

            # Run the optimal transport
            log_assign_matrix = log_optimal_transport(
                        scores, self.bin_score,
                        iters=self.hparams['sinkhorn_iterations']
                    )
            assign_matrix = log_assign_matrix.exp()
            conf_matrix = assign_matrix[:, :, :]

            # Get the matches with score above "match_threshold"
            max0, max1 = conf_matrix[:, :-1, :-1].max(2), conf_matrix[:, :-1, :-1].max(1)
            indices0, indices1 = max0.indices, max1.indices
            mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
            mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
            zero = conf_matrix.new_tensor(0)
            mscores0 = torch.where(mutual0, max0.values, zero)
            mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
            valid0 = mutual0 & (mscores0 > self.hparams['match_threshold'])
            valid1 = mutual1 & valid0.gather(1, indices1)
            indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
            indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

            preds = {
                'matches0': indices0, # use -1 for invalid match
                'matches1': indices1, # use -1 for invliad match
                'matching_scores0': mscores0,
                'matching_scores1': mscores1,
            }

        elif self.match_type == 'softmax':
            dim = torch.Tensor([mdesc_2d.shape[1]]).cuda()
            scores = torch.einsum('bdn, bdm->bnm', mdesc_2d, mdesc_3d) / 0.1

            conf_matrix = F.softmax(scores, 1) * F.softmax(scores, 2)
            # Get the matches with score above "match_threshold"
            max0, max1 = conf_matrix[:, :, :].max(2), conf_matrix[:, :, :].max(1)
            indices0, indices1 = max0.indices, max1.indices
            mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
            mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
            zero = conf_matrix.new_tensor(0)
            mscores0 = torch.where(mutual0, max0.values, zero)
            mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
            valid0 = mutual0 & (mscores0 > self.hparams['match_threshold'])
            valid0 = mscores0 > self.hparams['match_threshold']
            valid1 = mutual1 & valid0.gather(1, indices1)
            indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
            indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

            preds = {
                'matches0': indices0[0], # use -1 for invalid match
                'matches1': indices1[0], # use -1 for invliad match
                'matching_scores0': mscores0[0],
                'matching_scores1': mscores1[0],
            }

        return preds, conf_matrix
