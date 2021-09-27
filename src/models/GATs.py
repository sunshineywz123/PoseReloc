import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h_2d, h_3d, adj):
        b, n1, dim = h_3d.shape
        b, n2, dim = h_2d.shape
        num_leaf = int(n2 / n1)

        wh_2d = torch.matmul(h_2d, self.W)
        wh_3d = torch.matmul(h_3d, self.W)
        e = self._prepare_attentional_mechanism_input(wh_2d, wh_3d, num_leaf)

        # zero_vec = -9e15 * torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(e, dim=2)

        h_2d = torch.reshape(h_2d, (b, n1, num_leaf, dim))
        h_prime = torch.einsum('bncd,bncq->bnq', attention, h_2d)
        # wh_2d = torch.reshape(wh_2d, (b, n1, num_leaf, dim))
        # h_prime = torch.einsum('bncd,bncq->bnq', attention, wh_2d)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _prepare_attentional_mechanism_input(self, wh_2d, wh_3d, num_leaf):
        b, n1, dim = wh_3d.shape
        b, n2, dim = wh_2d.shape

        wh_2d_ = torch.matmul(wh_2d, self.a[:self.out_features, :]) # [b, N2, 1]
        wh_2d_ = torch.reshape(wh_2d_, (b, n1, num_leaf, -1)) # [b, n1, 6, 1]
        wh_3d_ = torch.matmul(wh_3d, self.a[self.out_features:, :]) # [b, N1, 1]

        # e = torch.einsum('bnd,bncd->bncd', wh_3d_, wh_2d_)
        e = wh_3d_.unsqueeze(2) + wh_2d_
        return self.leakyrelu(e)


class GraphAttentionLayer_v2(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer_v2, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, desc_2d, adj):
        wh_2d = torch.matmul(desc_2d, self.W)
        e = self._prepare_attentional_mechanism_input(wh_2d) # [b, n_2d, 1]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attetion, wh_2d)

        if self.concat:
            return F.elu(h_prime)        
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, wh):
        wh1 = torch.matmul(wh, self.a[:self.out_features, :])
        wh2 = torch.matmul(wh, self.a[self.out_features:, :])

        import ipdb; ipdb.set_trace()
        e = wh1 + wh2.permute(0, 2, 1)
        return self.leakyrelu(e)


class GraphAttentionLayer_orig(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer_orig, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape (N, in_features), wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)
