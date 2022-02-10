from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from .linear_attention import LinearAttention, FullAttention
from ..utils.position_encoding import KeypointEncoding, KeypointEncoding_linear, PositionEncodingSine

class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 n_head,
                 d_ff=2048,
                 dropout=0.1,
                 attention='linear', kernel_fn='elu + 1', redraw_interval=1, d_kernel=None,
                 activation='relu', normalize_before=False):
        super().__init__()
        
        self.dim = d_model // n_head
        self.n_head = n_head
        self.normalize_before = normalize_before
        
        # multi-head attention  # TODO: bias=True
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention(self.dim, kernel_fn=kernel_fn, redraw_interval=redraw_interval, d_kernel=d_kernel) \
                            if attention == 'linear' else FullAttention()
        
        # Feed-forward Network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = _get_activation_fn(activation)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _forward_attn(self, x, source, x_mask=None, source_mask=None):
        bs = x.size(0)
        query, key, value = x, source, source
        
        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.n_head, self.dim)  # [N, L, H, D]
        key = self.k_proj(key).view(bs, -1, self.n_head, self.dim)  # [N, S, H, D]
        value = self.v_proj(value).view(bs, -1, self.n_head, self.dim)  # [N, L, H, D]
        
        x2 = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, H, D]
        return x2.flatten(-2)  # FIXME: Lack of the final perjection / merge operation
        
    def _forward_pre(self, x, source, x_mask=None, source_mask=None):
        # norm - attn - add
        x2, source2 = map(self.norm1, [x, source])
        x2 = self._forward_attn(x2, source2, x_mask, source_mask)
        x = x + self.dropout1(x2)
        
        # norm - ffn - add
        x2 = self.norm2(x2)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.dropout2(x2)
        return x
    
    def _forward_post(self, x, source, x_mask=None, source_mask=None):
        # attn - add - norm
        x2 = self._forward_attn(x, source, x_mask, source_mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        
        # ffn - add - norm
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x
    
    def forward(self, x, source, x_mask=None, source_mask=None):
        if self.normalize_before:
            return self._forward_pre(x, source, x_mask=x_mask, source_mask=source_mask)
        return self._forward_post(x, source, x_mask=x_mask, source_mask=source_mask)
        
        
class RZTXEncoderLayer(nn.Module):
    """Rezero Transformer Encoder Layer"""
    def __init__(self,
                 d_model,
                 n_head,
                 d_ff=2048,
                 dropout=0.1,
                 attention='linear', kernel_fn='elu + 1', redraw_interval=1, d_kernel=None,
                 activation='relu'):
        super().__init__()
        
        self.dim = d_model // n_head
        self.n_head = n_head
        
        # multi-head attention  # TODO: bias=True
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention(self.dim, kernel_fn=kernel_fn, redraw_interval=redraw_interval, d_kernel=d_kernel) \
                            if attention == 'linear' else FullAttention()
                            
        # Feed-forward Network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        
        self.res_weight = nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def _forward_attn(self, x, source, x_mask=None, source_mask=None):
        bs = x.size(0)
        query, key, value = x, source, source
        
        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.n_head, self.dim)  # [N, L, H, D]
        key = self.k_proj(key).view(bs, -1, self.n_head, self.dim)  # [N, S, H, D]
        value = self.v_proj(value).view(bs, -1, self.n_head, self.dim)  # [N, L, H, D]
        
        x2 = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, H, D]
        return x2.flatten(-2)
    
    def forward(self, x, source, x_mask=None, source_mask=None):
        # attn - rezero - add
        x2 = self._forward_attn(x, source, x_mask, source_mask)
        x2 = x2 * self.res_weight
        x = x + self.dropout1(x2)
        
        # ffn - rezero -add
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x2 = x2 * self.res_weight
        x = x + self.dropout2(x2)
        
        return x
        

class LoFTREncoderLayerConv1d(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dropout=0.1,
                 attention='linear', kernel_fn='elu + 1', redraw_interval=1, d_kernel=None,
                 rezero=None):
        """LoFTREncoderLayer using `nn.conv1d` instead of `nn.Linear`"""
        super(LoFTREncoderLayerConv1d, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.k_proj = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.v_proj = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.attention = LinearAttention(self.dim, kernel_fn=kernel_fn, redraw_interval=redraw_interval, d_kernel=d_kernel) \
                            if attention == 'linear' else FullAttention()
        self.merge = nn.Conv1d(d_model, d_model, 1, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Conv1d(d_model*2, d_model*2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(d_model*2, d_model, 1, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # rezero  TODO: Remove LayerNorm while using rezero
        if rezero is not None:
            self.res_weight = nn.Parameter(torch.Tensor([rezero]), requires_grad=True)
        self.rezero = True if rezero is not None else False

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        x = x.permute(0, 2, 1)
        source = source.permute(0, 2, 1)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).permute(0, 2, 1).view(bs, -1, self.nhead, self.dim)  # [N, L, H, D]
        key = self.k_proj(key).permute(0, 2, 1).view(bs, -1, self.nhead, self.dim)  # [N, S, H, D]
        value = self.v_proj(value).permute(0, 2, 1).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, H, D]
        message = self.dropout1(message)
        message = self.merge(message.view(bs, -1, self.nhead*self.dim).permute(0, 2, 1)).permute(0, 2, 1)  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.dropout2(message)
        message = self.mlp(torch.cat([x.permute(0, 2, 1), message], dim=2).permute(0,2,1))
        message = self.norm2(message.permute(0,2,1))

        return (x.permute(0, 2, 1) + message) if not self.rezero else (x + self.res_weight * message)
    
        
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class PositionalEncodingLayer(nn.Module):
    def __init__(self, d_model_2D, max_shape_2D, inp_dim_3D, feature_dim_3D, layers_3D, norm_method_3D='instancenorm', encoding_type_3D='mlp_cov'):
        super().__init__()
        if encoding_type_3D == "mlp_cov":
            self.keypoint3D_encoder = KeypointEncoding(inp_dim_3D, feature_dim_3D, layers_3D, norm_method_3D)
        elif encoding_type_3D == "mlp_linear":
            self.keypoint3D_encoder = KeypointEncoding_linear(inp_dim_3D, feature_dim_3D, layers_3D, norm_method_3D)
        else:
            raise NotImplementedError
        
        self.sine_encoder = PositionEncodingSine(d_model_2D, max_shape_2D)
    
    def forward(self, keypoints3D, feature3D, feature2D, data):
        """
        Parameters:
        --------------
        keypoints3D: B*L*3
        feature3D: B*L*D
        feature2D: B*N*D
        """
        # 3D position encoding
        feature3D = self.keypoint3D_encoder(keypoints3D, feature3D.transpose(1,2)).transpose(1,2) # B*L*D

        # 2D position encoding
        h, w = data["q_hw_c"]
        feature_map = rearrange(feature2D, "n (h w) c -> n c h w", h=h, w=w)
        feature_map = self.sine_encoder(feature_map)
        feature2D = rearrange(feature_map, "n c h w -> n (h w) c")

        return feature3D, feature2D
