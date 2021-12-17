import copy
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention
from .layers import TransformerEncoderLayer, RZTXEncoderLayer, LoFTREncoderLayerConv1d


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dropout=0.1,
                 attention='linear', kernel_fn='elu + 1', redraw_interval=1, d_kernel=None,
                 rezero=None):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention(self.dim, kernel_fn=kernel_fn, redraw_interval=redraw_interval, d_kernel=d_kernel) \
                            if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
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
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.dropout1(message)  # dropout before merging multi-head queried outputs
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.dropout2(message)
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message if not self.rezero else x + self.res_weight * message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        if config['redraw_interval'] is not None:
            assert config['redraw_interval'] % 2 == 0, 'redraw_interval must be divisible by 2 since each attetnion layer is repeatedly called twice.'

        encoder_layer = build_encoder_layer(config)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])

        if config['final_proj']:
            self.final_proj = nn.Linear(config['d_model'], config['d_model'], bias=True)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError
        
        if self.config['final_proj']:
            feat0, feat1 = map(self.final_proj, [feat0, feat1])
        
        return feat0, feat1


def build_encoder_layer(config):
    if config['type'] == 'LoFTR':
        layer = LoFTREncoderLayer(config['d_model'],
                                  config['nhead'],
                                  config['dropout'],
                                  config['attention'], config['kernel_fn'], config['redraw_interval'], config['d_kernel'],
                                  rezero=config['rezero'])
    elif config['type'] == 'LoFTR-Conv1d':
        layer = LoFTREncoderLayerConv1d(config['d_model'],
                                        config['nhead'],
                                        config['dropout'],
                                        config['attention'], config['kernel_fn'], config['redraw_interval'], config['d_kernel'],
                                        rezero=config['rezero'])
    elif config['type'] in ['Pre-LN', 'Post-LN']:
        layer = TransformerEncoderLayer(config['d_model'],
                                        config['nhead'],
                                        config['d_ffn'],  # vanilla Transformer uses a much higher FFN dim
                                        config['dropout'],
                                        config['attention'], config['kernel_fn'], config['redraw_interval'], config['d_kernel'],
                                        'relu', config['type']=='Pre-LN')
    elif config['type'] == 'Rezero':
        layer = RZTXEncoderLayer(config['d_model'],
                                 config['nhead'],
                                 config['d_ffn'],
                                 config['dropout'],
                                 config['attention'], config['kernel_fn'], config['redraw_interval'], config['d_kernel'],
                                 'relu')
    else:
        raise ValueError()
    return layer
