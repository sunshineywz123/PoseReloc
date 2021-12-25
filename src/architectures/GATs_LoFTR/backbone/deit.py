# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified based on official code of DeiT
from os import path as osp
from functools import partial
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from .vision_transformer import VisionTransformer, _cfg


PRETRAINED_DEPTH = 12


def interp_pos_emb(ckpt, src_img_size=224, patch_size=16, stride=16, dst_img_size=(480, 640)):
    assert dst_img_size[0] % patch_size == 0 and dst_img_size[1] % patch_size == 0
    assert patch_size % stride == 0
    src_h = src_w = src_img_size // patch_size
    dst_h, dst_w = dst_img_size[0] // stride, dst_img_size[1] // stride
    
    src_pos_emb = ckpt['model']['pos_embed']
    src_pos_emb_hw = rearrange(src_pos_emb[:, 1:], 'n (h w) c -> n c h w', h=src_h, w=src_w)
    # TODO: whether align_corners or not?
    dst_pos_emb_hw = F.interpolate(src_pos_emb_hw, size=(dst_h, dst_w), mode='bicubic', align_corners=False)
    dst_pos_emb = torch.cat([src_pos_emb[:, [0]],
                             rearrange(dst_pos_emb_hw, 'n c h w -> n (h w) c')], 1)
    ckpt['model']['pos_embed'] = dst_pos_emb
    return ckpt
    

def filter_checkpoint(ckpt, depth):
    assert depth <= PRETRAINED_DEPTH
    del_keys = list(filter(lambda x: x.split('.')[0] == 'blocks' and int(x.split('.')[1]) >= depth, ckpt['model'].keys()))
    for key in del_keys:
        del ckpt['model'][key]
    return ckpt


@register_model
def deit_tiny_patch16(pretrained=False, img_size=(480,640), in_chans=3, stride=16, depth=PRETRAINED_DEPTH, 
                      attn_type='full', attn_cfg={}, **kwargs):
    patch_size = 16
    assert patch_size % stride == 0
    
    model = VisionTransformer(
        img_size=img_size, patch_size=patch_size, stride=stride, embed_dim=192, depth=depth, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), in_chans=in_chans, attn_type=attn_type, attn_cfg=attn_cfg)
    model.default_cfg = _cfg()
    if pretrained:
        cur_dir = osp.dirname(osp.realpath(__file__))
        if in_chans == 3:
            checkpoint = torch.load(osp.join(cur_dir, 'deit_tiny_patch16_224-a1311bcf_headless.pth'))
        elif in_chans == 1:
            checkpoint = torch.load(osp.join(cur_dir, 'deit_tiny_patch16_224-a1311bcf_gray_headless.pth'))
        else:
            raise NotImplementedError()
        
        if img_size != 224 or img_size != (224, 224) or stride != patch_size:
            img_size = (img_size, img_size) if not isinstance(img_size, Iterable) else img_size
            checkpoint = interp_pos_emb(checkpoint, dst_img_size=img_size, patch_size=patch_size, stride=stride)
        if depth != 12:
            checkpoint = filter_checkpoint(checkpoint, depth)
        model.load_state_dict(checkpoint["model"], strict=attn_type=='full')  # omega in performer
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
