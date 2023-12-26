import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from visualizer import get_local
# from zjt.trans.net import conv_init
from model.unit_agcn import unit_agcn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @get_local('attention_map')
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attention_map = attn
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class T_TCN(nn.Module):
    def __init__(self,class_num, A ,num_frame=180, num_joints=22, in_chans=128, embed_dim_ratio=256, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,mask_learning = False,use_local_bn=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 3D joints have 3 channels: (x,y,z)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.class_num = class_num
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * 2   #### temporal embed_dim is num_joints * spatial embedding dim ratio
            #### output dimension is num_joints * 3

        ### temporal_pos_embedding
        self.temporal_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame , embed_dim_ratio))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim_ratio))  # nn.Parameter()定义可学习参数

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(embed_dim_ratio, embed_dim)
        # self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints + 1, embed_dim))
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.Spatial_cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)




        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim)
        self.Temporal_norm = norm_layer(embed_dim_ratio)
        self.A =A
        in_channels = 256
        out_channels = 512
        self.gcn0 = unit_agcn(
            in_channels,
            out_channels,
            self.A,
            mask_learning=mask_learning,
            use_local_bn=use_local_bn)
        ####### A easy way to implement weighted mean
        pool = 'cls'

        assert pool in {'cls', 'mean'}

        self.pool = pool
        self.to_latent = nn.Identity()
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=1, kernel_size=1)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, class_num)
        )
        self.num_class = class_num
        self.fcn = nn.Conv1d(512, class_num, kernel_size=1)
        # conv_init(self.fcn)

    def Temporal_forward_features(self,x):

        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b p) f c', )
        x = self.temporal_patch_to_embedding(x)
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        b, f, c = x.shape
        x = rearrange(x, '(b  p)  f  c  -> b  c  f  p',p=p)

        return x

    def Spatial_forward_features(self, x):
        # print("s",x.shape)   # 32 256 180 22

        x = self.gcn0(x)

        return x




    def forward(self, x):
        x = self.Temporal_forward_features(x)

        x = self.Spatial_forward_features(x)

        b, c, f, p = x.shape

        x = F.avg_pool2d(x, kernel_size=(1, p))

        # M pooling
        c = x.size(1)
        t = x.size(2)
        x = x.view(b, c, t)

        # T pooling
        x = F.avg_pool1d(x, kernel_size=x.size()[2])
        # 8 180 1
        # C fcn
        x = self.fcn(x)
        x = F.avg_pool1d(x, x.size()[2:])
        pred = x.view(b, self.num_class)

        return pred