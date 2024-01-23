
from model.AltFormer.model_ST import ST
from model.AltFormer.model_TS import TS

import torch
import torch.nn as nn

import numpy as np


from model.net import Unit2D, import_class
from model.unit_agcn import unit_agcn

class ST_GCN_AltFormer(nn.Module):

    def __init__(self,
                 channel,
                 num_class,
                 backbone_in_c = 128,
                 num_frame = 180,
                 num_joints=22,
                 style = None,
                 graph=None,
                 graph_args=dict(),
                 mask_learning=False,
                 use_local_bn=False,
                 ):
        super(ST_GCN_AltFormer, self).__init__()
        if graph is None:
            raise ValueError()
        else:
            # print(graph)
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
            self.A = torch.from_numpy(self.graph.A.astype(np.float32))

        self.num_frame = num_frame
        self.num_joints = num_joints
        self.num_class = num_class
        self.backbone_in_c = backbone_in_c
        self.style = style

        self.gcn0 = unit_agcn(
            channel,
            self.backbone_in_c,
            self.A,
            mask_learning=mask_learning,
            use_local_bn=use_local_bn)

        self.tcn0 = Unit2D(self.backbone_in_c, self.backbone_in_c, kernel_size=9)


        self.modelA = ST(num_class, num_frame=self.num_frame, num_joints= self.num_joints, in_chans=128, embed_dim_ratio=256, depth=6,
                             num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)

        self.modelB = TS(num_class, num_frame=self.num_frame, num_joints= self.num_joints, in_chans=128, embed_dim_ratio=256,
                             depth=6,
                             num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)



    def forward(self, x):
        # input shape: [batch_size, time_len, joint_num, 3]
        x = x.permute(0,3,1,2)

        N, C, T, V= x.size()  ##Batch  channel  frame  joint

        x = x.permute(0, 1, 2, 3).contiguous().view(N, C, T, V)

        x = self.gcn0(x)

        x = self.tcn0(x)

        if self.style == 'ST':
            x_st = self.modelA(x)
            pred = x_st

        elif self.style == 'TS':
            x_ts = self.modelB(x)
            pred = x_ts

        else:
            x_st = self.modelA(x)
            x_ts = self.modelB(x)
            pred = x_ts + x_st

        return pred

