
import torch
import torch.nn as nn

import numpy as np

from STR import STR
from TTR import TTR
from model.net import Unit2D, conv_init, import_class
from model.unit_agcn import unit_agcn

class STR_TTR(nn.Module):

    def __init__(self,
                 channel,
                 num_class,
                 backbone_in_c = 128,
                 num_frame = 180,
                 num_joints = 22,
                 style = None,
                 graph=None,
                 graph_args=dict(),
                 mask_learning=False,
                 use_local_bn=False,
                 ):
        super().__init__()
        if graph is None:
            raise ValueError()
        else:
            # print(graph)
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
            self.A = torch.from_numpy(self.graph.A.astype(np.float32))


        self.num_joints = num_joints
        self.num_frame = num_frame
        self.num_class = num_class
        self.backbone_in_c = backbone_in_c
        self.style = style

        self.gcn = unit_agcn(
            channel,
            self.backbone_in_c,
            self.A,
            mask_learning=mask_learning,
            use_local_bn=use_local_bn)

        self.tcn = Unit2D(self.backbone_in_c, self.backbone_in_c, kernel_size=9)

        if self.style == 'STR':
            self.modelA = STR(num_class, num_frame=self.num_frame, num_joints= self.num_joints, in_chans=128, embed_dim_ratio=256, depth=6,
                              num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)

        else:
            self.modelB = TTR(num_class, num_frame=self.num_frame, num_joints= self.num_joints, in_chans=128, embed_dim_ratio=256,
                             depth=6,
                             num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)





    def forward(self, x):
        # input shape: [batch_size, time_len, joint_num, 3]
        x = x.permute(0,3,1,2)

        N, C, T, V= x.size()  ##Batch  channel  frame  joint

        x = x.permute(0, 1, 2, 3).contiguous().view(N, C, T, V)

        x = self.gcn(x)

        x = self.tcn(x)

        if self.style == 'STR':
            x_st = self.modelA(x)
            pred = x_st


        else:
            x_ts = self.modelB(x)
            pred = x_ts

        return pred