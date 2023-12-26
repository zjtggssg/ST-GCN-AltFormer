import torch
import torch.nn as nn
import torch.nn.functional as F

class DATA_MAX(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DATA_MAX, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.proj = nn.Linear(2*in_channels, out_channels)

    def forward(self, x):

        x_pool = self.pool(x)
        x = x.transpose(1, 2)
        x_pool = x_pool.transpose(1, 2)# (batch_size, T, C)
        x_diff = torch.abs(x[:, 1::2, :] - x[:, 0::2, :])  # (batch_size, T/2, C)
        x_concat = torch.cat([x_pool, x_diff], dim=-1)  # (batch_size, T/2, 2*C)
        # 16 90 1024
        x_proj = self.proj(x_concat)  # (batch_size, T/2, C)
        # 16 90 512
         #16 90 512
        x_proj = x_proj.transpose(1, 2)  # (batch_size, C, T/2)
         # 16 512 90
        x_proj = self.pool(x_proj)  # (batch_size, C, T/4)
        x_proj = self.pool(x_proj)  # (batch_size, C, T/8)
        t = x_proj.shape[2]
        #16 512 22
        x_proj = F.avg_pool2d(x_proj, kernel_size=(1, t))
        return x_proj