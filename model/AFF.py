import torch
import torch.nn.functional as F
import torch.nn as nn

class AFF(nn.Module):
    def __init__(self, op_channel_in: int, group_num: int = 4, group_size: int = 2, group_kernel_size: int = 3):
        super().__init__()
        self.op_channel1 = int(op_channel_in * 7/8)
        self.op_channel2 = int(op_channel_in-self.op_channel1)
        self.gn = nn.GroupNorm(num_channels=op_channel_in, num_groups=group_num)
        # Squeeze layers
        self.squeeze1 = nn.Conv1d(self.op_channel1, self.op_channel1, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv1d(self.op_channel2, self.op_channel2, kernel_size=1, bias=False)
        # Group-wise convolution
        self.GWC = nn.Conv1d(self.op_channel1, 4*self.op_channel1, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv1d(self.op_channel1, 4*self.op_channel1, kernel_size=1, bias=False)
        self.PWC2 = nn.Conv1d(self.op_channel2, 4*self.op_channel2, kernel_size=1, bias=False)
        self.advavg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # print("x:", x)
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / self.gn.weight.sum()
        # print("self.gn.weight", self.gn.weight)
        _, sorted_indices = torch.sort(w_gamma, descending=True)
        xnew = gn_x[:, sorted_indices, :]
        # print("重新排列后的序号:", sorted_indices.tolist())

        A_new = xnew[:, :self.op_channel1, :]
        B_new = xnew[:, self.op_channel1:, :]
        # Process A
        up = self.squeeze1(A_new)
        Y1 = self.GWC(up) + self.PWC1(up)
        # Process B
        low = self.squeeze2(B_new)
        Y2 = self.PWC2(low)
        # Concatenate results
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        return out

