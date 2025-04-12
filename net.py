import torch

from utils import *
import scipy.io as sio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class APG(torch.nn.Module):
    def __init__(self, LayerNo, rb_num):
        super(APG, self).__init__()

        onelayer = []
        self.LayerNo = LayerNo
        basicblock = BasicBlock

        for i in range(LayerNo):
            onelayer.append(basicblock(rb_num=rb_num))

        self.fcs = nn.ModuleList(onelayer)


    def forward(self, gt,mask):

        xu_real = zero_filled(gt, mask)


        x = xu_real

        x_prev = x  # 初始化 x_prev
        t = 1
        t_prev = t
        for i in range(self.LayerNo):
            [x, x_prev, t, t_prev] = self.fcs[i](x, xu_real, mask, x_prev, t, t_prev)

        x_final = x

        return x_final



class BasicBlock(torch.nn.Module):
    def __init__(self,rb_num):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        kernel_size = 3
        bias = True
        in_channels = 32
        growth_rate = 32


        self.conv_C = nn.Conv2d(1, in_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

        modules_body1 = [Residual_Block(in_channels, in_channels, 3, bias=True, res_scale=1) for _ in range(rb_num)]
        self.body1 = nn.Sequential(*modules_body1)


        modules_body2 = [Incep_Attention(in_channels, in_channels) for _ in range(1)]
        self.body2 = nn.Sequential(*modules_body2)


        self.conv_D = nn.Conv2d(in_channels, in_channels, 3, padding=(kernel_size // 2), bias=bias)


        modules_body3 = [Residual_Block(in_channels, in_channels, 3, bias=True, res_scale=1) for _ in range(rb_num)]
        self.body3 = nn.Sequential(*modules_body3)


        self.conv_E = nn.Conv2d(in_channels, 1, kernel_size, padding=(kernel_size // 2), bias=bias)

    def forward(self, x, PhiTb, mask, x_prev, t, t_prev):
        # APG迭代步骤
        # 加入动量
        v = x + (t_prev - 1) / t * ( x- x_prev)  # 计算v^(k)
        t_prev = t
        x_prev = x

        t = (1 + (1 + 4 * t_prev ** 2) ** 0.5) / 2

        # r_k的更新

        x = v - self.lambda_step * zero_filled(v, mask)
        # print(x.shape)
        x = x + self.lambda_step * PhiTb

        x_input = x


        x_C = self.conv_C(x_input)


        x_D = self.body1(x_C)
        x_forward = self.body2(x_D)


        x_G = self.conv_D(x_forward)
        x_G = self.body3(x_G)
        x_G = torch.mul(torch.sign(x_G), F.relu(torch.abs(x_G) - self.soft_thr))
        x_G = self.conv_E(x_G)

        x_pred = x_input + x_G
        # print(x_pred.shape)


        return [x_pred, x_prev, t, t_prev]



