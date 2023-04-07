import math
import torch
import numpy as np
from torch.nn import functional as F
from torch._jit_internal import Optional
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import nn
from torch.nn import init
import matplotlib.pyplot as plt
import seaborn as sns
import operator
from functools import reduce
from collections import OrderedDict
import torch.optim as optim


def get_parameter_number(net):
    total_num = sum(np.prod(p.size()) for p in net.parameters())
    trainable_num = sum(np.prod(p.size()) for p in net.parameters() if p.requires_grad)
    print('Total: ', total_num)
    print('Trainable: ', trainable_num)
    

class DOConv2d(Module):
    """
       DOConv2d can be used as an alternative for torch.nn.Conv2d.
       The interface is similar to that of Conv2d, with one exception:
            1. D_mul: the depth multiplier for the over-parameterization.
       Note that the groups parameter switchs between DO-Conv (groups=1),
       DO-DConv (groups=in_channels), DO-GConv (otherwise).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, D_mul=None, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros', simam=False):
        super(DOConv2d, self).__init__()

        kernel_size = (kernel_size, kernel_size)
        stride = (stride, stride)
        padding = (padding, padding)
        dilation = (dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))
        
        #################################### Initailization of D & W ###################################
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
        self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, self.D_mul))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

        if M * N > 1:
            self.D = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)
            self.D.data = torch.from_numpy(init_zero)

            eye = torch.reshape(torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N))
            D_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0:  # the cases when D_mul > M * N
                zeros = torch.zeros([in_channels, M * N, self.D_mul % (M * N)])
                self.D_diag = Parameter(torch.cat([D_diag, zeros], dim=2), requires_grad=False)
            else:  # the case when D_mul = M * N
                self.D_diag = Parameter(D_diag, requires_grad=False)
        ##################################################################################################
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def __setstate__(self, state):
        super(DOConv2d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def _conv_forward(self, input, weight):
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        DoW_shape = (self.out_channels, self.in_channels // self.groups, M, N)
        if M * N > 1:
            ######################### Compute DoW #################
            # (input_channels, D_mul, M * N)
            D = self.D + self.D_diag
            W = torch.reshape(self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul))

            # einsum outputs (out_channels // groups, in_channels, M * N),
            # which is reshaped to
            # (out_channels, in_channels // groups, M, N)
            DoW = torch.reshape(torch.einsum('ims,ois->oim', D, W), DoW_shape)
            #######################################################
        else:
            DoW = torch.reshape(self.W, DoW_shape)

        return self._conv_forward(input, DoW)

    

class BasicConv_do(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=False, norm=False, relu=True, transpose=False,
                 relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
        super(BasicConv_do, self).__init__()

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                DOConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

    
class ResBlock_do_fft_bench(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_do_fft_bench, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=False)
        )
        self.dim = out_channel
        self.norm = norm
    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y


    
class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8, ResBlock=None):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DBlock(nn.Module):
    def __init__(self, channel, num_res=8, ResBlock=None):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class AFF(nn.Module):
    def __init__(self, in_channel, out_channel, BasicConv):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)

class SCM(nn.Module):
    def __init__(self, out_plane, BasicConv, inchannel=3):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(inchannel, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-inchannel, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)

class FAM(nn.Module):
    def __init__(self, channel, BasicConv):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out
    
    
class DeepRFT(nn.Module):
    def __init__(self):
        super(DeepRFT, self).__init__()
        
        num_res = 4
        base_channel = 32
        data_channel = 2
        BasicConv = BasicConv_do
        ResBlock = ResBlock_do_fft_bench
        

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res, ResBlock=ResBlock),
            EBlock(base_channel*2, num_res, ResBlock=ResBlock),
            EBlock(base_channel*4, num_res, ResBlock=ResBlock),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(data_channel, base_channel, kernel_size=3, relu=True, stride=1),  # 不变
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, data_channel, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res, ResBlock=ResBlock),
            DBlock(base_channel * 2, num_res, ResBlock=ResBlock),
            DBlock(base_channel, num_res, ResBlock=ResBlock)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, data_channel, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, data_channel, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1, BasicConv=BasicConv),
            AFF(base_channel * 7, base_channel*2, BasicConv=BasicConv)
        ])

        self.FAM1 = FAM(base_channel * 4, BasicConv=BasicConv)
        self.SCM1 = SCM(base_channel * 4, BasicConv=BasicConv,inchannel = data_channel)
        self.FAM2 = FAM(base_channel * 2, BasicConv=BasicConv)
        self.SCM2 = SCM(base_channel * 2, BasicConv=BasicConv,inchannel = data_channel)

    def forward(self, x):
        
        
        # Input (3,64,64)
        x_2 = F.interpolate(x, scale_factor=0.5)  # (3,32,32)
        x_4 = F.interpolate(x_2, scale_factor=0.5)  # (3,16,16)
        MAX0,MAX1,MAX2 = x.abs().max(),x_2.abs().max(),x_4.abs().max()
        x,x_2,x_4 = x/MAX0,x_2/MAX1,x_4/MAX2
        z2 = self.SCM2(x_2)  # (64,32,32)
        z4 = self.SCM1(x_4)  # (128,16,16)

        outputs = list()

        x_ = self.feat_extract[0](x) # (32,64,64)
        res1 = self.Encoder[0](x_) # (32,64,64)

        z = self.feat_extract[1](res1) # (64,32,32)
        z = self.FAM2(z, z2) # (64,32,32)
        res2 = self.Encoder[1](z) # (64,32,32)

        z = self.feat_extract[2](res2) # (128,16,16)
        z = self.FAM1(z, z4)  # (128,16,16)
        z = self.Encoder[2](z)  # (128,16,16)

        z12 = F.interpolate(res1, scale_factor=0.5)  # (32,32,32)
        z21 = F.interpolate(res2, scale_factor=2)  # (64,64,64)
        z42 = F.interpolate(z, scale_factor=2)  # (128,32,32)
        z41 = F.interpolate(z42, scale_factor=2) # (128,64,64)

        res2 = self.AFFs[1](z12, res2, z42)  # (64,32,32)
        res1 = self.AFFs[0](res1, z21, z41)  # (32,64,64)

        z = self.Decoder[0](z)  # (128,16,16)
        z_ = self.ConvsOut[0](z)  # (3,16,16)
        z = self.feat_extract[3](z)  # (64,32,32)
        outputs.append((z_+x_4)*MAX2)

        z = torch.cat([z, res2], dim=1)  # (128,32,32)
        z = self.Convs[0](z)  # (64,32,32)
        z = self.Decoder[1](z)  # (64,32,32)
        z_ = self.ConvsOut[1](z)  # (3,32,32)
        z = self.feat_extract[4](z)  # (32,64,64)
        outputs.append((z_+x_2)*MAX1)

        z = torch.cat([z, res1], dim=1) # (64,64,64)
        z = self.Convs[1](z)  # (32,64,64)
        z = self.Decoder[2](z)  # (32,64,64)
        z = self.feat_extract[5](z)  # (3,64,64)
        outputs.append((z+x)*MAX0)
        return outputs[::-1]
        

        
        
######################################################################################################
######################################################################################################


def generate_phi_incident(ntrain,N,k,angle,order):
    phi_incident = np.zeros((2,N + 1, N + 1))
    l = np.linspace(0,1,N+1)
    y,x = np.meshgrid(l,l)
    phii = np.exp(1j*k*(x*np.cos(2*np.pi*order/angle)+y*np.sin(2*np.pi*order/angle)))
    phi_incident[0],phi_incident[1] = phii.real,phii.imag
    return torch.from_numpy(phi_incident).unsqueeze(0).repeat(ntrain,1,1,1).to(torch.float64)



class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.eps = 1e-8

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[2] - 1.0)

        all_norms = (h**(self.d / self.p)) * \
            torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(
            num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / (y_norms+self.eps))
            else:
                # print(diff_norms,y_norms)
                return torch.sum(diff_norms / (y_norms+self.eps))

        return diff_norms / y_norms
    
    def rel1(self,x,y):
        batchsize = x.shape[0]
        angle = x.shape[1]
        tmp = 0
        for i in range(batchsize):
            tmp1,tmp2 = 0,0
            for j in range(angle):
                tmp1+=torch.norm((x[i,j]-y[i,j]),2)
                tmp2+=torch.norm(y[i,j],2)
            tmp += tmp1/tmp2
        return tmp
    
    
    def rel2(self,x,y):
        batchsize = x.shape[0]
        angle = x.shape[1]
        tmp = 0
        for i in range(batchsize):
            tmp1,tmp2 = 0,0
            for j in range(angle):
                tmp1+=torch.norm((x[i,j]-y[i,j]),2)
                tmp2+=torch.norm(y[i,j],2)
                tmp += tmp1/tmp2
        return tmp
    
    def fft_loss(self,x,y):
        diff = torch.fft.fft2(x) - torch.fft.fft2(y)
        loss = torch.mean(abs(diff))
        return loss*x.shape[0]
                

    def __call__(self, x, y):
        return self.rel(x, y)
    

def list_loss(x,y,s=64,loss_type='rel',average=False):
    ans = 0.
    if loss_type=='rel':
        loss_fun = LpLoss(size_average=False).abs
    elif loss_type=='rel':
        loss_fun = LpLoss(size_average=False).fft_loss
    for i in range(len(x)):
        ans +=  loss_fun(x[i].reshape(-1,2,s,s),y)
        y = y[:,:,::2,::2]
        s = s//2
    if average:
        ans = ans/x[0].shape[0]
        