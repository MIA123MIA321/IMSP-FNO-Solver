from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import operator
from functools import reduce
from collections import OrderedDict
import torch.optim as optim
from utils import * 


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv2d, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, self.modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes, :self.modes] = \
            self.compl_mul2d(x_ft[:, :, :self.modes, :self.modes], self.weights1)
        out_ft[:, :, -self.modes:, :self.modes] = \
            self.compl_mul2d(x_ft[:, :, -self.modes:, :self.modes], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    
class ResBlock_Simple(nn.Module):
    def __init__(self, width):
        super(ResBlock_Simple, self).__init__()
        self.width = width
        self.conv0 = nn.Conv2d(4,self.width,kernel_size=3,padding=1)
        self.conv1 = nn.Conv2d(self.width,2,kernel_size=1)
    def forward(self, u,f):
        u_init = u.clone()
        u = self.conv0(torch.cat([u,f],dim=1))
        u = F.gelu(u)
        u = self.conv1(u)
        return u+u_init

class ResBlock_FNO(nn.Module):
    def __init__(self, width, modes,depth_inner=2):
        super(ResBlock_FNO, self).__init__()
        self.width = width
        self.modes = modes
        self.depth_inner = depth_inner
        self.conv0 = nn.Conv2d(4,self.width,kernel_size=3,padding=1)
        self.spec_convs = nn.ModuleList([SpectralConv2d(self.width, self.width,
                                        self.modes) for _ in range(self.depth_inner)])
        self.weights = nn.ModuleList([nn.Conv2d(self.width, self.width, kernel_size=3,
                                               padding=1) for _ in range(self.depth_inner)])
        self.conv1 = nn.Conv2d(self.width,2,kernel_size=1)
    def forward(self, u,f):
        u_init = u.clone()
        u = self.conv0(torch.cat([u,f],dim=1))
        for i in range(len(self.spec_convs)):
            u1 = self.spec_convs[i](u)
            u2 = self.weights[i](u)
            u = F.gelu(u1+u2)
        u = self.conv1(u)
        return u+u_init

    

class DDNet(nn.Module):

    def __init__(self, N,k,depth=4,
                 depth_inner = 2,method = 'Simple',
                 Q_in = None,initial_guess=None,
                 width = 32,modes=12):
        super().__init__()
        self.N = N
        self.k = k
        self.depth = depth
        self.depth_inner = depth_inner
        self.method = method
        self.modes = modes
        self.width = width
        self.initial_guess = initial_guess
        if Q_in:
            self.Q = Q_in
        else:
            self.Q = np.zeros((self.N+1)**2)
        self.forward_mat = Matrix_Gen(self.N, self.Q, self.k,ToTensor=True).to(torch.float32).to(0)
        self.forward_mat_T = Matrix_Gen(self.N, self.Q, self.k,ToTensor=True,Transpose=True).to(torch.float32).to(0)
        self.mat_norm = torch.norm(self.forward_mat).to(0)
        self.diag = Diag_Gen(self.N, self.Q, self.k).to(0)
        if self.method == 'Simple':
            self.resblocks = nn.ModuleList([ResBlock_Simple(self.width) for _ in range(self.depth)])
        elif self.method == 'FNO':
            self.resblocks = nn.ModuleList([ResBlock_FNO(self.width,self.modes,
                                            self.depth_inner) for _ in range(self.depth)])
        
    def grad_eval(self,u,f):
        #  u,f:    (batchsize,2,size,size)
        #  output: (batchsize,2,size,size)
        size = f.shape
        batchsize = size[0]
        f1 = f.reshape(batchsize,-1)
        u1 = u.reshape(batchsize,-1)
        output = torch.stack([torch.mv(self.forward_mat_T,(torch.mv(self.forward_mat,u1[i])-f1[i]) ) for i in range(batchsize)],dim = 0)
        output = torch.stack([output[:,:(size[2]**2)],output[:,(size[2]**2):]],dim=1)
        return output.reshape(size)/(2*self.mat_norm**2)
        
        
    def forward(self,f):  # (batchsize,2,size,size)
        size = f.shape
        batchsize = size[0]
        if self.initial_guess:
            u = self.initial_guess
            
        else:
            u_tmp = ((f[:,0]+1j*f[:,1]).reshape(batchsize,-1)/(self.diag.repeat(batchsize,1)).to(f.device))
            u = torch.stack([u_tmp.real,u_tmp.imag],1)
            u = u.reshape(size)
        # u = u/(abs(u).max())
            
        for i in range(len(self.resblocks)):
            u = self.resblocks[i](u,self.grad_eval(u,f))
        size = f.shape
        f1 = f.reshape(batchsize,-1)
        u1 = u.reshape(batchsize,-1)
        output = [0.5*torch.norm(torch.mv(self.forward_mat,u1[i])-f1[i])**2 for i in range(batchsize)]
        
        return u,output