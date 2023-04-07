from utils import *


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.w1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.w2 = nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1)
    def forward(self,x):
        x = self.w1(x)
        x = F.gelu(x)
        x = self.w2(x)
        return x
        
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

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
    

class Fourier_layer(nn.Module):
    def __init__(self, width, modes,last = False):
        super(Fourier_layer, self).__init__()
        self.width = width
        self.modes = modes
        self.last = last
        self.conv = SpectralConv2d(self.width, self.width, self.modes)
        self.w = nn.Conv2d(self.width, self.width, kernel_size=1)
        # self.bn = nn.BatchNorm2d(self.width)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.w(x)
        if self.last:
            return x1+x2
        else:
            return F.gelu(x1+x2)
        
        
class FNO2d_part(nn.Module):
    def __init__(self, modes, width, depth, in_channel,out_channel):
        super(FNO2d_part, self).__init__()

        self.modes = modes
        self.width = width
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.dc = DoubleConv(1,8)
        self.fc0 = nn.Linear(2+in_channel, self.width)
        self.depth = depth
        self.fourier_layers = nn.ModuleList([Fourier_layer(self.width,self.modes,False) for i in range(self.depth-1)]+[Fourier_layer(self.width,self.modes,True)])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.out_channel)

    def forward(self, x):
        MAX = x.abs().max()
        x = x/MAX
        x = self.dc(x)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for i in range(len(self.fourier_layers)):
            x = self.fourier_layers[i](x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x*MAX
    
    def get_grid(self, shape, device):
        batchsize, channel, size_x, size_y = shape[0], shape[1], shape[2], shape[3]
        x = np.linspace(0, 1, size_x)
        y = np.linspace(0, 1, size_y)
        X, Y = np.meshgrid(x, y)
        X = torch.tensor(X, dtype=torch.float).to(device)
        Y = torch.tensor(Y, dtype=torch.float).to(device)
        gridx = X.reshape(1, 1, size_x, size_y).repeat([batchsize, 1, 1, 1])
        gridy = Y.reshape(1, 1, size_x, size_y).repeat([batchsize, 1, 1, 1])
        return torch.cat((gridx, gridy), dim=1)

    
class FNO2d(nn.Module):
    def __init__(self, modes, width, depth):
        super(FNO2d, self).__init__()
        self.modes = modes
        self.width = width
        self.depth = depth
        self.conv = FNO2d_part(self.modes, self.width, self.depth, 1, 2)
    def forward(self,x):
        return self.conv(x)
        

class FNO2d_2d(nn.Module):
    def __init__(self, modes, width, depth, k):
        super(FNO2d_2d, self).__init__()
        self.modes = modes
        self.width = width
        self.depth = depth
        self.k = k
        self.conv1 = FNO2d_part(self.modes, self.width, self.depth, self.k, 1, 1)
        self.conv2 = FNO2d_part(self.modes, self.width, self.depth, self.k, 1, 1)
    def forward(self,x):
        return torch.cat([self.conv1(x[:,:1])-self.conv2(x[:,1:2]),
                          self.conv1(x[:,1:2])+self.conv2(x[:,:1])],dim=1)
    
class FNO2d_modified(nn.Module):
    def __init__(self, modes, width, depth):
        super(FNO2d_modified, self).__init__()
        self.modes = modes
        self.width = width
        self.depth = depth
        self.conv = FNO2d_part(self.modes, self.width, self.depth, 1, 2)
    def forward(self,data):
        # x[:,0:1] --> q
        # x[:,1:3] --> f
        q = data[:,0:1]
        f = data[:,1:3]
        u0_approx = model_eval(self.conv, q, f)
        return u0_approx
                        