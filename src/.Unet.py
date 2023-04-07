from .utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FFTCONV(nn.Module):
    def __init__(self, in_channels,out_channels,modes=8):
        super(FFTCONV, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.spec_conv = SpectralConv2d(in_channels, out_channels, modes=self.modes)
        self.w = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1)
        self.three_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.three_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        batchsize = x.shape[0]
        x1, x2 = x.clone(), x.clone()
        
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x1 = F.gelu(self.w(x1)+self.spec_conv(x1))
        
        x2 = self.three_conv1(x2)
        x2 = nn.GELU()(x2)
        x2 = self.three_conv2(x2)
        
        return x1+x2+self.bn(self.conv(x))
    

class DoubleConvolution(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # First $3 \times 3$ convolutional layer
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.GELU()
        
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor):
        # Apply the two convolution layers and activations
        x = self.first(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)


class DownSample(nn.Module):

    def __init__(self):
        super().__init__()
        # Max pooling layer
        self.pooler = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.pooler(x)
        


class UpSample(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Up-convolution
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)


class Concat(nn.Module):

    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor, dim = 1):

        x = torch.cat([x, contracting_x], dim=dim)
        return x                              
        
        
    
class UNet(nn.Module):
    """
    ## U-Net
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: number of channels in the input image
        :param out_channels: number of channels in the result feature map
        """
        super().__init__()
        

        # Double convolution layers for the contracting path.
        # The number of features gets doubled at each step starting from $64$.
        self.down_conv = nn.ModuleList([DoubleConvolution(8, 64),DoubleConvolution(64,128)])
        # Down sampling layers for the contracting path
        self.down_sample = nn.ModuleList([DownSample() for _ in range(3)])

        # The two convolution layers at the lowest resolution (the bottom of the U).
        self.middle_conv = DoubleConvolution(128, 256)

        # Up sampling layers for the expansive path.
        # The number of features is halved with up-sampling.
        self.up_sample = nn.ModuleList([UpSample(256, 128),UpSample(128, 64)])
        # Double convolution layers for the expansive path.
        # Their input is the concatenation of the current feature map and the feature map from the
        # contracting path. Therefore, the number of input features is double the number of features
        # from up-sampling.
        self.up_conv = nn.ModuleList([DoubleConvolution(256, 128),DoubleConvolution(128, 64)])
        # Crop and concatenate layers for the expansive path.
        self.concat = nn.ModuleList([Concat() for _ in range(3)])
        # Final $1 \times 1$ convolution layer to produce the output
        self.last_conv = nn.Conv2d(64, 16, kernel_size=2,padding=1)
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        self.x_normalizer = None
        self.y_normalizer = None

    def forward(self, x: torch.Tensor):
        """
        :param x: input image
        """
        # To collect the outputs of contracting path for later concatenation with the expansive path.
        pass_through = []
        MAX = x.abs().max()
        x = x/MAX
        x_init = x.clone()
        x = torch.cat((x_init[:,:,:(-1),:(-1)],x_init[:,:,:(-1),1:]),dim = 1)
        x = torch.cat((x,x_init[:,:,1:,1:]),dim = 1)
        x = torch.cat((x,x_init[:,:,1:,:-1]),dim = 1)
        # Contracting path
        for i in range(len(self.down_conv)):
            # Two $3 \times 3$ convolutional layers
            x = self.down_conv[i](x)
            # Collect the output
            pass_through.append(x)
            # Down-sample
            x = self.down_sample[i](x)

        # Two $3 \times 3$ convolutional layers at the bottom of the U-Net
        x = self.middle_conv(x)
        # Expansive path
        for i in range(len(self.up_conv)):
            # Up-sample
            x = self.up_sample[i](x)
            # Concatenate the output of the contracting path
            x = self.concat[i](x, pass_through.pop())
            # Two $3 \times 3$ convolutional layers
            x = self.up_conv[i](x)

        # Final $1 \times 1$ convolution layer
        x = self.last_conv(x)
        x = F.gelu(x)
        x = self.final_conv(x)
        x = x*MAX
        return x