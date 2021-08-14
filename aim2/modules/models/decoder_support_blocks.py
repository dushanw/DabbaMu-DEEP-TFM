from torch import nn
import torch
    
class upsample_transconv_relu_bn_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsample_transconv_relu_bn_block, self).__init__()
        self.seq= nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size= 4, padding=1, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels))
    def forward(self, x):
        return self.seq(x)    
    
class conv_relu_bn_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_relu_bn_block, self).__init__()
        self.seq= nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size= 3, padding=1, stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels))
    def forward(self, x):
        return self.seq(x)
    

class conv_bn_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(conv_bn_block, self).__init__()
        self.seq= nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size= kernel_size, padding=padding, stride=stride),
                nn.BatchNorm2d(out_channels))
    def forward(self, x):
        return self.seq(x)
    
    
class ResNet_block_v1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet_block_v1,self).__init__()
        
        mid_channels= in_channels
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())
        
        self.conv_block = conv_bn_block(in_channels, out_channels, kernel_size= 3, padding=1, stride=1)
    def forward(self,x):
        res_out= self.resblock(x) + x
        out = self.conv_block(res_out)
        return out #no activation 
    
    
class upsample_custom1_block(nn.Module):
    """
    used in decoder_upsampling_nets.custom1()
    
    Idea: each spatial location with c-channel values will be upsampled to corresponding region of upsampled image through fully 
    connected layers. Throughout the upsampling, number of channels will remain at T. 
    Therefore one pixel (with all channels) of yt will be inputted to linear layer which have inputs of 
    :: in_channels: T, out_channels: (upscale_factor**2)*T.
    There are (yt_size*yt_size) pixels in yt. Therefore, there will be (yt_size*yt_size) number of linear layers.
    """
    def __init__(self, in_channels, out_channels, upscale_factor): #eg: (batch_size, in_channels)-> (batch_size, out_channels, upscale_factor, upscale_factor)
        super(upsample_custom1_block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale_factor = upscale_factor

        self.block = nn.Sequential(
            nn.Linear(self.in_channels, (self.upscale_factor**2)*self.out_channels),
            nn.ReLU()
        )
    def forward(self, x): # shape: (batch_size, in_channels) -> channel along single spatial location
        x= x.view(-1, self.in_channels)
        x= self.block(x)

        x = x.view(-1, self.out_channels, self.upscale_factor, self.upscale_factor)

        return x