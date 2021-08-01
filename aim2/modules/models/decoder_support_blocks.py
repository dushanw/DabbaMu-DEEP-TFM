from torch import nn
import torch
    
class upsample_transconv_relu_bn_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(transposeconv_block, self).__init__()
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
    