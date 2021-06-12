
from torch import nn
import torch
class simple_generator(nn.Module):
    def __init__(self, T, img_size=32):
        super(simple_generator, self).__init__()
        self.T= T
        self.img_size= img_size
        self.model= nn.Sequential(
            nn.Conv2d(self.T, 4, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(4, 3, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(3, 2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(2, 1, kernel_size=3, padding=1, stride=1))
    def forward(self, x):
        x= x.view(-1, self.T, self.img_size, self.img_size)
        return self.model(x).view(-1, 1, self.img_size, self.img_size)
    
    
class conv_relu_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(conv_relu_block, self).__init__()
        self.seq= nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size= kernel_size, padding=padding, stride=stride),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels))
    def forward(self, x):
        return self.seq(x)
    
    
class genv1(nn.Module):
    def __init__(self, T, img_size=32, img_channels=1, channel_list=[4,3,2,1], last_activation=None):
        super(genv1, self).__init__()
        self.T= T
        self.img_size= img_size
        self.img_channels=img_channels
        self.channel_list= channel_list
        self.last_activation= last_activation
        
        self.convrelu_blocks: nn.ModuleList[conv_relu_block] = nn.ModuleList()
            
        
        self.convrelu_blocks.append(conv_relu_block(self.T, self.channel_list[0], kernel_size= 3, stride= 1, padding=1))
        
        for idx in range(1, len(self.channel_list)):
            self.convrelu_blocks.append(conv_relu_block(self.channel_list[idx-1], self.channel_list[idx], kernel_size= 3, stride= 1, padding=1))

        self.last_layer = nn.Conv2d(in_channels=self.channel_list[-1], out_channels=self.img_channels, kernel_size= 3, stride= 1, padding=1)
    
    def forward(self, x):
        x= x.view(-1, self.T, self.img_size, self.img_size)
        
        for i in range(len(self.convrelu_blocks)):
            x= self.convrelu_blocks[i](x)
        x= self.last_layer(x)
        out= x.view(-1, self.img_channels, self.img_size, self.img_size)
        if self.last_activation=='sigmoid':out= torch.sigmoid(out)
        
        return out