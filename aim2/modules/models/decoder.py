
from torch import nn
import torch
from modules.models.decoder_support_blocks import *

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
    

class base_decoder(nn.Module):
    def __init__(self, T, img_size=32, img_channels=1, channel_list=[4,3,2,1], last_activation=None, decoder_block=None, upsampling_net= None):
        super(base_decoder, self).__init__()
        self.T= T
        self.img_size= img_size
        self.img_channels=img_channels
        self.channel_list= channel_list
        self.last_activation= last_activation
        
        self.upsample_net = upsampling_net
        
        self.decoder_blocks: nn.ModuleList[decoder_block] = nn.ModuleList()
        self.decoder_blocks.append(decoder_block(self.T, self.channel_list[0]))
        for idx in range(1, len(self.channel_list)):
            self.decoder_blocks.append(decoder_block(self.channel_list[idx-1], self.channel_list[idx]))
        self.last_layer = nn.Conv2d(in_channels=self.channel_list[-1], out_channels=self.img_channels, kernel_size= 3, stride= 1, padding=1)
    
    def forward(self, x, **kwargs):
        x= self.upsample_net(x, **kwargs) ## do upsampling 
        x= x.view(-1, self.T, self.img_size, self.img_size)
        
        for i in range(len(self.decoder_blocks)):
            x= self.decoder_blocks[i](x)
        x= self.last_layer(x)
        out= x.view(-1, self.img_channels, self.img_size, self.img_size)
        if self.last_activation=='sigmoid':out= torch.sigmoid(out)
        
        return out

    
    
class genv1(nn.Module):
    def __init__(self, T, img_size=32, img_channels=1, channel_list=[4,3,2,1], last_activation=None, upsampling_net=None):
        super(genv1, self).__init__()
        self.decoder = base_decoder(T, img_size, img_channels, channel_list, last_activation, decoder_block= conv_relu_bn_block, upsampling_net= upsampling_net)
        
        #print('decoder : \n', self.decoder)
        #for name, param in self.decoder.named_parameters():
        #    print(f'{name} : {param.requires_grad}')
    def forward(self, x, **kwargs):
        out = self.decoder(x, **kwargs)
        return out
    
class genv2(nn.Module):
    def __init__(self, T, img_size=32, img_channels=1, channel_list=[4,3,2,1], last_activation=None, upsampling_net=None):
        super(genv2, self).__init__()
        self.decoder = base_decoder(T, img_size, img_channels, channel_list, last_activation, decoder_block= ResNet_block_v1, upsampling_net= upsampling_net)
        #print('decoder : \n', self.decoder)
        #for name, param in self.decoder.named_parameters():
        #    print(f'{name} : {param.requires_grad}')
            
    def forward(self, x, **kwargs):
        out = self.decoder(x, **kwargs)
        return out