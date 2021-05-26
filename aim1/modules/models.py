import math
import torch 
from torch import nn
import torchvision

class LinearCNNEncoder(nn.Module):
    def __init__(self, input_channels, min_channels, max_channels, latent_dim, img_size):
        super(LinearCNNEncoder, self).__init__()
        self.n_layers = int(math.log2(img_size))
        print(f'Initializing LinearCNNEncoder : number of blocks : {self.n_layers}')

        self.latent_dim= latent_dim
        self.input_channels= input_channels
        self.img_size= img_size

        prev_channels= self.input_channels
        next_channels= min_channels

        self.encode_blocks: nn.ModuleList[nn.Conv2d] = nn.ModuleList()

        for idx in range(self.n_layers):
            self.encode_blocks.append(nn.Conv2d(prev_channels, next_channels, kernel_size= 3, stride= 2, padding=1))
            prev_channels= next_channels
            if idx == self.n_layers-2:
                next_channels= self.latent_dim
            else:
                next_channels= min(prev_channels*2, max_channels)

    def forward(self, x):
        x= x.view(-1, self.input_channels, self.img_size, self.img_size)
        for idx in range(self.n_layers):
            x= self.encode_blocks[idx](x)
        return x.view(-1, self.latent_dim)

class ResNet_block(nn.Module):  #inspired from https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781789538205/5/ch05lvl1sec34/gan-model-architectures
    def __init__(self,in_channels,mid_channels):
        super(ResNet_block,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=mid_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels,out_channels=in_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(in_channels))
    def forward(self,x):
        out= self.model(x) + x
        return torch.relu(out)

class decode_block(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, resnet=False):
        #print(f'decode block :: in_channels : {in_channels}, out_channels : {out_channels}, is_last : {is_last}, resnet : {resnet}')
        super(decode_block, self).__init__()
        self.is_last= is_last
        if resnet:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size= 4, stride=2, padding=1), # to change n_channels
                ResNet_block(out_channels, out_channels),
                ResNet_block(out_channels, out_channels),
                ResNet_block(out_channels, out_channels)) # last layer activation will be added considering is_last during forward pass
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size= 4, stride=2, padding=1), # using Upsample instead of TrasnposeConv was not given results but it may be effective if we train for more epochs/ with lower learning rate ... 
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        if self.is_last:
            self.last_block = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size= 3, stride=1, padding=1),
                nn.Tanh())
        
    def forward(self, x):
        out= self.block(x)
        if self.is_last:
            return self.last_block(out)
        else:
            return out

class DeepCNNDecoder(nn.Module):
    def __init__(self, latent_dim, max_channels, min_channels, img_channels, img_size, use_resnet=False):
        super(DeepCNNDecoder, self).__init__()
        self.img_size = img_size
        self.img_channels= img_channels
        self.latent_dim = latent_dim
        self.n_layers = int(math.log2(self.img_size)) 
        print(f'Initializing DeepCNNDecoder : number of blocks : {self.n_layers}') 
        self.decode_block = decode_block

        prev_channels = self.latent_dim
        next_channels= max_channels

        self.decode_blocks: nn.ModuleList[self.decode_block] = nn.ModuleList()

        for idx in range(self.n_layers):
            if idx==self.n_layers-1:
                self.decode_blocks.append(self.decode_block(prev_channels, next_channels, is_last=True, resnet= use_resnet))
            else:
                self.decode_blocks.append(self.decode_block(prev_channels, next_channels, is_last=False, resnet= use_resnet))
            
            prev_channels= next_channels
            if idx==self.n_layers-2:
                next_channels= self.img_channels
            else:
                next_channels= max(prev_channels//2, min_channels)
    def forward(self, z):
        x= z.view(-1, self.latent_dim, 1, 1)
        for idx in range(self.n_layers):
            x= self.decode_blocks[idx](x)
        return x.view(-1, self.img_channels, self.img_size, self.img_size)
    

class AE(nn.Module):
    def __init__(self, latent_dim, use_resnet=False, img_size= 32, img_channels=1, min_channels= 4, max_channels= 16):
        super(AE, self).__init__()

        self.img_size= img_size
        self.img_channels= img_channels
        min_channels= min_channels
        max_channels= max_channels
        

        self.LinearEncoder = LinearCNNEncoder(self.img_channels, min_channels, max_channels, latent_dim, self.img_size)
        self.Decoder = DeepCNNDecoder(latent_dim, max_channels, min_channels, self.img_channels, self.img_size, use_resnet)

    def forward(self, x):
        latent = self.LinearEncoder(x)
        return self.Decoder(latent).view(-1, self.img_channels, self.img_size, self.img_size)
         
