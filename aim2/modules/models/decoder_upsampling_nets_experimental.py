import torch
from torch import nn
from torch.nn import functional as F
from modules.models.decoder_support_blocks import upsample_transconv_relu_bn_block, upsample_custom1_block
import math
import numpy as np
from modules.models.special_relations import *

class custom_v5(nn.Module): ##Cuda memory error for img_size= 128
    """
    Idea: Do Transpose correctly, repeat it over T channels, Optimize only the non-zero entries
    """
    def __init__(self, **kwargs):
        super(custom_v5, self).__init__()
        
        self.lambda_scale_factor = kwargs['lambda_scale_factor']
        self.T= kwargs['T']
        self.recon_img_size= kwargs['recon_img_size']
        Ht= kwargs['Ht'].detach()
        
        self.upscale_factor= 2**(self.lambda_scale_factor-1)
        self.yt_img_size= self.recon_img_size//self.upscale_factor
        
        self.Ht = nn.Parameter(Ht, requires_grad= True)
        self.A_transpose_special = convert_Ht2Atranspose(Ht, self.lambda_scale_factor) # shape: (1, recon_img_size^2, T*yt_img_size^2)
        self.seq_block= kwargs['upsample_postproc_block']

        
    def forward(self, yt, **kwargs):
        batch_size= yt.shape[0]
        yt= yt.view(batch_size, self.T, self.yt_img_size, self.yt_img_size) 
        yt_flatten = yt.flatten(start_dim= 1).unsqueeze(2) # shape: (n_samples, T*yt_img_size^2, 1)
        yt_upsample  =(self.A_transpose_special @ yt_flatten).reshape(batch_size, 1, self.recon_img_size, self.recon_img_size) # shape: (n_samples, 1, recon_img_size, recon_img_size)
        
        output= self.seq_block(yt_upsample) # shape: (batch_size, T, recon_img_size, recon_img_size)                    

        return output # (batch_size, T, recon_img_size, recon_img_size)
    

class custom_v6(nn.Module):
    """
    Idea: Do Transpose correctly, repeat it over T channels, No detaching from Ht
    """
    def __init__(self, **kwargs):
        super(custom_v6, self).__init__()
        
        self.lambda_scale_factor = kwargs['lambda_scale_factor']
        self.T= kwargs['T']
        self.recon_img_size= kwargs['recon_img_size']
        self.init_method=  kwargs['init_method']
        
        self.upscale_factor= 2**(self.lambda_scale_factor-1)
        self.yt_img_size= self.recon_img_size//self.upscale_factor
        
        self.seq_block= kwargs['upsample_postproc_block']
                
    def forward(self, yt, **kwargs):
        batch_size= yt.shape[0]
        yt= yt.view(batch_size, self.T, self.yt_img_size, self.yt_img_size) 
        yt_flatten = yt.flatten(start_dim= 1).unsqueeze(2) # shape: (n_samples, T*yt_img_size^2, 1)
        
        Ht= kwargs['Ht']
        A_transpose_special = convert_Ht2Atranspose(Ht, self.lambda_scale_factor) # shape: (1, recon_img_size^2, T*yt_img_size^2)
        
        yt_upsample  =(A_transpose_special @ yt_flatten).reshape(batch_size, 1, self.recon_img_size, self.recon_img_size) # shape: (n_samples, 1, recon_img_size, recon_img_size)
        
        output= self.seq_block(yt_upsample) # shape: (batch_size, T, recon_img_size, recon_img_size)                    

        return output # (batch_size, T, recon_img_size, recon_img_size)

    
    
class custom_v7(nn.Module):
    """
    Idea: efficient implementation of custom_v6
    """
    def __init__(self, **kwargs):
        super(custom_v7, self).__init__()
        
        self.lambda_scale_factor = kwargs['lambda_scale_factor']
        self.T= kwargs['T']
        self.recon_img_size= kwargs['recon_img_size']
        self.init_method=  kwargs['init_method']
        self.bias= kwargs['custom_upsampling_bias']
        
        self.upscale_factor= 2**(self.lambda_scale_factor-1)
        self.yt_img_size= self.recon_img_size//self.upscale_factor
        #self.expected_out_channels= self.T * self.upscale_factor**2
        
        
        self.seq_block= kwargs['upsample_postproc_block']
        if self.bias:self.biases= nn.Parameter(torch.randn(self.yt_img_size* self.yt_img_size, 1, self.upscale_factor**2), requires_grad= True)
                
    def forward(self, yt, **kwargs):
        batch_size= yt.shape[0]
        yt_input = yt.view(batch_size, self.T, self.yt_img_size, self.yt_img_size)
        yt= yt_input.reshape(batch_size, self.T, self.yt_img_size* self.yt_img_size).permute(2, 0, 1)  #shape: (yt_img_size**2, batch_size, T)
        
        weights= convert_Ht2Weights(kwargs['Ht'], self.upscale_factor) # shape: (yt_img_size**2, T, upscale_factor**2)

        if self.bias:
            yt_upsample= torch.matmul(yt, weights) + self.biases  #shape: (yt_img_size**2, batch_size, upscale_factor**2)
        else:yt_upsample= torch.matmul(yt, weights)  #shape: (yt_img_size**2, batch_size, upscale_factor**2)
            
        yt_upsample = yt_upsample.view(self.yt_img_size, self.yt_img_size, batch_size, self.upscale_factor, self.upscale_factor).permute(2, 0, 3, 1, 4) # shape: (batch_size, yt_img_size, upscale_factor, yt_img_size, upscale_factor)
        yt_upsample= yt_upsample.reshape(batch_size, self.recon_img_size, self.recon_img_size).unsqueeze(dim= 1) # shape: (batch_size, 1, yt_img_size*upscale_factor, yt_img_size*upscale_factor)
        output= self.seq_block(yt_upsample) # shape: (batch_size, T, recon_img_size, recon_img_size)                    
        return output
                