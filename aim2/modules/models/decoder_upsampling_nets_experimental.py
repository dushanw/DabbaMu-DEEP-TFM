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
        self.A_transpose_special_tiled = convert_Ht2Atranspose(Ht, self.lambda_scale_factor)
        
    def forward(self, yt, **kwargs):
        batch_size= yt.shape[0]
        yt= yt.view(batch_size, self.T, self.yt_img_size, self.yt_img_size) 
        yt_flatten = yt.flatten(start_dim= 1).unsqueeze(2).unsqueeze(1) # shape: (n_samples, 1, T*yt_img_size^2, 1)
        output  =(self.A_transpose_special_tiled @ yt_flatten).reshape(batch_size, self.T, self.recon_img_size, self.recon_img_size) # shape: (n_samples, T, recon_img_size, recon_img_size)
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
                
    def forward(self, yt, **kwargs):
        batch_size= yt.shape[0]
        yt= yt.view(batch_size, self.T, self.yt_img_size, self.yt_img_size)
        yt_flatten = yt.flatten(start_dim= 1).unsqueeze(2).unsqueeze(1) # shape: (n_samples, 1, T*yt_img_size^2, 1)
        
        Ht= kwargs['Ht']
        A_transpose_special_tiled = convert_Ht2Atranspose(Ht, self.lambda_scale_factor)
        output  =(A_transpose_special_tiled @ yt_flatten).reshape(batch_size, self.T, self.recon_img_size, self.recon_img_size) # shape: (n_samples, T, recon_img_size, recon_img_size)
        
        return output # (batch_size, T, recon_img_size, recon_img_size)
