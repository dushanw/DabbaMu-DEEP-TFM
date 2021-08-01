import torch
from torch import nn
from torch.nn import functional as F
from modules.models.decoder_support_blocks import upsample_transconv_relu_bn_block


class bicubic_interp():
    def __init__(self, lambda_scale_factor, T):
        self.lambda_scale_factor= lambda_scale_factor

    def __call__(self, yt):
        for _ in range(self.lambda_scale_factor-1): # upscaling
            yt= F.interpolate(yt, scale_factor= 2, mode='bicubic')
        return yt
    
class learnable_transpose_conv(nn.Module):
    def __init__(self, lambda_scale_factor, T):
        super(learnable_transpose_conv, self).__init__()
        self.lambda_scale_factor= lambda_scale_factor
        self.T= T
        
        upsample_block =  upsample_transconv_relu_bn_block # this doing 2x2 upsampling using (TransposeConv + relu + BN)
        
        self.upsample_blocks: nn.ModuleList[upsample_block] = nn.ModuleList()
        for _ in range(self.lambda_scale_factor-1):
            self.upsample_blocks.append(upsample_block(self.T , self.T))
            
    def forward(self, x):        
        for i in range(self.lambda_scale_factor-1):
            x= self.upsample_blocks[i](x)
        return x