import torch
from torch import nn
from torch.nn import functional as F
from modules.models.decoder_support_blocks import upsample_transconv_relu_bn_block, upsample_custom1_block


class bicubic_interp():
    def __init__(self, lambda_scale_factor, T, recon_img_size= None): # T, recon_img_size will not be needed
        self.lambda_scale_factor= lambda_scale_factor

    def __call__(self, yt):
        for _ in range(self.lambda_scale_factor-1): # upscaling
            yt= F.interpolate(yt, scale_factor= 2, mode='bicubic', align_corners=False)
        return yt
    
class learnable_transpose_conv(nn.Module):
    def __init__(self, lambda_scale_factor, T, recon_img_size= None): # recon_img_size will not be needed
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
    
class custom_v1(nn.Module):
    """
    Idea: each spatial location with c-channel values will be upsampled to corresponding region of upsampled image through fully 
    connected layers. Throughout the upsampling, number of channels will remain at T. 
    Therefore one pixel (with all channels) of yt will be inputted to linear layer which have inputs of 
    :: in_channels: T, out_channels: (upscale_factor**2)*T.
    There are (yt_size*yt_size) pixels in yt. Therefore, there will be (yt_size*yt_size) number of linear layers.
    """
    def __init__(self, lambda_scale_factor, T, recon_img_size= 64):
        super(custom_v1, self).__init__()

        self.lambda_scale_factor = lambda_scale_factor
        self.T= T
        self.recon_img_size= recon_img_size
        self.upscale_factor= 2**(self.lambda_scale_factor-1)
        self.yt_img_size= self.recon_img_size//self.upscale_factor
        
        self.upsample_blocks: nn.ModuleList[upsample_custom1_block] = nn.ModuleList()

        for _ in range(self.yt_img_size**2):
            self.upsample_blocks.append(upsample_custom1_block(self.T , self.T, upscale_factor=self.upscale_factor))

    def forward(self, x):
        x= x.view(-1, self.T, self.yt_img_size, self.yt_img_size)

        device= x.device
        output = torch.zeros(x.shape[0], self.T, self.recon_img_size, self.recon_img_size).to(device)
        
        for i in range(self.yt_img_size):
            for j in range(self.yt_img_size):
                x_patch = self.upsample_blocks[i*self.yt_img_size+j](x[:, :, i, j]) #shape: (batch_size, T, self.upscale_factor, self.upscale_factor)
                output[:,:,i*self.upscale_factor:(i+1)*self.upscale_factor, j*self.upscale_factor:(j+1)*self.upscale_factor] = x_patch
        return output
                