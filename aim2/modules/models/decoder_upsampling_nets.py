import torch
from torch import nn
from torch.nn import functional as F
from modules.models.decoder_support_blocks import upsample_transconv_relu_bn_block, upsample_custom1_block


class bicubic_interp():
    def __init__(self, **kwargs): # T, recon_img_size will not be needed
        self.lambda_scale_factor= kwargs['lambda_scale_factor']

    def __call__(self, yt):
        for _ in range(self.lambda_scale_factor-1): # upscaling
            yt= F.interpolate(yt, scale_factor= 2, mode='bicubic', align_corners=False)
        return yt
    
class learnable_transpose_conv(nn.Module):
    def __init__(self, **kwargs): # recon_img_size will not be needed
        super(learnable_transpose_conv, self).__init__()
        self.lambda_scale_factor= kwargs['lambda_scale_factor']
        self.T= kwargs['T']
        
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
    def __init__(self, **kwargs):
        super(custom_v1, self).__init__()

        self.lambda_scale_factor = kwargs['lambda_scale_factor']
        self.T= kwargs['T']
        self.recon_img_size= kwargs['recon_img_size']
        
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
                
        
class custom_v2(nn.Module):
    """
    Idea: the same concept in "custom_v1" but implemented in very much efficient way
    """
    def __init__(self, **kwargs):
        super(custom_v2, self).__init__()
        
        self.lambda_scale_factor = kwargs['lambda_scale_factor']
        self.T= kwargs['T']
        self.recon_img_size= kwargs['recon_img_size']
        self.init_method=  kwargs['init_method']
        
        self.upscale_factor= 2**(self.lambda_scale_factor-1)
        self.yt_img_size= self.recon_img_size//self.upscale_factor
        self.expected_out_channels= self.T * self.upscale_factor**2

        self.weights= nn.Parameter(torch.randn(self.yt_img_size* self.yt_img_size, self.T, self.expected_out_channels), requires_grad= True)
        self.biases= nn.Parameter(torch.randn(self.yt_img_size* self.yt_img_size, 1, self.expected_out_channels), requires_grad= True)
        
        if self.init_method== 'linear_default':
            stdv = 1. / math.sqrt(self.weights.size(1))
            self.weights.data.uniform_(-stdv, stdv)
            self.biases.data.uniform_(-stdv, stdv)

        elif self.init_method=='xavier_normal':
            torch.nn.init.xavier_normal_(self.weights)
            torch.nn.init.zeros_(self.biases)

        elif self.init_method=='randn':
            pass # default is this
        
    def forward(self, yt):
        batch_size= yt.shape[0]
        yt_input = yt.view(batch_size, self.T, self.yt_img_size, self.yt_img_size)

        yt= yt_input.reshape(batch_size, self.T, self.yt_img_size* self.yt_img_size).permute(2, 0, 1) #shape: (n_pixels_in_yt, batch_size, self.T)

        yt_upsample= torch.matmul(yt, self.weights) + self.biases  #shape: (yt_img_size**2, batch_size, T*upscale_factor**2)
        yt_upsample = yt_upsample.view(self.yt_img_size, self.yt_img_size, batch_size, self.T, self.upscale_factor, self.upscale_factor).permute(2, 3, 0, 1, 4, 5)
        output= self.reshape_special(yt_upsample)

        return output
                
    def reshape_special(self, a): 
        '''
        input
        ----
        a: torch.tensor
            Shape `(batch_size, T, yt_img_size1, yt_img_size2, upscale_factor1, upscale_factor2)`
        return
        -----
        out: torch.tensor
            Shape `(batch_size, T, yt_img_size1*upscale_factor1, yt_img_size2*upscale_factor2)` 
            Description: Direct reshape cant use for this.
        '''

        a_dash = a.permute(0, 1, 2, 3, 5, 4) # Shape `(batch_size, T, yt_img_size1, yt_img_size2, upscale_factor2, upscale_factor1)`
        batch_size, T, yt_img_size1, yt_img_size2, upscale_factor2, upscale_factor1= a_dash.shape

        b  = a_dash.reshape(batch_size, T, yt_img_size1, yt_img_size2*upscale_factor2, upscale_factor1)  # Shape `(batch_size, T, yt_img_size1, yt_img_size2*upscale_factor2, upscale_factor1)`
        b_dash = b.permute(0, 1, 2, 4,3)  # Shape `(batch_size, T, yt_img_size1, upscale_factor1, yt_img_size2*upscale_factor2)`
        out= b_dash.reshape(batch_size, T, yt_img_size1*upscale_factor1, yt_img_size2*upscale_factor2)  # Shape `(batch_size, T, yt_img_size1*upscale_factor1, yt_img_size2*upscale_factor2)`

        return out