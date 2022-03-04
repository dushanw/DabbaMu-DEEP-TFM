from modules.models.decoder import *
from modules.models.decoder_upsampling_nets import *
from modules.models.decoder_upsampling_nets_experimental import *
from modules.models.decoder_support_blocks import conv_bn_block

from torch import nn

class Customv2UpGenv1Decoder(nn.Module):
    def __init__(self, opts, prev_cfg):
        super(Customv2UpGenv1Decoder, self).__init__() 
        
        print('Initializing reconstruction network ... : Customv2UpGenv1Decoder')
        
        T= prev_cfg.MODEL.MODEL_H.T
        lambda_scale_factor= prev_cfg.MODEL.MODEL_A.lambda_scale_factor
        img_size= prev_cfg.DATASET.img_size
        decoder_upsample_init_method= prev_cfg.MODEL.MODEL_DECODER.upsample_net_init_method
        custom_upsampling_bias= prev_cfg.MODEL.MODEL_DECODER.custom_upsampling_bias
        img_channels= prev_cfg.DATASET.img_channels
        channel_list= prev_cfg.MODEL.MODEL_DECODER.channel_list
        last_activation= prev_cfg.MODEL.MODEL_DECODER.last_activation
        
        if T!=1:
            upsample_postproc_block= nn.Sequential(conv_bn_block(in_channels= 1, out_channels= T//2, kernel_size= 3, padding= 1, stride=1),
                                                   conv_bn_block(in_channels= T//2, out_channels= T, kernel_size= 3, padding= 1, stride=1))
        else:
            upsample_postproc_block= nn.Sequential(conv_bn_block(in_channels= 1, out_channels= T, kernel_size= 3, padding= 1, stride=1),
                                                   conv_bn_block(in_channels= T, out_channels= T, kernel_size= 3, padding= 1, stride=1))

        self.decoder_upsample_net= custom_v2(lambda_scale_factor= lambda_scale_factor, 
                                                  T= T, 
                                                  recon_img_size= img_size, 
                                                  init_method= decoder_upsample_init_method, 
                                                  custom_upsampling_bias= custom_upsampling_bias, 
                                                  upsample_postproc_block= upsample_postproc_block)

        self.reconstruction_net= genv1(T, img_size, img_channels, channel_list, last_activation)

        
    def forward(self, yt_down):
        yt_up = self.decoder_upsample_net(yt_down)
        X_hat = self.reconstruction_net(yt_up)
        return X_hat