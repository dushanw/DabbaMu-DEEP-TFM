import torch
from torch.nn import functional as F
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from ignite.metrics import SSIM as SSIM_IGNITE
from ignite.metrics import PSNR

import modules.pytorch_colors as colors


def mse_distance(X_hat, X):
    return F.mse_loss(X_hat, X).item()

def l1_distance(X_hat, X):
    return F.l1_loss(X_hat, X).item()


def ssim_custom_lib(X_hat, X):
    data_range= 1.0
    return ssim(X, X_hat, data_range=data_range, size_average=False).mean().item()
    
def ssim_ignite(X_hat, X, k= 11):
    metric = SSIM_IGNITE(data_range = 1.0, kernel_size= (k,k))
    metric.update((X_hat, X))
    return metric.compute().item()


def get_y_channel(output):
    y_pred, y = output
    y_pred = colors.rgb_to_ycbcr(y_pred)[:,0,:,:] # select y-channel
    y = colors.rgb_to_ycbcr(y)[:,0,:,:] # select y-channel
    return y_pred, y

def psnr_luminance(X_hat, X):
    metric = PSNR(data_range=219, output_transform=get_y_channel) #(data_range= 235- 16= 219) ->  https://scikit-image.org/docs/dev/api/skimage.color.html#rgb2ycbcr 
    metric.update((X_hat, X))
    return metric.compute().item()
    
def ssim_luminance(X_hat, X, k= 11):
    metric = SSIM_IGNITE(data_range = 219, kernel_size= (k,k), output_transform=get_y_channel) #(data_range= 235- 16= 219) ->  https://scikit-image.org/docs/dev/api/skimage.color.html#rgb2ycbcr 
    metric.update((X_hat, X))
    return metric.compute().item()
    
    