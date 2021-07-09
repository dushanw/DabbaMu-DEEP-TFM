import torch
from torch.nn import functional as F
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from ignite.metrics import SSIM as SSIM_IGNITE


def mse_distance(X_hat, X):
    return F.mse_loss(X_hat, X).item()


def ssim_custom_lib(X_hat, X):
    data_range= 1.0
    return ssim(X, X_hat, data_range=data_range, size_average=False).mean().item()
    
def ssim_ignite(X_hat, X, k= 11):
    metric = SSIM_IGNITE(data_range = 1.0, kernel_size= (k,k))
    metric.update((X_hat, X))
    return metric.compute().item()
    