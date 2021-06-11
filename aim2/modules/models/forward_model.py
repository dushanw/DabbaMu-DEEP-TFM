
import torch
from torch.nn import functional as F

def forward_model_singleH(X, Ht, sPSF, exPSF): #X: (m, 1, Nx, Ny), Ht: (1, 1, Nx, Ny), sPSF: (N1, N1), exPSF: (N2, N2)
    padding_spsf = (sPSF.shape[0]-1)//2
    padding_expsf = (exPSF.shape[0]-1)//2

    sPSF= sPSF.view(1, 1, sPSF.shape[0], sPSF.shape[1]) 
    exPSF= exPSF.view(1, 1, exPSF.shape[0], exPSF.shape[1])
    
    A1= F.conv2d(Ht, exPSF, padding= padding_expsf)*X
    yt= F.conv2d(A1, sPSF, padding= padding_spsf)
    return yt

def forward_modelA(X, H, sPSF, exPSF, device, noise=False, K=1, scale_factor=1):    
    lambda_= torch.zeros((X.shape[0], H.shape[1], X.shape[2], X.shape[3])).to(device)
    for t in range(H.shape[1]):
        lambda_[:, t:t+1, :, :]= forward_model_singleH(X, H[:,t:t+1,:,:], sPSF, exPSF)
        
        
    if noise==True:
        for _ in range(scale_factor-1):
            lambda_ = F.max_pool2d(lambda_, kernel_size= 2, stride=2, padding=0)
        print(f"downscaled lambda : {lambda_.shape}")
        
        z= torch.randn_like(lambda_) 
        yt = lambda_ + torch.sqrt(lambda_/K)*z 
        yt= F.interpolate(yt, scale_factor= 2**(scale_factor-1))
        print(f"upscaled yt : {yt.shape}")
    else:yt=lambda_
    return yt