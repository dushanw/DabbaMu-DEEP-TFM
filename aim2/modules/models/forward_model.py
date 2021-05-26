
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

def forward_modelA(X, H, sPSF, exPSF, device, noise=False):    
    output= torch.zeros((X.shape[0], H.shape[1], X.shape[2], X.shape[3])).to(device)
    for t in range(H.shape[1]):
        output[:, t:t+1, :, :]= forward_model_singleH(X, H[:,t:t+1,:,:], sPSF, exPSF)
    if noise==True:
        z= torch.randn_like(output)   
        output = output + torch.sqrt(output)*z # sample_from_normal which is similar to poisson         
    return output