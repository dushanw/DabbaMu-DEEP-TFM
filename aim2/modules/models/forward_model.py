
import torch
from torch.nn import functional as F

class modelA_class:
    def __init__(self, sPSF, exPSF, noise=False, device = 'cpu', scale_factor=1, rotation_lambda=1, shift_lambda_real=0):
        self.sPSF= sPSF
        self.exPSF= exPSF
        self.device= device
        self.noise= noise # boolean
        
        self.scale_factor= scale_factor
        
        self.shift_lambda_real= shift_lambda_real # to shift lambda_real st lambda_real>10
        self.rotation_lambda= rotation_lambda # = lambda_real/ lambda
        
    def forward_model_singleH(self, X, Ht): #X: (m, 1, Nx, Ny), Ht: (1, 1, Nx, Ny), sPSF: (N1, N1), exPSF: (N2, N2)
        sPSF= self.sPSF
        exPSF= self.exPSF
        
        padding_spsf = (sPSF.shape[0]-1)//2
        padding_expsf = (exPSF.shape[0]-1)//2

        sPSF= sPSF.view(1, 1, sPSF.shape[0], sPSF.shape[1]) 
        exPSF= exPSF.view(1, 1, exPSF.shape[0], exPSF.shape[1])

        A1= F.conv2d(Ht, exPSF, padding= padding_expsf)*X
        yt= F.conv2d(A1, sPSF, padding= padding_spsf)
        return yt

    def compute_yt(self, X, H):    
        lambda_= torch.zeros((X.shape[0], H.shape[1], X.shape[2], X.shape[3])).to(self.device)
        for t in range(H.shape[1]):
            lambda_[:, t:t+1, :, :]= self.forward_model_singleH(X, H[:,t:t+1,:,:])
            
        for _ in range(self.scale_factor-1): # downscaling
            lambda_ = F.avg_pool2d(lambda_, kernel_size= 2, stride=2, padding=0)*4

        if self.noise==True: # add noise
            z= torch.randn_like(lambda_)         
            yt = lambda_ + (self.shift_lambda_real/ self.rotation_lambda) + torch.sqrt(lambda_/self.rotation_lambda + self.shift_lambda_real/(self.rotation_lambda**2))*z 
        else:yt=lambda_
            
        #for _ in range(self.scale_factor-1): # upscaling
        #    yt= F.interpolate(yt, scale_factor= 2, mode='bicubic')
            
        
        return yt ## returns downsampled image with/ without noise