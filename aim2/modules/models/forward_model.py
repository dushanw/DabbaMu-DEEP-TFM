
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
        lambda_up= torch.zeros((X.shape[0], H.shape[1], X.shape[2], X.shape[3])).to(self.device)
        for t in range(H.shape[1]):
            lambda_up[:, t:t+1, :, :]= self.forward_model_singleH(X, H[:,t:t+1,:,:])
            
        lambda_down= lambda_up
        for _ in range(self.scale_factor-1): # downscaling
            lambda_down = F.avg_pool2d(lambda_down, kernel_size= 2, stride=2, padding=0)*4
            
            #if _==0:lambda_down = F.avg_pool2d(lambda_down, kernel_size= 2, stride=2, padding=0)*4
            #else:lambda_down = F.avg_pool2d(lambda_down, kernel_size= 2, stride=2, padding=0)*4

        if self.noise==True: # add noise
            z= torch.randn_like(lambda_down)         
            yt_down = lambda_down + (self.shift_lambda_real/ self.rotation_lambda) + torch.sqrt(lambda_down/self.rotation_lambda + self.shift_lambda_real/(self.rotation_lambda**2))*z 
        else:
            yt_down=lambda_down
            
        #for _ in range(self.scale_factor-1): # upscaling
        #    yt= F.interpolate(yt, scale_factor= 2, mode='bicubic')
            
        
        return lambda_up, yt_down ## returns downsampled image with/ without noise