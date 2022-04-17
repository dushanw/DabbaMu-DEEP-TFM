
import torch
from torch.nn import functional as F

def fft_conv2d(X, filter_, **args): # X.shape: (b, 1, m, m), filter_.shape: (1, 1, f, f) -> similar to standard conv [note: f> m]
    #print('fft conv used !!!')
    X= X[:, 0]
    filter_= filter_[:, 0]
    
    _, m, _ = X.shape
    _, f, _ = filter_.shape
    a, b, c, d= (f-m)//2 + 1, (f-m)//2, (f-m)//2 + 1, (f-m)//2
    
    X= torch.nn.functional.pad(X, (a, b, c, d)) # make X.shape= filter_.shape [note: filter_ is larger than image]
    X = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(X)))
    filter_ = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(filter_)))
    X = X * filter_
    X = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(X)))
    
    X= X[:, c:-d, a:-b] ## c+1: -d+1 instead c: -d -> because, freq-domain mult. shift the image (DO NOT KNOW WHY??)
    return X.abs().unsqueeze(dim= 1) # shape: (b, 1, m, m)



def special_Conv2d(X, filter_, padding, conv_func= None): #X.shape: (b, T, m, m), filter.shape: (1, 1, f, f) 
    if conv_func ==None: conv_func= F.conv2d
    
    b, T, m, _ = X.shape
    out = conv_func(X.reshape(b*T, 1, m, m), filter_, padding= padding)
    return out.reshape(b, T, m, m)

class modelA_class:
    def __init__(self, sPSF, exPSF, emPSF, noise=False, device = 'cpu', scale_factor=1, rotation_lambda=1, shift_lambda_real=0, readnoise_std= 0):
        self.sPSF= sPSF
        self.exPSF= exPSF
        self.emPSF= emPSF
        self.device= device
        self.noise= noise # boolean
        self.readnoise_std= readnoise_std
        
        self.scale_factor= scale_factor
        
        self.shift_lambda_real= shift_lambda_real # to shift lambda_real st lambda_real>10
        self.rotation_lambda= rotation_lambda # = lambda_real/ lambda
    '''    
    def forward_model_singleH(self, X, Ht): #X: (m, 1, Nx, Ny), Ht: (1, 1, Nx, Ny), sPSF: (N1, N1), exPSF: (N2, N2)
        sPSF= self.sPSF
        exPSF= self.exPSF
        emPSF= self.emPSF
        
        padding_spsf = (sPSF.shape[0]-1)//2
        padding_expsf = (exPSF.shape[0]-1)//2
        padding_empsf = (emPSF.shape[0]-1)//2

        sPSF= sPSF.view(1, 1, sPSF.shape[0], sPSF.shape[1]) 
        exPSF= exPSF.view(1, 1, exPSF.shape[0], exPSF.shape[1])
        emPSF= emPSF.view(1, 1, emPSF.shape[0], emPSF.shape[1])

        A1= F.conv2d(Ht, exPSF, padding= padding_expsf)*X
        #yt= F.conv2d(A1, sPSF, padding= padding_spsf)
        
        A2= F.conv2d(A1, sPSF, padding= padding_spsf)
        yt= F.conv2d(A2, emPSF, padding= padding_empsf)
        #print('emPSF is used !!!')
        return yt
    '''
    
    def forward_model_allH(self, X, Ht, conv_func): #X: (m, 1, Nx, Ny), Ht: (1, T, Nx, Ny), sPSF: (N1, N1), exPSF: (N2, N2)
        sPSF= self.sPSF
        exPSF= self.exPSF
        emPSF= self.emPSF

        padding_spsf = (sPSF.shape[0]-1)//2
        padding_expsf = (exPSF.shape[0]-1)//2
        padding_empsf = (emPSF.shape[0]-1)//2

        sPSF= sPSF.view(1, 1, sPSF.shape[0], sPSF.shape[1]) 
        exPSF= exPSF.view(1, 1, exPSF.shape[0], exPSF.shape[1])
        emPSF= emPSF.view(1, 1, emPSF.shape[0], emPSF.shape[1])

        A1= special_Conv2d(Ht, exPSF, padding= padding_expsf, conv_func= conv_func)*X
        A2= special_Conv2d(A1, sPSF, padding= padding_spsf, conv_func= conv_func)
        yt= special_Conv2d(A2, emPSF, padding= padding_empsf, conv_func= conv_func)
        #print('faster code !!!')
        return yt


    def compute_yt(self, X, H): 
        '''  # old code: with for loops
        lambda_up= torch.zeros((X.shape[0], H.shape[1], X.shape[2], X.shape[3])).to(self.device)
        for t in range(H.shape[1]):
            lambda_up[:, t:t+1, :, :]= self.forward_model_singleH(X, H[:,t:t+1,:,:])
        '''
        lambda_up= self.forward_model_allH(X, H, conv_func= fft_conv2d)
        
        lambda_down= lambda_up
        for _ in range(self.scale_factor-1): # downscaling
            lambda_down = F.avg_pool2d(lambda_down, kernel_size= 2, stride=2, padding=0)*4
            
            #if _==0:lambda_down = F.avg_pool2d(lambda_down, kernel_size= 2, stride=2, padding=0)*4
            #else:lambda_down = F.avg_pool2d(lambda_down, kernel_size= 2, stride=2, padding=0)*4

        if self.noise==True: # add noise
            z= torch.randn_like(lambda_down)   
            read_noise= self.readnoise_std * torch.randn_like(lambda_down) 
            
            yt_down = lambda_down + (self.shift_lambda_real/ self.rotation_lambda) + torch.sqrt(lambda_down/self.rotation_lambda + self.shift_lambda_real/(self.rotation_lambda**2))*z  + read_noise/ self.rotation_lambda
        else:
            yt_down=lambda_down
            
        #for _ in range(self.scale_factor-1): # upscaling
        #    yt= F.interpolate(yt, scale_factor= 2, mode='bicubic')
            
        
        return lambda_up, yt_down ## returns downsampled image with/ without noise

