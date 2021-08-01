
import torch
from torch import nn
from modules.custom_activations import sigmoid_custom
from modules.models.initialize_H_weights import *

class modelH_class(nn.Module):
    def __init__(self, T, img_size, preprocess_H_weights=None, device='cpu', initialization_bias=0, activation= None, init_method = 'randn', enable_train=True, lambda_scale_factor=1):
        super(modelH_class, self).__init__()
        if init_method == 'randn':
            init_weights = torch.randn((1, T, img_size, img_size), dtype= torch.float32)
            
        elif init_method == 'randn_complex':
            init_weights = torch.randn((1, T, img_size, img_size), dtype= torch.cfloat)
            
        elif init_method == 'randn_FourierBased':
            init_weights = torch.randn((1, T, img_size, img_size), dtype= torch.float32)
            init_weights = torch.fft.fft2(init_weights)
            
        elif init_method == 'uniformones_FourierBased':
            init_weights = torch.ones((1, T, img_size, img_size), dtype= torch.float32)
            init_weights = torch.fft.fft2(init_weights)
            
        elif init_method == 'canonical_FourierBased':
            init_weights = canonical(T, img_size, lambda_scale_factor) # returns {0, 1} torch tensor with shape: (1, T, img_size, img_size)
            init_weights = torch.fft.fft2(init_weights)
        
        elif init_method == 'hadamard_FourierBased':
            init_weights = hadamard_norescale(T, img_size, lambda_scale_factor) # returns {0, 1} torch tensor with shape: (1, T, img_size, img_size)
            init_weights = torch.fft.fft2(init_weights)
            
        
        self.T= T
        self.weights = nn.Parameter(initialization_bias+ init_weights)
        self.const = torch.ones_like(self.weights).to(device)
        self.preprocess_weights= preprocess_H_weights
        self.activation = activation
        self.enable_train= enable_train
        
        if not self.enable_train:self.weights.requires_grad= False
        else:self.weights.requires_grad= True
            

    def forward(self, m=1):
        x = self.weights*self.const
        
        if self.preprocess_weights !=None:
            x= self.preprocess_weights(x)
            
        if self.activation!=None:x=self.activation(x, m)
        else:x= sigmoid_custom(x, m)
            
        return x