
import torch
from torch import nn
from modules.custom_activations import sigmoid_custom

class modelH_class(nn.Module):
    def __init__(self, T, img_size, preprocess_H_weights=None, complex_init=False, device='cpu', initialization_bias=0, activation= None, init_method = 'randn', enable_train=True):
        super(modelH_class, self).__init__()
        if init_method == 'randn':
            if complex_init:init_weights = torch.randn((1, T, img_size, img_size), dtype= torch.cfloat)
            else:init_weights = torch.randn((1, T, img_size, img_size), dtype= torch.float32)
        elif init_method == 'fft':
            init_weights = torch.randn((1, T, img_size, img_size), dtype= torch.float32)
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