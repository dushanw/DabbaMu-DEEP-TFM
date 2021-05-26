
import torch
from torch import nn
from modules.custom_activations import sigmoid_custom

class modelH(nn.Module):
    def __init__(self, T, img_size, preprocess_H_weights=None, complex_init=False, device='cpu', initialization_bias=0, activation= None):
        super(modelH, self).__init__()
        if complex_init:
            self.weights = nn.Parameter(initialization_bias+ torch.randn((1, T, img_size, img_size), dtype= torch.cfloat))
        else:
            self.weights = nn.Parameter(initialization_bias+ torch.randn((1, T, img_size, img_size), dtype= torch.float32))
        self.const = torch.ones_like(self.weights).to(device)
        self.preprocess_weights= preprocess_H_weights
        
        self.activation = activation

    def forward(self, m=1):
        x = self.weights*self.const
        
        if self.preprocess_weights !=None:
            x= self.preprocess_weights(x)
            
        if self.activation!=None:x=self.activation(x, m)
        else:x= sigmoid_custom(x, m)
            
        return x